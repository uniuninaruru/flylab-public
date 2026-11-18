"""Reproduce the upstream-data checks of docs/audit/AUDIT_2026-11.md.

Run from the repository root with the project environment (`uv run python ...`). Only public
objects of gs://flyem-male-cns are read, over HTTPS, without credentials. Nothing is written
inside the repository unless you point an output directory there (don't: data is gitignored
for a reason).

    registry                    HEAD every registered file; compare with flylab.data.sources
    ipc-rows NAME               exact row count of a remote Feather file from its batch headers
    node-sets RAW_DIR           node-set sizes and composition from the annotation file
    sample RAW_DIR OUT_DIR      every 10th batch of the weights file + the two small files
    direction PROCESSED_DIR     direction checks on a processed graph built from the sample
    memory SAMPLE_DIR           per-stage peak allocations of stream_edges / build_graph_indexed

RAW_DIR must hold the annotation and transmitter Feather files (14 MB + 43 MB), e.g. from
`flylab data fetch --files body-annotations-male-cns-v1.0-minconf-0.5.feather
body-neurotransmitters-male-cns-v1.0.feather`.
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
import struct
import sys
import tracemalloc
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

from flylab.data import malecns as mc
from flylab.data.graph import DuplicateRule, NeuronIndex, build_graph_indexed
from flylab.data.processed import load_graph
from flylab.data.sources import MALE_CNS_V1_0 as RELEASE


def _get(url: str, start: int, n: int) -> bytes:
    rng = {"Range": f"bytes={start}-{start + n - 1}"}
    req = urllib.request.Request(url, headers=rng)  # noqa: S310 - fixed https prefix
    with urllib.request.urlopen(req, timeout=120) as r:  # noqa: S310 - fixed https prefix
        data: bytes = r.read()
    return data


class RangeFile(io.RawIOBase):
    """Seekable remote file: pyarrow fetches only the footer and the batches it reads."""

    def __init__(self, url: str, size: int) -> None:
        self.url, self.size, self.pos, self.fetched = url, size, 0, 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, off: int, whence: int = 0) -> int:
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def readinto(self, b: Any) -> int:
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        data = _get(self.url, self.pos, n)
        b[: len(data)] = data
        self.pos += len(data)
        self.fetched += len(data)
        return len(data)


def cmd_registry(_: argparse.Namespace) -> None:
    ok = True
    for f in RELEASE.files:
        req = urllib.request.Request(RELEASE.url(f.name), method="HEAD")  # noqa: S310
        with urllib.request.urlopen(req, timeout=60) as r:  # noqa: S310
            hashes = dict(h.split("=", 1) for h in r.headers.get_all("x-goog-hash"))
            got = (int(r.headers["Content-Length"]), hashes["md5"], hashes["crc32c"])
            got_lm = r.headers["Last-Modified"]
        match = got == (f.size_bytes, f.md5_base64, f.crc32c_base64) and got_lm == f.last_modified
        ok &= match
        print(f"{'MATCH' if match else 'DIFF '} {f.name} size={got[0]} md5={got[1]}")
    print("ALL_MATCH" if ok else "MISMATCH")


def _table(buf: bytes, pos: int) -> Any:
    """Flatbuffer table at `pos`: returns field(i) -> offset of field i within the table."""
    vt = pos - struct.unpack_from("<i", buf, pos)[0]
    vlen = struct.unpack_from("<H", buf, vt)[0]

    def field(i: int) -> int:
        o = 4 + 2 * i
        return 0 if o >= vlen else int(struct.unpack_from("<H", buf, vt + o)[0])

    return field


def cmd_ipc_rows(args: argparse.Namespace) -> None:
    """Arrow IPC file: footer -> record-batch blocks -> each message's RecordBatch.length."""
    spec = RELEASE.file(args.name)
    url, size = RELEASE.url(spec.name), spec.size_bytes
    tail = _get(url, size - 10, 10)
    if tail[-6:] != b"ARROW1":
        raise SystemExit("not an Arrow IPC file")
    flen = struct.unpack("<i", tail[:4])[0]
    fb = _get(url, size - 10 - flen, flen)
    root = struct.unpack_from("<I", fb, 0)[0]
    vpos = root + _table(fb, root)(3)  # Footer.recordBatches
    vpos += struct.unpack_from("<I", fb, vpos)[0]
    n = struct.unpack_from("<I", fb, vpos)[0]
    blocks = [struct.unpack_from("<qi4xq", fb, vpos + 4 + 24 * i) for i in range(n)]
    lengths = []
    for off, mlen, _body in blocks:
        m = _get(url, off, mlen)
        msg = m[8:] if m[:4] == b"\xff\xff\xff\xff" else m[4:]
        r = struct.unpack_from("<I", msg, 0)[0]
        mf = _table(msg, r)
        hp = r + mf(2)  # Message.header (a RecordBatch table)
        hp += struct.unpack_from("<I", msg, hp)[0]
        lengths.append(struct.unpack_from("<q", msg, hp + _table(msg, hp)(0))[0])
    print(f"{spec.name}: batches={n} rows={sum(lengths)}", end=" ")
    print(f"min/max batch rows={min(lengths)}/{max(lengths)}")


def cmd_node_sets(args: argparse.Namespace) -> None:
    ann = feather.read_table(str(Path(args.raw_dir) / mc.ANNOTATIONS))
    masks = mc.node_set_masks(ann)
    ids = ann["bodyId"].to_numpy()
    status = ann["status"].to_pylist()
    types = ann["type"].to_pylist()
    photo = np.array([t in set(mc.PHOTORECEPTOR_TYPES) for t in types])
    null_status = np.array([s is None for s in status])
    print("rows", ann.num_rows, "unique bodyId", np.unique(ids).size)
    print({k: int(v.sum()) for k, v in masks.items()})
    sup, traced = masks["superclass_annotated"], masks["traced"]
    print("photoreceptor-typed and not Traced:", int((photo & ~traced).sum()))
    print("default set by status:", Counter(np.array(status, dtype=object)[sup]).most_common())
    print("Traced without superclass:", int((traced & ~sup).sum()))
    by_type = Counter(np.array(types, dtype=object)[sup & null_status]).most_common(4)
    print("null-status bodies in default set, by type:", by_type)


def cmd_sample(args: argparse.Namespace) -> None:
    spec = RELEASE.file(mc.WEIGHTS)
    rf = RangeFile(RELEASE.url(spec.name), spec.size_bytes)
    reader = ipc.open_file(pa.PythonFile(rf, mode="r"))
    batches = [reader.get_batch(i) for i in range(0, reader.num_record_batches, 10)]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_batches(batches)
    feather.write_feather(table, str(out / mc.WEIGHTS), compression="lz4", chunksize=65_536)
    for name in (mc.ANNOTATIONS, mc.NEUROTRANSMITTERS):
        shutil.copy(Path(args.raw_dir) / name, out / name)
    print(f"batches={len(batches)} rows={table.num_rows} fetched_bytes={rf.fetched}")


def cmd_direction(args: argparse.Namespace) -> None:
    """If src were not presynaptic, all three ratios below would invert."""
    d = Path(args.processed_dir)
    g = load_graph(d)
    t = pq.read_table(d / "neurons.parquet", columns=["superclass", "type"]).to_pydict()
    sup, typ = t["superclass"], t["type"]
    w = g.synapse_count
    motor = np.array([s in {"vnc_motor", "cb_motor"} for s in sup])
    sensory = np.array([isinstance(s, str) and s.endswith("sensory") for s in sup])
    r16 = np.array([x == "R1-R6" for x in typ])
    lam = np.array([x in {"L1", "L2", "L3"} for x in typ])

    def side(m: Any) -> tuple[int, int]:
        return int(w[m[g.src]].sum()), int(w[m[g.dst]].sum())

    print(json.dumps({
        "motor_synapses_as_src_vs_dst": side(motor),
        "sensory_synapses_as_src_vs_dst": side(sensory),
        "R1R6_to_L1L3_vs_reverse": [int(w[r16[g.src] & lam[g.dst]].sum()),
                                    int(w[lam[g.src] & r16[g.dst]].sum())],
    }))  # fmt: skip


def cmd_memory(args: argparse.Namespace) -> None:
    d = Path(args.sample_dir)
    ann = feather.read_table(str(d / mc.ANNOTATIONS))
    index = NeuronIndex.from_body_ids(
        ann["bodyId"].to_numpy()[mc.node_set_masks(ann)[mc.DEFAULT_NODE_SET]]
    )
    tracemalloc.start()
    s, t, c, st = mc.stream_edges(d / mc.WEIGHTS, index.body_ids)
    held, peak_stream = tracemalloc.get_traced_memory()
    tracemalloc.reset_peak()
    build_graph_indexed(index, s, t, c, dataset_id="audit", duplicate_rule=DuplicateRule.ERROR)
    peak_build = tracemalloc.get_traced_memory()[1]
    k = st["rows_kept"]
    print(f"rows_kept={k} bodies_in_weights={st['bodies_in_weights']}")
    print(f"stream_edges peak={peak_stream / 1e6:.0f} MB held_after={held / 1e6:.0f} MB")
    print(f"build_graph_indexed peak={peak_build / 1e6:.0f} MB ({peak_build / k:.1f} B/kept row)")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("registry").set_defaults(func=cmd_registry)
    s = sub.add_parser("ipc-rows")
    s.add_argument("name", help="registered file name, e.g. " + mc.WEIGHTS)
    s.set_defaults(func=cmd_ipc_rows)
    s = sub.add_parser("node-sets")
    s.add_argument("raw_dir")
    s.set_defaults(func=cmd_node_sets)
    s = sub.add_parser("sample")
    s.add_argument("raw_dir")
    s.add_argument("out_dir")
    s.set_defaults(func=cmd_sample)
    s = sub.add_parser("direction")
    s.add_argument("processed_dir")
    s.set_defaults(func=cmd_direction)
    s = sub.add_parser("memory")
    s.add_argument("sample_dir")
    s.set_defaults(func=cmd_memory)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
