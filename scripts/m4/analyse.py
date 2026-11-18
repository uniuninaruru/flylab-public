"""Turn finished M4 replicate groups into committed evidence files (PR-04).

    uv run python scripts/m4/analyse.py calibration --group-dir runs/m4_calibration_wsyn_rate \\
        --out docs/acceptance/evidence/m4-calibration-wsyn.json
    uv run python scripts/m4/analyse.py export --group-dir runs/<group_id> \\
        --out docs/acceptance/evidence/m4-<group_id>.json

`export` copies `group_stats.json` and adds the member list (run ids, config hashes, code commit)
from `group_manifest.json`, so the evidence file names exactly which runs it summarises. Run
directories themselves stay under runs/ (git-ignored).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from flylab.experiments.calibration import calibrate_group_dirs


def _load(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return out


def _provenance(gd: Path) -> dict[str, Any]:
    man = _load(gd / "group_manifest.json")
    return {
        "group_manifest": {
            k: man[k]
            for k in (
                "group_id",
                "created_utc",
                "flylab_version",
                "git",
                "base_config",
                "base_config_sha256",
                "group_config_sha256",
            )
        }
        | {k: man[k] for k in ("git_note", "members_code_state", "member_commits") if k in man},
        "members": [
            {"run_id": m["run_id"], "config_sha256": m["config_sha256"]} for m in man["members"]
        ],
        "group_yaml": (gd / "group.yaml").read_text(encoding="utf-8"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("calibration")
    c.add_argument("--group-dir", type=Path, nargs="+", required=True, help="coarse + refinement")
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--stimulus", default="lb3c_left")
    c.add_argument("--readouts", nargs="+", default=["mn9_left", "mn9_right"])
    e = sub.add_parser("export")
    e.add_argument("--group-dir", type=Path, required=True)
    e.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.cmd == "calibration":
        res = {
            "schema_version": 1,
            "kind": "flylab_m4_calibration_evidence",
            "by_readout": {
                r: calibrate_group_dirs(a.group_dir, readout=r, stimulus=a.stimulus)
                for r in a.readouts
            },
            "provenance": {str(d): _provenance(d) for d in a.group_dir},
        }
    else:
        res = _load(a.group_dir / "group_stats.json") | {"provenance": _provenance(a.group_dir)}
    a.out.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {a.out} ({a.out.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
