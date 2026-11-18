"""Registry of upstream dataset releases and files.

Every value below was read from the upstream server, not copied from the spec. File sizes,
MD5 and CRC32C come from HTTP HEAD responses of the public Google Cloud Storage bucket on
the date in `verified_on`. Re-check before relying on them; upstream objects can change.

Nothing here downloads data. The importer (PR-01) uses this registry to know what to fetch
and to verify downloads against the upstream hashes.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass

from flylab.provenance.labels import Evidence


@dataclass(frozen=True)
class UpstreamFile:
    name: str
    role: str  # what the file contains, from the official download page
    size_bytes: int
    md5_base64: str  # x-goog-hash md5
    crc32c_base64: str  # x-goog-hash crc32c
    last_modified: str  # HTTP Last-Modified header

    def md5_hex(self) -> str:
        return base64.b64decode(self.md5_base64).hex()


@dataclass(frozen=True)
class DatasetRelease:
    dataset_id: str
    release_date: str
    license_spdx: str
    license_url: str
    attribution: str
    download_page: str
    release_notes: str
    gs_prefix: str
    https_prefix: str
    files: tuple[UpstreamFile, ...]
    verified_on: str
    verified_by: str
    evidence: Evidence

    def url(self, name: str) -> str:
        self.file(name)  # raises KeyError for unregistered names
        return f"{self.https_prefix}/{name}"

    def file(self, name: str) -> UpstreamFile:
        for f in self.files:
            if f.name == name:
                return f
        raise KeyError(f"{name!r} is not a registered file of {self.dataset_id}")


MALE_CNS_V1_0 = DatasetRelease(
    dataset_id="male-cns:v1.0",
    release_date="2026-06-08",
    license_spdx="CC-BY-4.0",
    license_url="https://creativecommons.org/licenses/by/4.0/",
    attribution=(
        "Male CNS connectome v1.0: FlyEM (HHMI Janelia), University of Cambridge "
        "(Dept. of Zoology), MRC Laboratory of Molecular Biology and Google Research. "
        "https://male-cns.janelia.org/ . Licensed CC-BY 4.0."
    ),
    download_page="https://male-cns.janelia.org/download/",
    release_notes="https://male-cns.janelia.org/release/",
    gs_prefix="gs://flyem-male-cns/v1.0/connectome-data/flat-connectome",
    https_prefix=(
        "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome"
    ),
    verified_on="2026-11-04",
    verified_by="HTTP HEAD against storage.googleapis.com (no credentials needed)",
    evidence=Evidence.EMPIRICAL,
    files=(
        UpstreamFile(
            name="body-annotations-male-cns-v1.0-minconf-0.5.feather",
            role="Curated neuron annotations (classes, types, sides; no transmitters)",
            size_bytes=14_483_314,
            md5_base64="UKdxh3DFciDxYLpPQxq4ng==",
            crc32c_base64="vjz9cg==",
            last_modified="Wed, 03 Jun 2026 13:54:38 GMT",
        ),
        UpstreamFile(
            name="body-neurotransmitters-male-cns-v1.0.feather",
            role="Per-neuron aggregate neurotransmitter predictions (not ground truth)",
            size_bytes=43_282_834,
            md5_base64="PYQrEv5cSe763lKNfdJKHw==",
            crc32c_base64="jcpNFg==",
            last_modified="Mon, 08 Jun 2026 05:01:39 GMT",
        ),
        UpstreamFile(
            name="body-stats-male-cns-v1.0-minconf-0.5.feather",
            role="Synapse count summary statistics for all segments with synapses",
            size_bytes=778_062_826,
            md5_base64="QEwzScKFgBSOFoFeuZ84Kg==",
            crc32c_base64="MGOCPQ==",
            last_modified="Wed, 03 Jun 2026 13:54:48 GMT",
        ),
        UpstreamFile(
            name="connectome-weights-male-cns-v1.0-minconf-0.5.feather",
            role="Segment-to-segment connection strengths for all segments (full graph)",
            size_bytes=1_051_241_946,
            md5_base64="8w6dzKJc/QIb8eez2XVZng==",
            crc32c_base64="dKRPVQ==",
            last_modified="Wed, 03 Jun 2026 13:54:47 GMT",
        ),
        UpstreamFile(
            name="syn-points-male-cns-v1.0-minconf-0.5.feather",
            role="Pre/post synapse locations (8 nm voxels), body id, ROI. Not needed for MVP.",
            size_bytes=13_061_489_098,
            md5_base64="xp0IdY3gdYIDXMiENXRJOg==",
            crc32c_base64="1E7gRw==",
            last_modified="Wed, 03 Jun 2026 13:56:31 GMT",
        ),
        UpstreamFile(
            name="syn-partners-male-cns-v1.0-minconf-0.5.feather",
            role="Synaptic partner pairs with body ids and primary neuropil. Not needed for MVP.",
            size_bytes=6_777_179_098,
            md5_base64="WO/PcS+MTU3l8q1R6X3vdg==",
            crc32c_base64="jTlNIA==",
            last_modified="Wed, 03 Jun 2026 13:55:42 GMT",
        ),
        UpstreamFile(
            name="tbar-neurotransmitters-male-cns-v1.0.feather",
            role="Per-presynapse neurotransmitter prediction probabilities. Not needed for MVP.",
            size_bytes=2_651_680_218,
            md5_base64="UbAsEWkGYq7e8o+G05T/DQ==",
            crc32c_base64="RrR5/g==",
            last_modified="Mon, 08 Jun 2026 05:02:07 GMT",
        ),
    ),
)

#: Files the first MVP importer needs (about 1.1 GB together). The rest are optional.
MVP_FILES = (
    "body-annotations-male-cns-v1.0-minconf-0.5.feather",
    "body-neurotransmitters-male-cns-v1.0.feather",
    "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
)

RELEASES: dict[str, DatasetRelease] = {MALE_CNS_V1_0.dataset_id: MALE_CNS_V1_0}
