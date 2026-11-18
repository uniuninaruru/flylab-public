from __future__ import annotations

import base64

import pytest

from flylab.data.sources import MALE_CNS_V1_0, MVP_FILES, RELEASES
from flylab.provenance.labels import Evidence


def test_release_metadata() -> None:
    r = MALE_CNS_V1_0
    assert r.dataset_id == "male-cns:v1.0"
    assert r.release_date == "2026-06-08"  # release page, not the September announcement
    assert r.license_spdx == "CC-BY-4.0"
    assert r.evidence is Evidence.EMPIRICAL
    assert r.verified_on
    assert RELEASES["male-cns:v1.0"] is r


def test_every_file_is_well_formed() -> None:
    assert len(MALE_CNS_V1_0.files) == 7
    for f in MALE_CNS_V1_0.files:
        assert "v1.0" in f.name
        assert f.name.endswith(".feather")
        assert f.size_bytes > 0
        assert len(base64.b64decode(f.md5_base64)) == 16
        assert len(base64.b64decode(f.crc32c_base64)) == 4
        assert len(f.md5_hex()) == 32
        url = MALE_CNS_V1_0.url(f.name)
        assert url.startswith("https://storage.googleapis.com/flyem-male-cns/v1.0/")


def test_mvp_files_registered_and_fit_plan() -> None:
    total = sum(MALE_CNS_V1_0.file(n).size_bytes for n in MVP_FILES)
    assert total < 1.2e9  # weights file is ~1.05 GB; synapse-level tables are excluded


def test_unknown_file_rejected() -> None:
    with pytest.raises(KeyError):
        MALE_CNS_V1_0.url("made-up.feather")
