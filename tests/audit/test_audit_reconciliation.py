"""Regression test for an audit finding: the M1 count reconciliation must stay written down.

Spec 11.2 M1: "counts and discrepancy reconciled/documented". At 67dc887,
docs/DATASET_COUNT_RECONCILIATION.md said "Status: OPEN" and named none of the importer's
node-set rules (issue below). 7e08246 documented them; this test keeps every rule in
`NODE_SETS` documented there. Sizes reproduced independently in this audit from the public
annotation file: superclass_annotated 166,700; traced 165,122; flybrain_compat 167,106.
"""

from __future__ import annotations

from pathlib import Path

from flylab.data.malecns import NODE_SETS

ISSUE = "https://github.com/uniuninaruru/flylab/issues/8"
DOC = Path(__file__).resolve().parents[2] / "docs" / "DATASET_COUNT_RECONCILIATION.md"


def test_reconciliation_doc_names_every_node_set_rule() -> None:
    text = DOC.read_text(encoding="utf-8")
    missing = [rule for rule in NODE_SETS if rule not in text]
    assert missing == [], f"undocumented node-set rule(s), see {ISSUE}"
