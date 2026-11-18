# Tiny synthetic graph fixture

**Not biological data.** Body ids `900001`–`900007` are made up. Signs are chosen by hand
to exercise edge cases. They are not transmitter predictions.

Hand-computed expectations (checked by `tests/unit/test_graph.py`):

| Quantity | Value | Why |
|---|---|---|
| neurons | 7 | rows in `neurons.csv` (written out of order on purpose) |
| edge rows | 8 | rows in `edges.csv` |
| unique directed pairs | 7 | `900002→900004` appears twice |
| total synapses | 25 | 5+2+3+4+1+2+1+7 |
| `900002→900004` after `sum` | 5 | 3 + 2, both sign −1 |
| self-loops | 1 | `900005→900005` |
| unknown-sign edges | 1 | `900006→900001` (sign 0) |
| isolated neurons | 1 | `900007` has no edges |

Index map (body ids sorted ascending): `900001→0`, `900002→1`, … `900007→6`.
