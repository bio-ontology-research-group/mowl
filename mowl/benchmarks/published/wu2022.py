"""Published success rates of the masked ABox revision task (Wu et al. 2022).

Transcribed from the LaTeX source of arXiv:2211.12006 (v1), the table "Success rate (%) in
four mask rate settings with alpha = 0.8" (label ``tab:result1``): per ontology and mask
rate the success rate of the masked grounding (M), of DF-ALC revised with the hierarchical
loss (D w/HL), of DF-ALC revised with the rule-based loss (D w/RL) and of Logical Tensor
Networks (L). Only the two DF-ALC columns are recorded as published results (mOWL
implements DF-ALC, not LTN); the success rate of the masked grounding and of LTN are in
the note of each row. Values are printed in %; they are stored as fractions.

The paper's success rate counts the axioms of the original ontology; the benchmark of mOWL
counts the subclass, equivalence and disjointness axioms of the released file (see the
regime ``masked-abox-revision``), so the two are comparable in trend, not in the last digit.
The training of the paper: Adam, learning rate 2e-4, early stopping with a tolerance of 10
checks (settings paragraph of the Experiments section); rule-based loss with alpha' = 0.8.
"""

from mowl.benchmarks.entries._refs import WU2022
from mowl.benchmarks.spec import PublishedResult

#: ontology -> {mask rate in %: (M, D w/HL, D w/RL, L)} in % of the axioms satisfied
_TABLE = {
    "family": {
        20: (0.0, 100.0, 93.1, 100.0),
        40: (0.0, 100.0, 93.1, 100.0),
        60: (0.0, 100.0, 93.1, 100.0),
        80: (0.0, 100.0, 93.1, 100.0),
    },
    "family2": {
        20: (0.0, 100.0, 73.8, 91.8),
        40: (0.0, 96.7, 73.8, 88.5),
        60: (0.0, 98.4, 73.8, 91.8),
        80: (0.0, 96.7, 73.8, 91.8),
    },
    "glycordf": {
        20: (4.1, 100.0, 90.9, 96.4),
        40: (4.1, 100.0, 90.9, 96.4),
        60: (4.1, 99.5, 90.9, 95.9),
        80: (4.1, 99.5, 90.9, 95.9),
    },
    "nifdys": {
        20: (7.3, 97.3, 89.2, 95.0),
        40: (2.7, 98.7, 84.5, 94.7),
        60: (0.4, 99.3, 82.7, 94.8),
        80: (0.1, 99.6, 82.7, 94.7),
    },
    "nihss": {
        20: (16.1, 100.0, 100.0, 51.6),
        40: (0.0, 100.0, 100.0, 48.4),
        60: (0.0, 100.0, 100.0, 48.4),
        80: (0.0, 100.0, 100.0, 48.4),
    },
    "ontodm": {
        20: (5.4, 91.6, 41.5, 98.5),
        40: (0.3, 91.2, 37.3, 98.2),
        60: (0.2, 92.1, 37.3, 97.6),
        80: (0.2, 91.3, 37.3, 97.4),
    },
    "sso": {
        20: (0.0, 100.0, 100.0, 100.0),
        40: (0.0, 100.0, 100.0, 100.0),
        60: (0.0, 100.0, 100.0, 100.0),
        80: (0.0, 100.0, 100.0, 100.0),
    },
}

RESULTS = []
for _name, _rates in _TABLE.items():
    for _rate, (_m, _hl, _rl, _l) in _rates.items():
        for _loss, _value in (("hierarchical", _hl), ("rule", _rl)):
            RESULTS.append(PublishedResult(
                benchmark=f"dfalc-revision-{_name}-mask{_rate}", method="dfalc",
                metrics={"success_rate": round(_value / 100, 4)}, source=WU2022,
                locator="tab:result1, " + ("D w/HL" if _loss == "hierarchical"
                                           else "D w/RL"),
                settings={"loss_type": _loss, "alpha": 0.8}, recipe=_loss,
                note=f"Success rate of the masked grounding M: {_m}%; Logical Tensor "
                     f"Networks: {_l}%. The paper's axiom counts differ from mOWL's (see "
                     "the benchmark notes)."))
