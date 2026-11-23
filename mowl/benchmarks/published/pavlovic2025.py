"""Published results of RESHUFFLE on the inductive benchmarks (Pavlovic, Sallinger and
Schockaert, KR 2025; arXiv:2406.09529), transcribed from the LaTeX source of the arXiv
version.

Every number is Hits@10 with 50 sampled negatives per side, the mean of three seeds:

* RESHUFFLE: Table 2 (``tab:InductiveRes``, row "RESHUFFLE"; it equals the "mean" row of the
  appendix Table ``tab:BenchRes`` and the RESHUFFLE row of Table 3);
* the ablations RESHUFFLE_nL (no loop, ``aggregation="max_self_noLoop"``) and RESHUFFLE^2
  (squared relation entries, ``row_function="square"``): Table 3 (``tab:ablationRes``);
* the standard deviation over the three seeds (appendix Table ``tab:BenchRes``) is in the
  note of each RESHUFFLE row.

The other rows of Table 2 (GraIL, CoMPILE, NBFNet, MorsE, RuleN, AnyBURL, DRUM, Neural-LP)
are methods mOWL does not implement and are not recorded.
"""

from mowl.benchmarks.spec import PublishedResult
from mowl.benchmarks.entries._refs import PAVLOVIC2025

_ORDER = [(d, v) for d in ("fb15k237", "wn18rr", "nell995") for v in ("v1", "v2", "v3", "v4")]


def _row(text):
    values = [float(x) for x in text.split()]
    assert len(values) == 12
    return dict(zip(_ORDER, values))


# Table 2, row RESHUFFLE; Table 3, rows RESHUFFLE_nL and RESHUFFLE^2 (columns: FB15k-237
# v1-v4, WN18RR v1-v4, NELL-995 v1-v4)
_MAIN = _row("0.747 0.885 0.903 0.918 0.710 0.729 0.602 0.694 0.638 0.861 0.882 0.812")
_NO_LOOP = _row("0.744 0.890 0.903 0.917 0.698 0.685 0.618 0.682 0.627 0.738 0.886 0.815")
_SQUARE = _row("0.304 0.569 0.385 0.916 0.293 0.309 0.155 0.270 0.488 0.558 0.334 0.370")
# appendix Table "RESHUFFLE's benchmark Hits@10 scores on all seeds", row stdv
_STDV = _row("0.004 0.007 0.005 0.001 0.003 0.006 0.024 0.004 0.010 0.013 0.011 0.004")
_SEEDS = {1: _row("0.751 0.879 0.905 0.918 0.713 0.727 0.614 0.693 0.630 0.874 0.871 0.816"),
          2: _row("0.744 0.892 0.908 0.916 0.707 0.726 0.574 0.690 0.650 0.860 0.893 0.808"),
          3: _row("0.746 0.883 0.897 0.918 0.710 0.736 0.617 0.698 0.635 0.848 0.881 0.812")}

_VARIANTS = (
    ("RESHUFFLE", "Table 2 (RESHUFFLE row)", _MAIN, {"variant": "reshuffle"}),
    ("RESHUFFLE_nL (no loop)", "Table 3 (RESHUFFLE_nL row)", _NO_LOOP,
     {"variant": "no_loop", "aggregation": "max_self_noLoop"}),
    ("RESHUFFLE^2 (squared relation entries)", "Table 3 (RESHUFFLE^2 row)", _SQUARE,
     {"variant": "square", "row_function": "square"}),
)

RESULTS = []
for _label, _locator, _table, _settings in _VARIANTS:
    for (_data, _version), _value in _table.items():
        _note = (f"{_label}, Hits@10 with 50 sampled negatives per side, mean of three seeds. "
                 "Hyper-parameters selected on the validation split of the training graph "
                 "(appendix Table hyperparas); trained for up to 1000 epochs with early "
                 "stopping on validation Hits@10.")
        if _settings["variant"] == "reshuffle":
            _note += (f" Standard deviation over the seeds: {_STDV[(_data, _version)]:.3f}; "
                      "seeds 1, 2, 3: " + ", ".join(
                          f"{_SEEDS[s][(_data, _version)]:.3f}" for s in (1, 2, 3)) + ".")
        RESULTS.append(PublishedResult(
            benchmark=f"kg-inductive-{_data}-{_version}", method="reshuffle",
            metrics={"hits@10": _value}, source=PAVLOVIC2025, locator=_locator,
            settings=dict(_settings), note=_note,
            recipe=_settings["variant"].replace("_", "-")))
