"""Results published in Yang, Chen, "RegD: Hierarchical Embeddings via Dissimilarity between
Arbitrary Euclidean Regions" (arXiv:2501.17518 v3, KR 2026): Tables 3 (inference) and 4
(prediction), F1 in %.

Transcribed from the rendered PDF page (checked against pdftotext). The values are stored as
fractions, like the F1 that mOWL reports (published 20.7 is 0.207). The rows of the
tau-Box plugin ("+ tauBox") are not recorded, since mOWL has no method for it.
The paper's Section 5.2 text calls ELBE a ball and ELEM a box model; the tables, the
baselines' definitions and the authors' code use ELBE for boxes and ELEM for balls.
Table 4 is referred to as "Table 3" in the text of Section 5.3.
"""

from mowl.benchmarks.spec import PublishedResult
from mowl.benchmarks.entries._refs import YANG2025_REGD

# (table, task, rows "name d5-galen d10-galen d5-go d10-go d5-anatomy d10-anatomy")
_TABLES = [
    ("Table 3", "inference", """
ELBE 20.7 21.2 36.9 42.4 43.1 43.0
ELBE+RegD 25.2 25.8 50.0 50.5 58.7 62.5
ELEM 16.9 17.3 23.5 27.4 34.6 38.7
ELEM+RegD 19.2 18.8 36.4 40.0 52.5 55.5"""),
    ("Table 4", "prediction", """
ELBE 21.0 24.2 32.8 37.9 25.1 25.5
ELBE+RegD 20.6 21.0 37.1 44.1 41.4 45.3
ELEM 16.7 16.6 54.5 54.2 23.1 26.0
ELEM+RegD 16.8 18.0 60.3 61.4 21.7 21.9"""),
]
_METHODS = {
    "ELBE": ("elbe", "ELBE (box baseline), without RegD."),
    "ELEM": ("elem", "ELEM (ball baseline), without RegD."),
    "ELBE+RegD": ("regd", "ELBE + RegD (boxes): RegDELBE."),
    "ELEM+RegD": ("regd", "ELEM + RegD (balls): RegDELEm."),
}
_COLUMNS = [("galen", 5), ("galen", 10), ("go", 5), ("go", 10), ("anatomy", 5),
            ("anatomy", 10)]

RESULTS = []
for _locator, _task, _text in _TABLES:
    for _line in _text.strip().splitlines():
        _name, *_values = _line.split()
        _method, _note = _METHODS[_name]
        for (_onto, _dim), _value in zip(_COLUMNS, _values):
            RESULTS.append(PublishedResult(
                benchmark=f"regd-{_onto}-{_task}", method=_method,
                metrics={"f1": round(float(_value) / 100, 4)}, source=YANG2025_REGD,
                locator=_locator, settings={"embed_dim": _dim}, note=_note))
