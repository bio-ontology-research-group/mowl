"""Results published in Zhapa-Camacho and Hoehndorf, "Lattice-based ALC ontology
embeddings with saturation" (arXiv:2305.07163v3), Tables 2-4.

All metrics are filtered (Section 4.1: "In all tasks we report filtered metrics only"),
hence the ``f_`` prefix. Transcribed with ``pdftotext -layout``. Table 2 columns: MR,
MRR, Hits@3, Hits@10, Hits@100, AUC; Table 3 (per ontology): MR, H@10, H@100, AUC;
Table 4 (left, PPI axioms): MR, MRR, H@3, H@10, H@100, AUC. The right half of Table 4
(GO subsumptions in the ELK closure) is not recorded: it is not a benchmark of the set.
Baselines: mOWL implementations, grid search over embedding dimension, margin, batch
size and learning rate (Section 4.1, Appendix A); FALCON without hyperparameter search.
"""

from mowl.benchmarks.spec import PublishedResult
from mowl.benchmarks.entries._refs import ZHAPA2023

_KEYS = {"ELEmbeddings": "elem", "Box2EL": "box2el", "FALCON": "falcon", "CatE": "cate"}
_FULL = ("f_mr", "f_mrr", "f_hits@3", "f_hits@10", "f_hits@100", "f_auc")
_TBOX = ("f_mr", "f_hits@10", "f_hits@100", "f_auc")

_TABLES = [
    ("cate-ore1-membership", "Table 2", _FULL, """
ELEmbeddings 105 0.12 0.08 0.22 0.87 0.99
Box2EL 122 0.10 0.08 0.18 0.70 0.98
FALCON 603 0.02 0.00 0.02 0.34 0.92
CatE 37 0.18 0.10 0.51 0.96 0.99"""),
    ("cate-go-completion", "Table 3 (GO)", _TBOX, """
ELEmbeddings 3562 0.19 0.37 0.92
Box2EL 6621 0.01 0.07 0.85
FALCON 8982 0.02 0.08 0.79
CatE 2968 0.22 0.58 0.93"""),
    ("cate-foodon-completion", "Table 3 (FoodOn)", _TBOX, """
ELEmbeddings 3336 0.25 0.38 0.88
Box2EL 2763 0.06 0.19 0.90
FALCON 3815 0.02 0.12 0.86
CatE 2764 0.30 0.47 0.90"""),
    ("cate-yeast-ppi", "Table 4 (PPI axioms)", _FULL, """
ELEmbeddings 289 0.10 0.09 0.25 0.73 0.95
Box2EL 188 0.17 0.19 0.43 0.81 0.97
CatE 223 0.08 0.07 0.18 0.69 0.96"""),
]

RESULTS = []
for _bench, _loc, _names, _text in _TABLES:
    for _line in _text.strip().splitlines():
        _parts = _line.split()
        _note = ""
        if _parts[0] == "FALCON" and _bench in ("cate-go-completion", "cate-foodon-completion"):
            _note = "Reported as 'FALCON (5 models)' in Table 3."
        _settings = {} if _parts[0] != "CatE" else {
            "training_regime": "cate-code-training"}
        RESULTS.append(PublishedResult(
            benchmark=_bench, method=_KEYS[_parts[0]],
            metrics={n: float(v) for n, v in zip(_names, _parts[1:])},
            source=ZHAPA2023, locator=_loc, note=_note, settings=_settings))
