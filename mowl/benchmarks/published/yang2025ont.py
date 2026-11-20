"""Results published in Yang, Chen, He, Gao, Horrocks, "Language Models as Ontology
Encoders" (ISWC 2025): Table 2 (prediction), Table 3 (inference) and Table 4 (ablation
on GALEN with three language models).

Transcribed from the rendered pages 12-14 of arXiv:2507.14334v1. The paper prints H@1,
H@10, H@100 and MRR as percentages rounded to integers; they are stored as fractions
(25 -> 0.25), like the metrics the ``ont-*`` benchmarks compute; MR is stored as printed.
The tables do not say which subset is reported; "Overall performance" is read as the
ranking of all normal forms combined (subset ``"combined"`` of the prediction
benchmarks), the inference benchmarks have only NF1.

OnT(w/o r), the ablation without role embeddings and logical losses (Section 5.1,
"Baselines"), is recorded as method ``ont`` with ``settings["use_roles"] = False``.
"""

from mowl.benchmarks.spec import PublishedResult
from mowl.benchmarks.entries.ont import YANG2025ONT

_KEYS = {"ELEM": "elem", "ELBE": "elbe", "BoxEL": "boxel", "Box2EL": "box2el",
         "TransBox": "transbox", "OPA2Vec": "opa2vec", "OWL2Vec*": "owl2vec_star",
         "HiT": "hit", "OnT(w/o_r)": "ont", "OnT": "ont"}

_BASELINE_NOTE = (
    "Re-implemented and run by the OnT authors under OnT's own normalisation and splits "
    "(Section 5.1: 'Due to dataset modifications, we also re-implemented all the other "
    "geometric embedding models ... based on the framework developed by Box2EL and "
    "TransBox'; d = 200, margin in {0, 0.05, 0.1, 0.15}, learning rate in {0.0005, "
    "0.005, 0.01}, 5,000 epochs, selection on validation). Not comparable with the "
    "box2el-* benchmarks, whose data and splits differ.")

_WORD2VEC_NOTE = (
    "Run by the OnT authors under OnT's own normalisation and splits, with fine-tuned "
    "word embeddings and a Random Forest classifier (Logistic Regression for the GO "
    "inference task) ranking all concepts (Section 5.1). Not comparable with the "
    "box2el-* benchmarks or with the original OWL2Vec* setting of about 50 candidates.")

_LM_NOTE = (
    "all-MiniLM-L12-v2, 1 epoch, 1 negative per axiom, α = 3.0, β = 0.5, learning rate "
    "1e-5, λ in {0, 0.1, ..., 1} selected on validation (Section 5.1; the reference "
    "code searches {0.0, ..., 1.9}, ont/evaluation/ont_eval.py). Run by the OnT "
    "authors under OnT's own normalisation and splits; not comparable with the box2el-* "
    "benchmarks.")

_LM_SETTINGS = {"encoder": "sentence-transformers/all-MiniLM-L12-v2", "epochs": 1,
                "num_negatives": 1, "clustering_margin": 3.0, "centripetal_margin": 0.5,
                "learning_rate": 1e-5, "training_regime": "ont-code-training"}

# Method H@1 H@10 H@100 MRR MR, per ontology (GALEN, GO, ANATOMY).
_TABLE2 = """
ELEM       14 50 68 26 2715   4 35 68 14 11764   10 53 78 24 1588
ELBE        9 37 55 18 4661  10 30 43 18 10236    9 41 66 20 2672
BoxEL       0  0  2  0 13824  0  0  2  0 65846    1  2  4  2 12257
Box2EL     12 38 58 21 4593   8 43 64 19 7975    11 39 65 20 2828
TransBox   11 41 62 22 2972   8 43 67 19 7092     9 49 73 22 1299
OPA2Vec     0  1  4  1 13547  0  1  4  0 18493    0  5 17  2 9537
OWL2Vec*    0  1  5  1 13660  0  0  2  0 19523    0  3 11  2 10309
HiT        25 47 62 33 2349  36 60 73 44 15080   19 54 78 31 722
OnT(w/o_r) 26 46 64 33 1546  38 66 79 48 2209    22 52 79 31 628
OnT        25 50 69 34 792   37 67 81 46 1121    22 57 82 33 475
"""

_TABLE3 = """
ELEM       0 3  9 1 8639     0  3 17 1 18377     0 4 22 2 4990
ELBE       0 4 20 2 2999     0  3 15 1 4021      0 6 39 2 979
BoxEL      0 0  3 0 11328    0  0  0 0 18186     0 0  0 0 8169
Box2EL     0 3 15 1 5530     0  1  7 1 11801     0 1  7 1 11801
TransBox   0 2  6 1 7111     0  2  9 1 4449      0 5 27 2 749
OPA2Vec    0 0  1 0 12722    3  5  6 3 95755     2 6 15 3 5143
OWL2Vec*   0 0  1 0 12647    3  7  8 5 88614     1 5 14 3 5441
HiT        0 4 26 2 953      0  1  4 0 44253     0 6 44 3 441
OnT(w/o_r) 0 4 20 1 1047     0  5 39 2 824      0 7 40 3 499
OnT        0 5 28 2 913      0 10 40 3 832       0 6 41 3 458
"""

# Table 4 (GALEN): rows other than all-MiniLM-L12-v2, which repeat Tables 2 and 3.
_TABLE4 = """
prediction OnT(w/o_r) sentence-transformers/all-MiniLM-L6-v2 26 47 64 33 1972
prediction OnT(w/o_r) sentence-transformers/all-mpnet-base-v2 26 43 63 32 979
prediction OnT sentence-transformers/all-MiniLM-L6-v2 26 51 69 35 1131
prediction OnT sentence-transformers/all-mpnet-base-v2 27 50 72 34 630
inference OnT(w/o_r) sentence-transformers/all-MiniLM-L6-v2 0 5 23 2 1062
inference OnT(w/o_r) sentence-transformers/all-mpnet-base-v2 0 2 13 1 1148
inference OnT sentence-transformers/all-MiniLM-L6-v2 0 3 26 2 923
inference OnT sentence-transformers/all-mpnet-base-v2 0 4 34 2 630
"""

_NAMES = ("hits@1", "hits@10", "hits@100", "mrr", "mr")
_ONTOLOGIES = ("galen", "go", "anatomy")


def _metrics(values):
    vals = [float(v) for v in values]
    out = {n: v / 100.0 for n, v in zip(_NAMES[:4], vals[:4])}
    out["mr"] = vals[4]
    return out


def _row(task, method, values, locator, settings_extra=None):
    key = _KEYS[method]
    if key in ("hit", "ont"):
        note, settings = _LM_NOTE, dict(_LM_SETTINGS)
        if method == "OnT(w/o_r)":
            settings["use_roles"] = False
            note = "OnT(w/o r): without role embeddings and the losses of Eqs. 6-7. " + note
        settings.update(settings_extra or {})
    elif key in ("opa2vec", "owl2vec_star"):
        note, settings = _WORD2VEC_NOTE, {}
    else:
        note, settings = _BASELINE_NOTE, {"embed_dim": 200, "epochs": 5000}
    return note, settings


RESULTS = []

for _task, _table, _loc in (("prediction", _TABLE2, "Table 2"),
                            ("inference", _TABLE3, "Table 3")):
    for _line in _table.strip().splitlines():
        _parts = _line.split()
        _method, _vals = _parts[0], _parts[1:]
        for _i, _onto in enumerate(_ONTOLOGIES):
            _note, _settings = _row(_task, _method, _vals, _loc)
            if _task == "inference" and _method == "Box2EL" and _onto in ("go", "anatomy"):
                _note += (" The GO and ANATOMY rows of Box2EL in Table 3 are identical "
                          "(0/1/7, 1, 11,801), possibly a copying error in the paper.")
            RESULTS.append(PublishedResult(
                benchmark=f"ont-{_onto}-{_task}", method=_KEYS[_method],
                metrics=_metrics(_vals[5 * _i:5 * _i + 5]), source=YANG2025ONT,
                locator=f"{_loc} ({_onto.upper()})",
                subset="combined" if _task == "prediction" else "all",
                settings=_settings, note=_note))

for _line in _TABLE4.strip().splitlines():
    _task, _method, _encoder, *_vals = _line.split()
    _note, _settings = _row(_task, _method, _vals, "Table 4", {"encoder": _encoder})
    RESULTS.append(PublishedResult(
        benchmark=f"ont-galen-{_task}", method=_KEYS[_method], metrics=_metrics(_vals),
        source=YANG2025ONT, locator="Table 4 (GALEN, ablation)",
        subset="combined" if _task == "prediction" else "all", settings=_settings,
        note=_note))
