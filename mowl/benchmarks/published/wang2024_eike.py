"""Published results of EIKE on YAGO39K and M-YAGO39K (Wang et al. 2024, arXiv:2402.01677).

Transcribed from the LaTeX source of the arXiv version by a script that reads the table
rows (and checked against the printed tables):

* Table 2: InstanceOf triple classification (Accuracy, Precision, Recall, F1), YAGO39K and
  M-YAGO39K;
* Table 3: SubClassOf triple classification, YAGO39K and M-YAGO39K;
* Table 4: relational triple classification and link prediction on YAGO39K (MRR raw and
  filtered, filtered Hits@1/3/10).

The numbers are printed in % (Hits, Accuracy, ...) and stored as fractions; MRR is as
printed. Only the rows of EIKE are recorded: its baselines (TransE, TransC, TransEllipsoid,
JECI, ...) are not methods of mOWL. The DB99K-242 columns are not recorded: the benchmark
cannot be run (no released splits or negatives, see :mod:`mowl.benchmarks.entries.two_view`).

The variants are named as in the paper: ``UNP`` (intensional vectors initialised at random)
or ``PRE`` (initialised with Sentence-BERT vectors), ``EYE`` (identity map to the virtual
instance vector) or ``MAT`` (learned matrix), ``unif`` or ``bern`` (uniform or Bernoulli
corruption for the negatives).
"""

from mowl.benchmarks.spec import PublishedResult
from mowl.benchmarks.entries._refs import WANG2024EIKE

_INSTANCEOF = {  # (intensional, map, negatives): (YAGO39K, M-YAGO39K), Acc P R F1 in %
    ('UNP', 'EYE', 'unif'): ((87.13, 86.83, 87.54, 87.18), (87.77, 89.81, 85.21, 87.45)),
    ('UNP', 'EYE', 'bern'): ((86.81, 87.26, 86.2, 86.2), (87.57, 89.12, 85.6, 87.32)),
    ('PRE', 'EYE', 'unif'): ((89.26, 88.81, 89.84, 89.32), (88.8, 90.08, 87.2, 88.62)),
    ('PRE', 'EYE', 'bern'): ((88.42, 88.61, 88.18, 88.39), (88.28, 91.34, 84.59, 87.83)),
    ('UNP', 'MAT', 'unif'): ((85.23, 86.43, 84.06, 85.23), (85.24, 86.33, 83.75, 85.02)),
    ('UNP', 'MAT', 'bern'): ((81.97, 83.99, 79.0, 81.42), (83.21, 89.71, 75.02, 81.71)),
    ('PRE', 'MAT', 'unif'): ((84.01, 85.42, 82.02, 83.69), (80.28, 85.56, 72.84, 78.69)),
    ('PRE', 'MAT', 'bern'): ((84.34, 85.76, 82.36, 84.02), (84.64, 88.58, 79.54, 83.82)),
}

_SUBCLASSOF = {  # (intensional, map, negatives): (YAGO39K, M-YAGO39K), Acc P R F1 in %
    ('UNP', 'EYE', 'unif'): ((76.8, 68.77, 98.2, 80.89), (77.3, 69.38, 97.73, 81.15)),
    ('UNP', 'EYE', 'bern'): ((76.85, 69.28, 96.6, 80.67), (77.13, 69.28, 97.47, 80.99)),
    ('PRE', 'EYE', 'unif'): ((90.0, 86.56, 94.7, 90.45), (87.02, 86.48, 87.78, 87.12)),
    ('PRE', 'EYE', 'bern'): ((89.05, 85.6, 93.9, 89.56), (86.81, 84.33, 90.47, 87.29)),
    ('UNP', 'MAT', 'unif'): ((77.85, 71.54, 92.5, 80.68), (77.59, 69.81, 97.22, 81.27)),
    ('UNP', 'MAT', 'bern'): ((77.8, 70.23, 96.5, 81.3), (77.55, 69.99, 96.46, 81.12)),
    ('PRE', 'MAT', 'unif'): ((77.7, 71.08, 93.4, 80.73), (77.55, 70.19, 95.79, 81.01)),
    ('PRE', 'MAT', 'bern'): ((77.2, 70.15, 94.7, 80.6), (76.87, 70.27, 93.18, 80.12)),
}

_RELATIONAL = {  # YAGO39K: MRR raw, MRR filtered, Hits@1/3/10 (%) filtered, then Acc P R F1 in %
    ('UNP', 'EYE', 'unif'): ((0.093, 0.563, 45.6, 63.8, 74.5), (92.69, 92.54, 92.87, 92.7)),
    ('UNP', 'EYE', 'bern'): ((0.115, 0.577, 47.1, 65.4, 76.2), (92.62, 92.68, 92.55, 92.62)),
    ('PRE', 'EYE', 'unif'): ((0.093, 0.528, 40.8, 61.0, 74.0), (92.19, 91.7, 92.78, 92.24)),
    ('PRE', 'EYE', 'bern'): ((0.113, 0.531, 40.9, 61.0, 74.8), (93.04, 92.46, 93.73, 93.09)),
    ('UNP', 'MAT', 'unif'): ((0.088, 0.522, 40.9, 60.3, 72.0), (93.14, 93.5, 92.74, 93.12)),
    ('UNP', 'MAT', 'bern'): ((0.113, 0.545, 42.4, 62.0, 74.7), (92.75, 93.35, 92.07, 92.7)),
    ('PRE', 'MAT', 'unif'): ((0.09, 0.542, 43.4, 62.0, 72.9), (93.19, 93.03, 93.38, 93.2)),
    ('PRE', 'MAT', 'bern'): ((0.108, 0.558, 45.1, 63.5, 73.9), (93.45, 93.86, 92.99, 93.42)),
}


_TC_NAMES = ("accuracy", "precision", "recall", "f1")
_LP_NAMES = ("mrr", "f_mrr", "f_hits@1", "f_hits@3", "f_hits@10")
_INTENSIONAL = {"UNP": "random", "PRE": "pretrained"}
_MAP = {"EYE": "eye", "MAT": "matrix"}
_NEG = {"unif": "uniform", "bern": "bernoulli"}


def _settings(key):
    intensional, instance_map, negatives = key
    return {"intensional_vectors": _INTENSIONAL[intensional],
            "instance_map": _MAP[instance_map], "negative_sampling": _NEG[negatives]}


def _name(key):
    return f"EIKE-{key[0]}-{key[1]} ({key[2]})"


def _percent(values, names):
    scale = {"mrr": 1.0, "f_mrr": 1.0}
    return {n: round(v * scale.get(n, 0.01), 6) for n, v in zip(names, values)}


RESULTS = []
for _key, (_yago, _myago) in _INSTANCEOF.items():
    for _variant, _values in (("yago39k", _yago), ("m-yago39k", _myago)):
        RESULTS.append(PublishedResult(
            benchmark=f"{_variant}-instanceof-classification", method="eike",
            metrics=_percent(_values, _TC_NAMES), source=WANG2024EIKE, locator="Table 2",
            settings=_settings(_key), note=_name(_key)))
for _key, (_yago, _myago) in _SUBCLASSOF.items():
    for _variant, _values in (("yago39k", _yago), ("m-yago39k", _myago)):
        RESULTS.append(PublishedResult(
            benchmark=f"{_variant}-subclassof-classification", method="eike",
            metrics=_percent(_values, _TC_NAMES), source=WANG2024EIKE, locator="Table 3",
            settings=_settings(_key), note=_name(_key)))
for _key, (_lp, _tc) in _RELATIONAL.items():
    RESULTS.append(PublishedResult(
        benchmark="yago39k-relational-classification", method="eike",
        metrics=_percent(_tc, _TC_NAMES), source=WANG2024EIKE, locator="Table 4",
        settings=_settings(_key), note=_name(_key)))
    RESULTS.append(PublishedResult(
        benchmark="yago39k-link-prediction", method="eike",
        metrics=_percent(_lp, _LP_NAMES), source=WANG2024EIKE, locator="Table 4",
        settings=_settings(_key),
        note=_name(_key) + ". Hits@N is the filtered setting (the table caption); the "
        "paper's MRR is given raw and filtered."))
