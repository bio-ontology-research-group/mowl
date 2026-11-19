"""Results published in Mondal, Bhatia & Mutharaju (2021), EmEL++ (AAAI-MAKE 2021).

Table 3, "Test Data" columns, for ElEm and EmEL++ on GALEN, GO and ANATOMY. Not recorded:
the knowledge graph baselines (TransE, TransH, DistMult; not EL methods run by this
benchmark), the "Inferences Data" columns (no matching mOWL benchmark) and SNOMED CT (not
distributed).
"""

from mowl.benchmarks.spec import PublishedResult
from mowl.benchmarks.entries._refs import MONDAL2021

_NOTE = (
    "Embedding dimension and margin were tuned on the validation set over n in {50, 100, "
    "200} and margin in {-0.1, 0, 0.1} (Section 4.3); the chosen values are in the paper's "
    "Table 2. The reference training script trains on the whole normalised ontology, "
    "which still contains the test pairs, so these numbers are not directly comparable "
    "with a run that removes the test pairs (see the notes of the benchmark).")

# (benchmark, method): hits@1, hits@10, hits@100, auc, median rank, 90th percentile rank
_ROWS = {
    ("emelpp-galen-prediction", "elem"): (0.01, 0.08, 0.15, 0.64, 6658, 19681),
    ("emelpp-galen-prediction", "emelpp"): (0.02, 0.11, 0.16, 0.65, 6623, 20635),
    ("emelpp-go-prediction", "elem"): (0.01, 0.08, 0.23, 0.73, 4302, 41225),
    ("emelpp-go-prediction", "emelpp"): (0.01, 0.09, 0.24, 0.78, 2908, 33547),
    ("emelpp-anatomy-prediction", "elem"): (0.04, 0.19, 0.41, 0.73, 640, 105111),
    ("emelpp-anatomy-prediction", "emelpp"): (0.04, 0.17, 0.39, 0.76, 381, 105274),
}

RESULTS = [
    PublishedResult(
        benchmark=benchmark, method=method,
        metrics=dict(zip(("hits@1", "hits@10", "hits@100", "auc", "med", "p90"), values)),
        source=MONDAL2021, locator="Table 3 (Test Data)", subset="all",
        settings={"model_selection": "validation set"}, note=_NOTE)
    for (benchmark, method), values in _ROWS.items()
]
