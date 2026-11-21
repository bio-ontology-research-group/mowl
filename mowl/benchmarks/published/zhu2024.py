"""Published results of BoxSEL (Zhu et al. 2024, arXiv:2407.11821).

Transcribed from the LaTeX source of the arXiv version (the e-print dated June 2026, UAI
template), table "Approximation quality of baselines and BoxSEL" (label ``tab:main``): soundness
accuracy (SA, printed in %, stored as a fraction), soundness error (SE) and approximation gap
(AG) of the interval estimates over N = 60 embeddings, on the Person, Country and Hybrid
Statistical EL ontologies derived from YAGO3. Only the BoxSEL rows are recorded; the fixed,
random and KDE estimators are not mOWL methods (the fixed and random estimators are
reported by the regime in ``info['baselines']``). The embedding error (MAE) is not
tabulated in the version read.

The ontologies are not released, so these numbers cannot be reproduced with mOWL; the
hyperparameters are those of the table "Best hyperparameters" (the recipes).
"""

from mowl.benchmarks.entries._refs import ZHU2024
from mowl.benchmarks.spec import PublishedResult

_ROWS = {"person": (89.30, 0.013, 0.233), "country": (95.60, 0.004, 0.241),
         "hybrid": (89.70, 0.020, 0.293)}

RESULTS = [
    PublishedResult(
        benchmark=f"boxsel-sel-{name}", method="boxsel",
        metrics={"soundness_accuracy": round(sa / 100, 4), "soundness_error": se,
                 "approximation_gap": ag},
        source=ZHU2024, locator="tab:main (Approximation quality), BoxSEL (ours)",
        settings={"n_embeddings": 60},
        note="Interval estimate from 60 embeddings (min and max of their probabilities), "
             "averaged over five query sets; the ontology of the paper is not released.")
    for name, (sa, se, ag) in _ROWS.items()
]
