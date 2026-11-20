"""Results published in Singh, Jiménez-Ruiz & Weyde (2026), ELK-Em (KG-NeSy 2026).

Only the ELK-Em rows of Table 3 are recorded here. The baseline rows of Table 3 are copied
by the paper from Table 6 of Jackermeier et al. (2024) and belong to that source. Table 4
(CAFA5 protein function prediction) has no matching mOWL benchmark.

The paper evaluates on the same inferred-subsumption data as Box2EL but with its own
ranking protocol (owl:Thing removed from the candidates, average-rank ties, margin-free
score), so the numbers are filed under the 'elkem-*-subsumption' benchmarks.
"""

from mowl.benchmarks.spec import PublishedResult
from mowl.catalog.entries.elkem import SINGH2026

_SETTINGS = {
    "embed_dim": 50,
    "optimizer": "Adam",
    "learning_rate": 0.01,
    "lr_decay": "x0.1 after 5,000 epochs",
    "margin": 0.1,
    "batch_size": 1024,
    "epochs": 10000,
    "model_selection": "validation median rank",
    "seeds": 5,
}

_NOTE = (
    "Means over five seeds; the paper prints standard deviations {std}. Unfiltered "
    "ranking over the whole concept signature by s(α) = exp(−ℓ⁰(α)), with owl:Thing and "
    "the query removed from the candidates; the authors' evaluation code gives tied "
    "candidates their average rank (its specification §8 D7 says so; D8 asks to keep "
    "owl:Thing, which the code deliberately does not, because owl:Thing saturates the "
    "score and would dominate every tie block). The paper itself does not describe tie "
    "handling. contrastive_factor 3.0 is the default of the authors' training script, "
    "not stated in the paper.")

# (benchmark, k_neg, metrics, standard deviations as printed)
_ROWS = (
    ("elkem-galen-subsumption", 3,
     {"hits@1": 0.00, "hits@10": 0.08, "hits@100": 0.35, "med": 226, "mrr": 0.03,
      "mr": 2323, "auc": 0.90},
     "H@1 0.00, H@10 0.00, H@100 0.00, Med 3, MRR 0.00, MR 30, AUC 0.00"),
    ("elkem-go-subsumption", 5,
     {"hits@1": 0.00, "hits@10": 0.10, "hits@100": 0.71, "med": 43, "mrr": 0.04,
      "mr": 529, "auc": 0.99},
     "H@1 0.00, H@10 0.01, H@100 0.01, Med 1, MRR 0.00, MR 51, AUC 0.00"),
    ("elkem-anatomy-subsumption", 3,
     {"hits@1": 0.03, "hits@10": 0.20, "hits@100": 0.50, "med": 104, "mrr": 0.08,
      "mr": 1368, "auc": 0.99},
     "H@1 0.00, H@10 0.01, H@100 0.01, Med 3, MRR 0.00, MR 95, AUC 0.00"),
)

RESULTS = [
    PublishedResult(
        benchmark=benchmark,
        method="elkem",
        metrics=metrics,
        source=SINGH2026,
        locator="Table 3",
        subset="all",
        settings=dict(_SETTINGS, k_neg=k_neg),
        note=_NOTE.format(std=std),
    )
    for benchmark, k_neg, metrics, std in _ROWS
]
