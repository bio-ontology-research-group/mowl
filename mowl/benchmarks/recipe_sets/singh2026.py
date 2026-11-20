"""Recipes of ELK-Em (Singh, Jiménez-Ruiz & Weyde, KG-NeSy 2026), from the paper and the
authors' code (city-artificial-intelligence/elk-em, paper/ranking/train.py)."""

from mowl.benchmarks.recipes import Recipe
from mowl.catalog.entries.elkem import SINGH2026

_LOCATOR = (
    "Section 5.1 of the paper; the defaults of paper/ranking/train.py and train.sh of "
    "city-artificial-intelligence/elk-em")

_NOTE = (
    "The paper averages five seeds. The contrastive factor 3.0 and the box-size "
    "regulariser weight 1.0 are defaults of the authors' training script and are not "
    "stated in the paper. mOWL's regime 'elkem-code-training' deviates from the code in "
    "four documented points (margin-ful scores inside the Pearson term, unsquared "
    "per-axiom losses without the code's x1000 scaling, rare C ⊑ C negatives, corruption "
    "pool without ⊤/⊥/unsatisfiable classes); see the regime card. Role inclusions and "
    "role chains are trained (load_role_axioms=True), as the code's bins contain nf5/nf6. "
    "Model selection by validation median rank every 100 epochs, as in the code.")

# k_neg = corruptions per positive: 5 for GO, 3 for GALEN and Anatomy (train.sh)
_K_NEG = {"galen": 3, "go": 5, "anatomy": 3}

RECIPES = tuple(
    Recipe(
        benchmark=f"elkem-{onto}-subsumption",
        method="elkem",
        source=SINGH2026,
        model_kwargs={"embed_dim": 50, "learning_rate": 0.01, "batch_size": 1024,
                      "margin": 0.1, "k_neg": k_neg, "contrastive_factor": 3.0,
                      "reg_factor": 1.0, "load_role_axioms": True},
        training_regime="elkem-code-training",
        regime_kwargs={"batch_size": 1024, "k_neg": k_neg, "contrastive_factor": 3.0,
                       "lr_milestones": [5000]},
        epochs=10000,
        validate_every=100,
        locator=_LOCATOR,
        notes=_NOTE,
    )
    for onto, k_neg in _K_NEG.items()
)
