"""Recipes of the faithfulness benchmarks of Mashkova & Hoehndorf (2026), from the paper
and the reference implementation (bio-ontology-research-group/kbc, ``run_el.py``).

One recipe covers one method on one of the two ontologies; the *normal form* (the split
that decides training data and metrics) is a dataset keyword, selected per run with
``dataset_kwargs={"normal_form": ...}`` (the benchmark's default is ``"gci0"``).
"""

from mowl.benchmarks.recipes import Recipe
from mowl.benchmarks.entries.faithfulness import MASHKOVA2026

_LOCATOR = (
    "run_el.py of bio-ontology-research-group/kbc (the fallback configuration it applies "
    "when no cell of configs/hyperparameters.json is selected)")

_COMMON = {"embed_dim": 50, "learning_rate": 0.001, "batch_size": 32768}

_MODEL_KWARGS = {
    "elem": dict(_COMMON, margin=0.0, reg_norm=1, gci1_radius_term=False),
    "emelpp": dict(_COMMON, margin=0.0),
    "box2el": dict(_COMMON, margin=0.0, delta=2, reg_factor=0.05),
}

_NOTE = (
    "The paper reports means over seeds (Appendix G) but does not print the per-cell "
    "hyperparameters, and the reference repository records a selected configuration for "
    "one cell only (box2el / Pizza / gci2: gamma 0.01, delta 1, reg_factor 0, embed_dim "
    "25, learning rate 0.01); its own hyperparameters.json says every other cell 'MUST "
    "be filled in ... before the numbers can be reproduced'. This recipe therefore pins "
    "the fallback configuration of run_el.py (embed_dim 50, learning rate 0.001, up to "
    "6,000 epochs, batch 32,768 = full batch; margin 0 for ELEm and EmEL++, reg_norm 1 "
    "for ELEm; gamma 0, delta 2, reg_factor 0.05 for Box2EL) with the training design of "
    "run_el.py's defaults, regime 'kbc-code-training' (one fixed gci2 corruption per "
    "axiom, pooled mean of all per-axiom losses, ReduceLROnPlateau patience 10, early "
    "stopping after 20 epochs, best validation loss). The one recorded cell can be "
    "selected on the command line (--param embed_dim=25 --param learning_rate=0.01 "
    "--param margin=0.01 --param delta=1 --param reg_factor=0 with --dataset-param "
    "normal_form=gci2). The models are mOWL's modules, not the code's own copies (ELEm "
    "there uses leaky ReLU), so the numbers are reproduced only up to this configuration "
    "uncertainty.")

RECIPES = tuple(
    Recipe(
        benchmark=f"faithfulness-{onto}",
        method=method,
        source=MASHKOVA2026,
        model_kwargs=dict(kwargs),
        training_regime="kbc-code-training",
        regime_kwargs={"patience": 10, "stop_patience": 20},
        epochs=6000,
        locator=_LOCATOR,
        notes=_NOTE,
    )
    for onto in ("pizza", "goplus")
    for method, kwargs in _MODEL_KWARGS.items()
)
