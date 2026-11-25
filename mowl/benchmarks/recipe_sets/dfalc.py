"""Settings of the masked ABox revision experiment of DF-ALC (Wu et al. 2022).

The paper: "Adam optimizer was used with a learning rate of 2e-4 to learn the grounding.
Early stopping with 10 epochs tolerance was used to limit the running time" (Experiments,
Settings); the rule-based loss uses alpha' = 0.8. The released code (``training/run.py``)
takes one Adam step per epoch on the sum of the losses of all axiom patterns, at most
``epoch_size = 50000`` epochs, and stops when the loss, checked every 100 epochs, has not
improved for ``earlystopping = 10`` checks.

mOWL's :class:`~mowl.models.dfalc.model.DFALCModel` also takes one Adam step per epoch on the
sum of the losses of all axiom patterns, but has no early stopping: a recipe runs the
maximum number of epochs of the code. The mOWL model weights the TBox loss (``alpha``,
default 0.4) which Adam's scale invariance makes immaterial for the learning rate, and it
has no ABox losses here because the regime removes the ABox from the training data.
"""

from mowl.benchmarks.recipes import Recipe
from mowl.benchmarks.entries._refs import WU2022
from mowl.benchmarks.entries.dfalc import LABELS, MASK_RATES

_NOTE = ("Learning rate and loss as in the paper; the code's maximum of 50000 epochs (one "
         "full-batch Adam step each) without its early stopping (10 checks of 100 epochs "
         "without improvement). The success rate of the paper is for the converged "
         "grounding.")


def _build():
    out = []
    for name in LABELS:
        for rate in MASK_RATES:
            for loss in ("hierarchical", "rule"):
                out.append(Recipe(
                    benchmark=f"dfalc-revision-{name}-mask{int(rate * 100)}", method="dfalc",
                    source=WU2022, name=loss,
                    model_kwargs={"learning_rate": 2e-4, "loss_type": loss,
                                  "threshold": 0.8, "batch_size": 65536},
                    epochs=50000,
                    locator="Experiments, Settings; training/run.py (learning_rate, "
                            "epoch_size, earlystopping)",
                    notes=_NOTE))
    return tuple(out)


RECIPES = _build()
