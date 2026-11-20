"""Recipes: the settings under which a publication obtained its numbers.

A :class:`Recipe` names, for one method on one benchmark, the model arguments, the
training regime (with its arguments), the number of epochs and the validation interval
of a publication, so that ``run_benchmark(benchmark, method, recipe=True)`` trains the
way the published numbers were produced. The evaluation regime is part of the
benchmark.

Recipes record what a paper *and its code* did. Where the two differ, the code is
followed and the difference is stated in ``notes``.
"""

from dataclasses import dataclass, field

from mowl.catalog.card import Reference
from mowl.benchmarks.entries._refs import JACKERMEIER2024, YANG2025


@dataclass(frozen=True)
class Recipe:
    """How a publication trained a method for a benchmark.

    :param benchmark: Key of the :class:`~mowl.benchmarks.Benchmark`.
    :param method: Key of the method in :mod:`mowl.catalog`.
    :param source: The publication.
    :param model_kwargs: Arguments of the model class.
    :param training_regime: Key of a training regime (or a list of keys).
    :param regime_kwargs: Constructor arguments of the training regime.
    :param epochs: Number of epochs.
    :param validate_every: Epochs between validations, if the regime selects a model.
    :param locator: Where the settings are stated (table, file of the code).
    :param notes: Deviations and caveats.
    """

    benchmark: str
    method: str
    source: Reference
    model_kwargs: dict = field(default_factory=dict)
    training_regime: object = None
    regime_kwargs: dict = field(default_factory=dict)
    epochs: int = 100
    validate_every: object = None
    locator: str = ""
    notes: str = ""

    def regime_spec(self):
        """The regime in the form :func:`mowl.benchmarks.runner.resolve_training_regime`
        accepts."""
        if self.training_regime is None:
            return None
        if isinstance(self.training_regime, (list, tuple)):
            return list(self.training_regime)
        return (self.training_regime, dict(self.regime_kwargs))


_B2E_LOC = "configs.json of KRR-Oxford/BoxSquaredEL (the values of the paper's Table 10)"
_B2E_NOTE = ("The margin is a slack margin (see the regime 'slack-inclusion-margin'). "
             "The paper states a maximum of 10,000 epochs (Section 4.2).")

# ontology -> task -> (margin, lr, delta, num_neg, reg_factor, lr milestone, epochs)
_BOX2EL = {
    "galen": {"prediction": (0.15, 0.01, 5, 1, 0.4, None, 10000),
              "inference": (0.05, 0.005, 1, 2, 0, 2000, 10000)},
    "go": {"prediction": (0.15, 0.01, 5.5, 5, 0.5, None, 10000),
           "inference": (0.05, 0.005, 3, 3, 0.05, None, 10000)},
    "anatomy": {"prediction": (0.05, 0.01, 5.5, 3, 0.3, None, 7000),
                "inference": (0.05, 0.001, 2, 2, 0.05, None, 7000)},
}

# method -> ontology -> task -> (embed_dim, margin or None, lr)
_BASELINES = {
    "elem": {"galen": {"prediction": (200, 0.05, 0.0005), "inference": (200, 0, 0.0005)},
             "go": {"prediction": (200, 0.1, 0.0005), "inference": (200, 0.1, 0.001)},
             "anatomy": {"prediction": (200, 0.05, 0.0005),
                         "inference": (200, 0.05, 0.0005)}},
    "elbe": {"galen": {"prediction": (200, 0.05, 0.005), "inference": (200, 0.15, 0.01)},
             "go": {"prediction": (200, 0.1, 0.005), "inference": (200, 0.05, 0.005)},
             "anatomy": {"prediction": (200, 0.05, 0.005),
                         "inference": (200, 0.05, 0.005)}},
    "boxel": {"galen": {"prediction": (50, None, 0.001), "inference": (25, None, 0.005)},
              "go": {"prediction": (25, None, 0.01), "inference": (25, None, 0.001)},
              "anatomy": {"prediction": (25, None, 0.001),
                          "inference": (25, None, 0.001)}},
}
_BASELINES["emelpp"] = _BASELINES["elem"]

# ontology -> (margin, reg_factor) of TransBox (configs_our.json, section
# "transrolebox": the model with box-shaped roles of model/TransBox_reg.py)
_TRANSBOX = {"galen": (0.1, 0.5), "go": (0.05, 0.3), "anatomy": (0, 0)}


def _build():
    out = []
    for onto, tasks in _BOX2EL.items():
        for task, (margin, lr, delta, num_neg, reg, milestone, epochs) in tasks.items():
            kwargs = {"num_negatives": num_neg, "margin": margin}
            if milestone:
                kwargs["lr_milestones"] = [milestone]
            out.append(Recipe(
                benchmark=f"box2el-{onto}-{task}", method="box2el", source=JACKERMEIER2024,
                model_kwargs={"embed_dim": 200, "learning_rate": lr, "delta": delta,
                              "reg_factor": reg},
                training_regime="box2el-code-training", regime_kwargs=kwargs,
                epochs=epochs, validate_every=100, locator=_B2E_LOC, notes=_B2E_NOTE))
    for method, ontos in _BASELINES.items():
        for onto, tasks in ontos.items():
            for task, (dim, margin, lr) in tasks.items():
                kwargs = {"num_negatives": 1}
                if margin is not None:
                    kwargs["margin"] = margin
                if method == "elbe":
                    kwargs["neg_target"] = 2.0
                out.append(Recipe(
                    benchmark=f"box2el-{onto}-{task}", method=method,
                    source=JACKERMEIER2024,
                    model_kwargs={"embed_dim": dim, "learning_rate": lr},
                    training_regime="box2el-code-training", regime_kwargs=kwargs,
                    epochs=10000, validate_every=100,
                    locator="Table 10 of the paper; model/*.py of KRR-Oxford/BoxSquaredEL",
                    notes=_B2E_NOTE))
    for onto, (margin, reg) in _TRANSBOX.items():
        out.append(Recipe(
            benchmark=f"transbox-{onto}-prediction", method="transbox", source=YANG2025,
            model_kwargs={"embed_dim": 200, "learning_rate": 0.005, "reg_factor": reg},
            training_regime="transbox-code-training",
            regime_kwargs={"num_negatives": 1, "margin": margin},
            epochs=5000, validate_every=100,
            locator="configs_our.json of HuiYang1997/TransBox, section 'transrolebox'",
            notes=("The paper averages 10 runs of 5,000 epochs (Section 6.2). The code "
                   "has a second configuration section ('transbox': margin 0, no "
                   "regularisation, learning rate 0.0005), which with mOWL's TransBox "
                   "does not train on GALEN and GO. With this recipe mOWL's TransBox "
                   "comes close to the published combined numbers on GALEN and GO but "
                   "not per normal form, and stays below on Anatomy; it is not a "
                   "verified reproduction.")))
    return tuple(out)


RECIPES = _build()


def list_recipes(benchmark=None, method=None, source=None):
    """Recipes, optionally filtered by benchmark key, method key and source key."""
    return [r for r in RECIPES
            if (benchmark is None or r.benchmark == benchmark)
            and (method is None or r.method == method)
            and (source is None or r.source.key == source)]


def get_recipe(benchmark, method, source=None):
    """The recipe of ``method`` on ``benchmark``.

    :param source: Key of the publication, needed only if several publications have a
        recipe for the pair.
    :raises KeyError: If there is none, or several and no ``source`` is given.
    """
    found = list_recipes(benchmark, method, source)
    if not found:
        raise KeyError(f"No recipe for method {method!r} on benchmark {benchmark!r}"
                       + (f" from {source!r}." if source else "."))
    if len(found) > 1:
        raise KeyError(f"Several recipes for {method!r} on {benchmark!r}: give source, "
                       f"one of {sorted(r.source.key for r in found)}.")
    return found[0]
