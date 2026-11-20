"""Recipes: the settings under which a publication obtained its numbers.

A :class:`Recipe` names, for one method on one benchmark, the model arguments, the
training regime (with its arguments), the number of epochs and the validation interval
of a publication, so that ``run_benchmark(benchmark, method, recipe=True)`` trains the
way the published numbers were produced. The evaluation regime is part of the
benchmark.

Recipes live in :mod:`mowl.benchmarks.recipe_sets`, one module per publication or code
base, and are discovered automatically.

Recipes record what a paper *and its code* did. Where the two differ, the code is
followed and the difference is stated in ``notes``.
"""

from dataclasses import dataclass, field

from mowl.catalog.card import Reference


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


_RECIPES = None


def _recipes():
    global _RECIPES
    if _RECIPES is None:
        import importlib
        import pkgutil
        from mowl.benchmarks import recipe_sets
        found = []
        for info in sorted(pkgutil.iter_modules(recipe_sets.__path__),
                           key=lambda i: i.name):
            if info.name.startswith("_"):
                continue
            module = importlib.import_module(f"{recipe_sets.__name__}.{info.name}")
            for recipe in module.RECIPES:
                if not isinstance(recipe, Recipe):
                    raise TypeError(f"{module.__name__} exposes a non-Recipe object.")
                found.append(recipe)
        _RECIPES = tuple(found)
    return _RECIPES



def list_recipes(benchmark=None, method=None, source=None):
    """Recipes, optionally filtered by benchmark key, method key and source key."""
    return [r for r in _recipes()
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
