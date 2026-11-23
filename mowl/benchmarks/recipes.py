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
    :param name: A name that distinguishes several recipes of the same publication for
        the same (benchmark, method) -- for example a model variant of the paper. Recipes
        without a name are selected by source alone.
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
    name: str = ""
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

def register_recipe(recipe):
    """Add a :class:`Recipe` at run time (for recipes defined outside
    :mod:`mowl.benchmarks.recipe_sets`, e.g. by a plugin of the command line). Adding an
    equal recipe twice has no effect."""
    global _RECIPES
    if not isinstance(recipe, Recipe):
        raise TypeError("register_recipe takes a Recipe.")
    if recipe not in _recipes():
        _RECIPES = _recipes() + (recipe,)
    return recipe


def list_recipes(benchmark=None, method=None, source=None, name=None):
    """Recipes, optionally filtered by benchmark key, method key, source key and name."""
    return [r for r in _recipes()
            if (benchmark is None or r.benchmark == benchmark)
            and (method is None or r.method == method)
            and (source is None or r.source.key == source)
            and (name is None or r.name == name)]


def get_recipe(benchmark, method, source=None, name=None):
    """The recipe of ``method`` on ``benchmark``.

    :param source: Key of the publication, needed only if several publications have a
        recipe for the pair.
    :param name: Name of the recipe (the model variant of the paper, for instance),
        needed only if one publication has several recipes for the pair.
    :raises KeyError: If there is none, or several and no ``source`` / ``name`` is given.
    """
    found = list_recipes(benchmark, method, source, name)
    if not found:
        raise KeyError(f"No recipe for method {method!r} on benchmark {benchmark!r}"
                       + (f" from {source!r}" if source else "")
                       + (f" named {name!r}" if name else "") + ".")
    if len(found) > 1:
        names = sorted(r.name or "(unnamed)" for r in found)
        raise KeyError(f"Several recipes for {method!r} on {benchmark!r}"
                       + (f" from {source!r}" if source else "")
                       + f": give name, one of {names} (as 'source:name').")
    return found[0]
