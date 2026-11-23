"""The direct implementations as an implementation of the benchmark runner.

Importing this module registers the implementation ``"baseline"`` with
:mod:`mowl.benchmarks.implementations`: for a catalogue method, the runner then builds
the direct implementation kept here (``mowl.baselines.models...``) instead of the
combination of parts, with the same benchmark, recipe, model arguments, training
regime, evaluation regime and seed. The library does not import this module; a run
names it::

    python -m mowl.benchmarks run box2el-galen-prediction --method elbe --recipe \\
        --plugin mowl.baselines.benchmarks --implementation baseline --out elbe.json

    # the same, shorter
    python -m mowl.baselines.benchmarks run box2el-galen-prediction --method elbe \\
        --recipe --out elbe.json

How a method is mapped (:data:`MODELS`, :data:`FACTORIES`):

* a method whose card names a class ``mowl.models.<pkg>.<module>.<Class>`` (or
  ``mowl.geometric.<module>.<Class>``) is the class of the same name under
  ``mowl.baselines.models`` (``mowl.baselines.geometric``), built with the same
  arguments and trained by the same :class:`~mowl.benchmarks.runner.ModelFamily` as the
  library class;
* a method whose card names a factory (DL2Vec, OWL2Vec*, taxonomy walks, Onto2Vec,
  OPA2Vec, the graph KGE) is configured by the same factory of
  :mod:`mowl.benchmarks.families` with the direct class (``model_class``);
* methods without a direct implementation (the general combinations ``composed*``, the
  MILP box construction, the least model) are not supported (:data:`UNSUPPORTED`).

Training regimes. A recipe's training regime (``box2el-code-training``, ...) is part of
the experiment, not of the method: both implementations are trained by the *same*
regime object of :mod:`mowl.training`, so that a difference between them is a difference
of the method. The direct classes accept only their own frozen regime class in
``training_regime``; :meth:`BaselineImplementation.attach_regime` checks that their loop
consults every hook the regime changes (as their setter does) and attaches it.

Options that differ (:data:`ARGUMENT_MAP`). Where a direct class spells an argument
differently, or lacks an option that the combination has, the mapping is explicit:
an argument is renamed, or the run is refused with the reason; nothing is dropped
silently.
"""

import importlib
import inspect
import sys

from mowl.benchmarks.implementations import (Implementation, register_implementation,
                                             with_run_seed)

__all__ = ["BaselineImplementation", "MODELS", "FACTORIES", "UNSUPPORTED", "ARGUMENT_MAP",
           "REFUSED_ARGUMENTS", "baseline_path", "baseline_class", "main"]

#: Methods whose direct class is not found by the naming rule: method -> dotted path.
MODELS = {
    "canonical_model": "mowl.baselines.geometric.core.completion.CanonicalModel",
    "alcone-exact": "mowl.baselines.geometric.cones.ALConeInterpretation",
    "temporal_cones": "mowl.baselines.geometric.temporal_cones.TemporalALConeInterpretation",
}

_WALKS = "mowl.baselines.models.graph_random_walk.random_walk_w2v_model.RandomWalkPlusW2VModel"
_CORPUS = "mowl.baselines.models.syntactic.w2v_model.SyntacticPlusW2VModel"
_PYKEEN = "mowl.baselines.models.graph_kge.graph_pykeen_model.GraphPlusPyKEENModel"

#: Methods that are a configuration of a general class: method -> (factory of
#: mowl.benchmarks.families, direct class it configures).
FACTORIES = {
    "dl2vec": ("mowl.benchmarks.families.dl2vec", _WALKS),
    "owl2vec_star": ("mowl.benchmarks.families.owl2vec_star", _WALKS),
    "taxonomy_walks": ("mowl.benchmarks.families.taxonomy_walks", _WALKS),
    "onto2vec": ("mowl.benchmarks.families.onto2vec", _CORPUS),
    "opa2vec": ("mowl.benchmarks.families.opa2vec", _CORPUS),
    "graph_kge": ("mowl.benchmarks.families.graph_kge", _PYKEEN),
}

#: Methods without a direct implementation: method -> reason.
UNSUPPORTED = {
    "composed": "the general combination of EL parts; it has no direct implementation "
                "(its named combinations have one each)",
    "composed-alc": "the general combination of ALC parts; no direct implementation",
    "composed_subspace": "the general combination of subspace parts; no direct "
                         "implementation",
    "composed_graph_kge": "the general combination of graph parts; no direct "
                          "implementation",
    "composed_sequences": "the general combination of walk and sentence parts; no direct "
                          "implementation",
    "box_milp": "written as a combination of parts from the start; no direct "
                "implementation",
    "least_model": "written as a combination of parts from the start; no direct "
                   "implementation",
}

#: Arguments of the combination that the direct class spells differently or lacks:
#: method -> {argument: new name, or None to refuse the run with a reason}.
#: ``None`` entries are given with the reason in :data:`REFUSED_ARGUMENTS`.
ARGUMENT_MAP = {}

#: method -> {argument: why the direct class cannot be run with it}.
REFUSED_ARGUMENTS = {}


def _load(path):
    module_name, _, attr = path.rpartition(".")
    return getattr(importlib.import_module(module_name), attr)


def baseline_path(card):
    """Dotted path of the direct class of a catalogue card (``None`` if the naming rule
    gives none)."""
    if card.key in MODELS:
        return MODELS[card.key]
    for library, frozen in (("mowl.models.", "mowl.baselines.models."),
                            ("mowl.geometric.", "mowl.baselines.geometric.")):
        if card.implementation.startswith(library):
            return frozen + card.implementation[len(library):]
    return None


def _defines(path):
    """``True`` if the module of the dotted ``path`` defines the class, read from its
    source (without importing it: the models need the JVM)."""
    import ast
    import os
    module_name, _, attr = path.rpartition(".")
    # mowl.baselines.<...> -> a file under this package's parent (the mowl folder),
    # found without importing the parent packages (some start Java imports).
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parts = module_name.split(".")[1:]
    origin = os.path.join(root, *parts) + ".py"
    if not os.path.exists(origin):
        origin = os.path.join(root, *parts, "__init__.py")
    if not os.path.exists(origin):
        return False
    with open(origin, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    return any(isinstance(n, ast.ClassDef) and n.name == attr for n in tree.body)


def baseline_class(card):
    """The direct class of a catalogue card (``None`` if there is none)."""
    path = baseline_path(card)
    if path is None or not _defines(path):
        return None
    return _load(path)


def _explicit_parameters(callable_):
    """Names of the explicit parameters of ``callable_`` and of the classes it passes
    ``**kwargs`` to (the method resolution order of a class), or ``None`` if a
    ``**kwargs`` is not resolved."""
    if isinstance(callable_, type):
        names = set()
        # The query models pass their extra keyword arguments to their geometry.
        geometry = getattr(callable_, "geometry_class", None)
        if isinstance(geometry, type):
            names |= _explicit_parameters(geometry) or set()
        for cls in callable_.__mro__:
            init = cls.__dict__.get("__init__")
            if init is None:
                continue
            params = inspect.signature(init).parameters
            names |= {n for n, p in params.items()
                      if p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL)}
            if not any(p.kind is p.VAR_KEYWORD for p in params.values()):
                return names
        return names
    params = inspect.signature(callable_).parameters
    if any(p.kind is p.VAR_KEYWORD for p in params.values()):
        return None
    return set(params)


class BaselineImplementation(Implementation):
    """The direct implementations of :mod:`mowl.baselines` (see the module docstring)."""

    name = "baseline"
    description = ("the direct implementation of each method (mowl.baselines), kept for "
                   "comparison")

    def unsupported_reason(self, method):
        if method in UNSUPPORTED:
            return f"{method}: {UNSUPPORTED[method]}"
        if method in FACTORIES:
            return ""
        from mowl.catalog import get_method
        try:
            card = get_method(method)
        except KeyError:
            return f"{method} is not a catalogue method"
        if not card.implemented:
            return f"{method} is not implemented in mOWL"
        path = baseline_path(card)
        if path is None or not _defines(path):
            return f"{method}: no direct implementation of {card.implementation}"
        return ""

    def supports(self, method):
        return not self.unsupported_reason(method)

    def _arguments(self, method, target, model_kwargs):
        kwargs = dict(model_kwargs)
        refused = {k: why for k, why in REFUSED_ARGUMENTS.get(method, {}).items()
                   if k in kwargs}
        if refused:
            from mowl.benchmarks.runner import UnsupportedModelError
            raise UnsupportedModelError(
                f"The direct implementation of {method} cannot be run with "
                + "; ".join(f"{k}={kwargs[k]!r} ({why})" for k, why in refused.items()))
        for old, new in ARGUMENT_MAP.get(method, {}).items():
            if old in kwargs:
                kwargs[new] = kwargs.pop(old)
        known = _explicit_parameters(target)
        if known is not None:
            unknown = sorted(set(kwargs) - known - {"device", "model_filepath"})
            if unknown:
                from mowl.benchmarks.runner import UnsupportedModelError
                raise UnsupportedModelError(
                    f"The direct implementation of {method} ({target.__qualname__}) has no "
                    f"argument {', '.join(unknown)}: the option exists only in the "
                    f"combination of parts. Add the mapping to "
                    f"mowl.baselines.benchmarks.ARGUMENT_MAP or REFUSED_ARGUMENTS.")
        return kwargs

    def build(self, card, dataset, model_kwargs, device, model_filepath, seed=0):
        from mowl.benchmarks.runner import UnsupportedModelError, family_for
        reason = self.unsupported_reason(card.key)
        if reason:
            raise UnsupportedModelError(reason)
        if card.key in FACTORIES:
            import os
            factory_path, class_path = FACTORIES[card.key]
            factory = _load(factory_path)
            kwargs = with_run_seed(factory, model_kwargs, seed)
            kwargs = self._arguments(card.key, factory, kwargs)
            kwargs.setdefault("device", device)
            kwargs.setdefault("model_filepath", model_filepath)
            os.makedirs(os.path.dirname(os.path.abspath(model_filepath)), exist_ok=True)
            model = factory(dataset, model_class=_load(class_path), **kwargs)
            # The same family as the library class the factory configures by default.
            return model, family_for(card.load())
        cls = baseline_class(card)
        # Trained the way the library class is trained.
        family = family_for(card.load())
        kwargs = with_run_seed(cls, model_kwargs, seed)
        kwargs = self._arguments(card.key, cls, kwargs)
        return family.build(cls, dataset, kwargs, device, model_filepath), family

    def attach_regime(self, model, regime):
        from mowl.benchmarks.runner import UnsupportedModelError
        if not hasattr(type(model), "training_regime"):
            raise UnsupportedModelError(
                f"{type(model).__name__} (direct implementation) has no 'training_regime' "
                f"attribute; the regime cannot be attached.")
        from mowl.baselines.training.regime import TrainingRegime as FrozenRegime
        if isinstance(regime, FrozenRegime):
            model.training_regime = regime
            return
        # The regime of the library, shared by both implementations: the checks of the
        # direct class's setter, without its check of the regime's class.
        missing = frozenset(regime.overridden_hooks()) - \
            frozenset(getattr(model, "training_hooks", ()))
        if missing:
            raise UnsupportedModelError(
                f"{type(regime).__name__} changes {sorted(missing)}, which the training "
                f"loop of the direct {type(model).__name__} does not consult.")
        regime.attach(model)
        model._training_regime = regime

    def describe(self, model):
        out = super().describe(model)
        regime = getattr(model, "training_regime", None)
        if regime is not None:
            out["training_regime_class"] = f"{type(regime).__module__}." \
                                           f"{type(regime).__qualname__}"
        return out


IMPLEMENTATION = register_implementation(BaselineImplementation())


def main(argv=None):
    """``python -m mowl.baselines.benchmarks run ...``: ``python -m mowl.benchmarks run
    ...`` with ``--implementation baseline`` (other commands are passed on unchanged)."""
    from mowl.benchmarks.__main__ import main as benchmarks_main
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "run" and not any(a == "--implementation" or
                                             a.startswith("--implementation=")
                                             for a in argv):
        argv += ["--implementation", "baseline"]
    return benchmarks_main(argv)


if __name__ == "__main__":
    sys.exit(main())
