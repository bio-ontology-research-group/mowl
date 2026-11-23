"""Implementations of a method that the benchmark runner can train and evaluate.

A catalogue method (``"elbe"``, ``"dl2vec"``, ...) is, by default, the combination of
parts users get from :mod:`mowl.models` (the implementation called ``"parts"``). Other
code bases can register further implementations of the same methods under another
name, so that one benchmark, one recipe (the settings of the paper), one evaluation
regime and one seed can be run with either and the results compared. The direct
implementations kept for comparison (:mod:`mowl.baselines`) register the implementation
``"baseline"`` when their module :mod:`mowl.baselines.benchmarks` is imported; the
library itself never imports them.

.. testcode::

   from mowl.benchmarks.implementations import (Implementation, get_implementation,
                                                list_implementations,
                                                register_implementation)

   class Mine(Implementation):
       name = "mine"
       description = "my own implementation of ELBE"

       def supports(self, method):
           return method == "elbe"

       def build(self, card, dataset, model_kwargs, device, model_filepath, seed=0):
           # build and return (model, family) here
           raise NotImplementedError

   register_implementation(Mine())
   print(get_implementation("mine").description)
   print("parts" in list_implementations())

.. testoutput::

   my own implementation of ELBE
   True

From the command line, ``--plugin MODULE`` imports a module before the run (so that it
can register implementations, benchmarks or recipes) and ``--implementation NAME``
selects the implementation::

    python -m mowl.benchmarks run box2el-galen-prediction --method elbe --recipe \\
        --plugin mowl.baselines.benchmarks --implementation baseline --out elbe.json
"""

import importlib

__all__ = ["Implementation", "PartsImplementation", "register_implementation",
           "get_implementation", "list_implementations", "load_plugins", "with_run_seed"]


def with_run_seed(callable_, kwargs, seed):
    """``kwargs`` with ``seed`` added if ``callable_`` (a class or a factory) has a
    parameter ``seed`` whose default is a fixed integer and ``kwargs`` do not set it.

    Such a parameter seeds a sampler or a walker of the model; with its fixed default,
    runs with different seeds would share it. A parameter defaulting to ``None`` draws
    from the global generators, which the runner seeds, and is left alone.
    """
    import inspect
    kwargs = dict(kwargs)
    if "seed" in kwargs:
        return kwargs
    target = callable_.__init__ if isinstance(callable_, type) else callable_
    try:
        param = inspect.signature(target).parameters.get("seed")
    except (TypeError, ValueError):
        return kwargs
    if param is not None and isinstance(param.default, int) \
            and not isinstance(param.default, bool):
        kwargs["seed"] = seed
    return kwargs


class Implementation:
    """How the runner obtains the model of a catalogue method.

    Subclasses set :attr:`name` and implement :meth:`build`; :meth:`attach_regime`,
    :meth:`supports` and :meth:`describe` have defaults.
    """

    #: Name under which the implementation is registered (``--implementation NAME``).
    name = ""
    #: One sentence for listings.
    description = ""

    def supports(self, method):
        """``True`` if the implementation has the catalogue method ``method``."""
        return True

    def unsupported_reason(self, method):
        """Why ``method`` cannot be run with this implementation (``""`` if it can)."""
        return "" if self.supports(method) else \
            f"the implementation {self.name!r} has no model for {method!r}"

    def build(self, card, dataset, model_kwargs, device, model_filepath, seed=0):
        """Build the model.

        :param card: The :class:`~mowl.catalog.MethodCard` of the method.
        :param dataset: The dataset prepared by the benchmark's protocol.
        :param model_kwargs: Model arguments (recipe and command line merged).
        :param device: Torch device.
        :param model_filepath: Where the model may write checkpoints.
        :param seed: The seed of the run (see :func:`with_run_seed`).
        :return: ``(model, family)``: the model and the
            :class:`~mowl.benchmarks.runner.ModelFamily` that trains it.
        """
        raise NotImplementedError

    def attach_regime(self, model, regime):
        """Attach a training regime (an instance of :class:`mowl.training.TrainingRegime`)
        to ``model``. The default assigns ``model.training_regime``."""
        from mowl.benchmarks.runner import UnsupportedModelError
        if not hasattr(type(model), "training_regime"):
            raise UnsupportedModelError(
                f"{type(model).__name__} has no 'training_regime' attribute; training "
                f"regimes can only be attached to models that support them.")
        model.training_regime = regime

    def describe(self, model):
        """What is recorded about the implementation in the result (JSON-serialisable)."""
        cls = type(model)
        return {"name": self.name, "model_class": f"{cls.__module__}.{cls.__qualname__}"}


class PartsImplementation(Implementation):
    """The methods users get: the class named by the catalogue card
    (``MethodCard.implementation``), or the card's factory for a method that is a
    configuration of a general class."""

    name = "parts"
    description = "the combination of parts of mOWL (mowl.models), what users get"

    def build(self, card, dataset, model_kwargs, device, model_filepath, seed=0):
        from mowl.benchmarks.runner import family_for
        if card.factory:
            from mowl.benchmarks.families import build_from_factory, load_factory
            kwargs = with_run_seed(load_factory(card.factory), model_kwargs, seed)
            model = build_from_factory(card, dataset, kwargs, device, model_filepath)
            return model, family_for(type(model))
        cls = card.load()
        family = family_for(cls)
        kwargs = with_run_seed(cls, model_kwargs, seed)
        return family.build(cls, dataset, kwargs, device, model_filepath), family


_IMPLEMENTATIONS = {}


def register_implementation(implementation, replace=False):
    """Register an :class:`Implementation` under its :attr:`~Implementation.name`."""
    name = implementation.name
    if not name:
        raise ValueError("An implementation needs a name.")
    # The same class imported twice (as a module and as ``__main__``) is the same.
    if name in _IMPLEMENTATIONS and not replace and \
            type(_IMPLEMENTATIONS[name]).__qualname__ != type(implementation).__qualname__:
        raise ValueError(f"An implementation named {name!r} is already registered.")
    _IMPLEMENTATIONS[name] = implementation
    return implementation


def get_implementation(name):
    """The registered implementation called ``name`` (an :class:`Implementation` is
    returned unchanged)."""
    if isinstance(name, Implementation):
        return name
    if name is None:
        name = "parts"
    if name not in _IMPLEMENTATIONS:
        raise KeyError(
            f"Unknown implementation {name!r}. Registered: {sorted(_IMPLEMENTATIONS)}. "
            f"Implementations of other code bases are registered by importing their "
            f"module, e.g. --plugin mowl.baselines.benchmarks for 'baseline'.")
    return _IMPLEMENTATIONS[name]


def list_implementations():
    """Names of the registered implementations."""
    return sorted(_IMPLEMENTATIONS)


def load_plugins(modules):
    """Import every module of ``modules`` (dotted names); a module may register
    implementations, model families, benchmarks or recipes when it is imported."""
    return [importlib.import_module(m) for m in modules or ()]


register_implementation(PartsImplementation())
