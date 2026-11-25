"""Run a method on a benchmark and compare the result with the published numbers.

.. code-block:: python

   from mowl.benchmarks.runner import run_benchmark, compare

   result = run_benchmark("box2el-galen-prediction", method="box2el",
                          model_kwargs={"embed_dim": 200}, epochs=1000, seed=0,
                          device="cuda")
   result.save("result.json")
   for row in compare(result):
       print(row["subset"], row["metric"], row["ours"], row["published"])

The same is available from the command line (``python -m mowl.benchmarks``). Runs are
deterministic for a given seed on a given device and software stack: Python's, NumPy's
and PyTorch's generators are seeded, and PyTorch's deterministic algorithms are requested
(``warn_only``; some CUDA kernels have no deterministic variant).
"""

from dataclasses import dataclass, field, asdict
import datetime
import inspect
import json
import math
import os
import platform
import random
import subprocess
import tempfile
import time

from mowl.benchmarks.registry import get_benchmark, published_results
from mowl.benchmarks.spec import Benchmark

__all__ = ["UnsupportedModelError", "BenchmarkResult", "ModelFamily", "ELFamily",
           "register_family", "family_for", "set_seed", "environment",
           "resolve_training_regime", "run_benchmark", "settings_differences", "compare"]


class UnsupportedModelError(TypeError):
    """The model cannot be run on the benchmark (no family handler applies)."""


# ---------------------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    """Outcome of :func:`run_benchmark`; serialisable to JSON.

    :param benchmark: Benchmark key.
    :param method: Catalogue key of the method (``None`` for a custom model factory).
    :param metrics: ``{subset: {metric: value}}``.
    :param settings: Hyperparameters of the run: the given model keyword arguments and
        the values read back from the model (``embed_dim``, ``learning_rate``, ...).
    :param epochs: Training epochs.
    :param seed: Random seed.
    :param device: Torch device.
    :param evaluation_regime: Key of the evaluation regime of the benchmark's protocol.
    :param training_regime: Key of the training regime attached to the model, if any.
    :param dataset_kwargs: Dataset arguments actually used.
    :param protocol_kwargs: Protocol arguments actually used.
    :param protocol_info: Details reported by the protocol (scoring used, query counts).
    :param wall_time: Seconds, split into ``load``, ``train``, ``evaluate``, ``total``.
    :param environment: Package version, git commit, Python/PyTorch versions, host.
    :param created: UTC timestamp.
    :param implementation: Which implementation of the method ran
        (:mod:`mowl.benchmarks.implementations`): ``{"name": "parts", "model_class": ...}``.
    :param recipe: The recipe used, ``{"source", "name", "epochs", "training_regime",
        "regime_kwargs", "validate_every"}``, or ``None``.
    :param options: Every argument of the run (model, dataset, protocol and regime
        arguments, epochs, seed, validation interval, model selection, determinism).
    """

    benchmark: str
    method: object
    metrics: dict
    settings: dict = field(default_factory=dict)
    epochs: int = 0
    seed: int = 0
    device: str = "cpu"
    evaluation_regime: str = ""
    training_regime: object = None
    dataset_kwargs: dict = field(default_factory=dict)
    protocol_kwargs: dict = field(default_factory=dict)
    protocol_info: dict = field(default_factory=dict)
    wall_time: dict = field(default_factory=dict)
    environment: dict = field(default_factory=dict)
    created: str = ""
    implementation: dict = field(default_factory=lambda: {"name": "parts"})
    recipe: object = None
    options: dict = field(default_factory=dict)

    def to_dict(self):
        return _jsonable(asdict(self))

    def to_json(self, indent=2):
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def save(self, path):
        """Write the result as JSON to ``path``."""
        with open(path, "w") as f:
            f.write(self.to_json())

    @classmethod
    def from_dict(cls, data):
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})

    @classmethod
    def load(cls, path):
        """Read a result written by :meth:`save`."""
        with open(path) as f:
            return cls.from_dict(json.load(f))


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, float):
        return None if math.isnan(obj) else obj
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    if hasattr(obj, "item"):
        try:
            return _jsonable(obj.item())
        except (ValueError, TypeError):
            pass
    return str(obj)


# ---------------------------------------------------------------------------------------
# Model families
# ---------------------------------------------------------------------------------------

class ModelFamily:
    """How the runner builds and trains models of one base class.

    Register new families with :func:`register_family`; the first family whose
    :meth:`matches` returns ``True`` for the model class is used.
    """

    name = ""

    def matches(self, cls):
        raise NotImplementedError

    def build(self, cls, dataset, model_kwargs, device, model_filepath):
        raise NotImplementedError

    def train(self, model, epochs, protocol, validate_every=None, select_best=True):
        raise NotImplementedError


def _accepts(callable_, name):
    params = inspect.signature(callable_).parameters
    return name in params or any(p.kind is p.VAR_KEYWORD for p in params.values())


class ELFamily(ModelFamily):
    """:class:`~mowl.base_models.EmbeddingELModel` subclasses."""

    name = "el"

    def matches(self, cls):
        from mowl.base_models.elmodel import EmbeddingELModel
        return isinstance(cls, type) and issubclass(cls, EmbeddingELModel)

    def build(self, cls, dataset, model_kwargs, device, model_filepath):
        kwargs = dict(model_kwargs)
        if _accepts(cls.__init__, "device"):
            kwargs.setdefault("device", device)
        if _accepts(cls.__init__, "model_filepath"):
            kwargs.setdefault("model_filepath", model_filepath)
        return cls(dataset, **kwargs)

    @staticmethod
    def _eval_gci(model, protocol):
        wanted = getattr(getattr(protocol, "regime", protocol), "gci_name", None)
        try:
            available = model.validation_datasets or {}
        except AttributeError:
            available = {}
        for name in (wanted, "gci0", "gci2", "gci1", "gci3"):
            if name and name in available and len(available[name]) > 0:
                return name
        return None

    def train(self, model, epochs, protocol, validate_every=None, select_best=True):
        validated = False
        if model.dataset.validation is not None and getattr(model, "eval_gci_name", 1) is None:
            gci = self._eval_gci(model, protocol)
            if gci is not None:
                model.eval_gci_name = gci
                validated = True
        elif model.dataset.validation is not None:
            validated = True
        kwargs = {}
        if validate_every is not None and _accepts(model.train, "validate_every"):
            kwargs["validate_every"] = validate_every
        selection = self._selection_regime(model)
        if selection is not None and validate_every is not None:
            selection.every = max(1, int(validate_every))
        model.train(epochs=epochs, **kwargs)
        eval_gci = getattr(model, "eval_gci_name", None)
        if selection is not None:
            # The regime has already loaded the parameters it selected.
            return {"selected": selection.key, "selected_epoch": selection.best_epoch,
                    "selection_value": selection.best_value, "eval_gci_name": eval_gci}
        if select_best and validated and os.path.exists(model.model_filepath):
            model.load_best_model()
        criterion = getattr(model, "validation_criterion", "loss")
        return {"selected": f"best validation {criterion}" if (select_best and validated)
                else "last epoch", "eval_gci_name": eval_gci}

    @staticmethod
    def _selection_regime(model):
        """The model selection regime attached to the model, if any."""
        from mowl.training.selection import ValidationSelection
        stack = [getattr(model, "training_regime", None)]
        while stack:
            part = stack.pop(0)
            if isinstance(part, ValidationSelection):
                return part
            stack.extend(getattr(part, "regimes", None) or ())
        return None


class _UnsupportedFamily(ModelFamily):

    def __init__(self, name, base_path, reason):
        self.name = name
        self.base_path = base_path
        self.reason = reason

    def matches(self, cls):
        import importlib
        module_name, _, attr = self.base_path.rpartition(".")
        try:
            base = getattr(importlib.import_module(module_name), attr)
        except (ImportError, AttributeError):
            return False
        return isinstance(cls, type) and issubclass(cls, base)

    def build(self, cls, *args, **kwargs):
        raise UnsupportedModelError(
            f"{cls.__name__} is a {self.name} model. {self.reason} Pass a model_factory "
            f"that returns a model with an EL module, or register a ModelFamily with "
            f"mowl.benchmarks.runner.register_family().")


_FAMILIES = [
    ELFamily(),
]


class _LazyFamily:
    """A :class:`ModelFamily` defined in another module, imported on first use (so that
    optional families do not have to be importable when the runner is)."""

    def __init__(self, path):
        self._path = path
        self._family = None

    def _resolve(self):
        if self._family is None:
            import importlib
            module_name, _, attr = self._path.rpartition(".")
            self._family = getattr(importlib.import_module(module_name), attr)()
        return self._family

    def __getattr__(self, name):
        return getattr(self._resolve(), name)


#: Language-model based methods (HiT, OnT): they train and select their checkpoint
#: themselves.
_FAMILIES.insert(0, _LazyFamily("mowl.benchmarks.protocols.ont.LanguageModelFamily"))
#: Models obtained by solving an optimisation problem once (BoxLitE): ``epochs`` is ignored.
_FAMILIES.insert(0, _LazyFamily("mowl.benchmarks.protocols.boxlite.SolverModelFamily"))


#: Every other model that declares how it scores axioms (ALC, quantum logic, graph and
#: corpus models); tried after the families above.
_FAMILIES.append(_LazyFamily("mowl.benchmarks.families.ScoredModelFamily"))


def register_family(family, first=True):
    """Add a :class:`ModelFamily` handler (before the built-in ones by default)."""
    if first:
        _FAMILIES.insert(0, family)
    else:
        _FAMILIES.append(family)


def family_for(cls):
    """Return the :class:`ModelFamily` handling ``cls``.

    :raises UnsupportedModelError: If none does.
    """
    for fam in _FAMILIES:
        if fam.matches(cls):
            return fam
    raise UnsupportedModelError(
        f"No benchmark runner support for {getattr(cls, '__name__', cls)!r}: it is neither "
        f"an EmbeddingELModel subclass nor a Model that declares how it scores axioms "
        f"(class attribute 'axiom_scoring', see mowl.evaluation.model_scoring). Register a "
        f"ModelFamily or pass a model_factory.")


# ---------------------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------------------

def set_seed(seed, deterministic=True):
    """Seed Python, NumPy and PyTorch (CPU and CUDA).

    Python's string hashes are randomised per process unless ``PYTHONHASHSEED`` is set
    before the interpreter starts; code that seeds from ``hash`` (gensim's default
    ``hashfxn``) then differs between processes. That cannot be changed from inside the
    process; :func:`environment` records the variable."""
    import numpy as np
    import torch as th
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        th.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(th.backends, "cudnn"):
            th.backends.cudnn.benchmark = False
            th.backends.cudnn.deterministic = True


def environment():
    """Package version, git commit and versions of the main dependencies."""
    info = {"python": platform.python_version(), "host": platform.node(),
            "pythonhashseed": os.environ.get("PYTHONHASHSEED")}
    try:
        from importlib.metadata import version, PackageNotFoundError
        for dist in ("mowl-borg", "mowl"):
            try:
                info["mowl_version"] = version(dist)
                break
            except PackageNotFoundError:
                continue
    except ImportError:  # pragma: no cover
        pass
    try:
        import torch as th
        info["torch"] = th.__version__
        info["cuda"] = th.version.cuda
    except ImportError:  # pragma: no cover
        pass
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=here, capture_output=True,
                                text=True, timeout=10)
        if commit.returncode == 0:
            info["git_commit"] = commit.stdout.strip()
            dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                                   cwd=here, capture_output=True, text=True, timeout=10)
            info["git_dirty"] = bool(dirty.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return info


def _ensure_jvm(memory):
    import jpype
    if not jpype.isJVMStarted():
        import mowl
        mowl.init_jvm(memory)


def resolve_training_regime(regime, seed=None, implementation=None):
    """An instance of :class:`mowl.training.TrainingRegime` from an instance, a class, a
    catalogue key, a ``(key, kwargs)`` pair, or a list of these.

    With ``implementation`` (an :class:`~mowl.benchmarks.implementations.Implementation`), a
    regime class is passed through its ``regime_class`` before it is built, so that an
    implementation whose models cannot take the library's class uses its own.

    With ``seed``, a regime built here whose constructor takes ``seed`` gets it unless
    its arguments set one (its samplers would otherwise draw from fresh entropy, see
    :func:`mowl.benchmarks.implementations.with_run_seed`); instances are left alone.

    A list is combined with :class:`mowl.training.ComposedRegime`, in the order given
    (for negatives, loss and optimizer the first regime that implements the hook
    decides). A string may carry constructor arguments as a JSON object after a colon:
    ``'per-normal-form-minibatch:{"batch_size": 512}'``.
    """
    if regime is None:
        return None
    if isinstance(regime, list):
        parts = [resolve_training_regime(r, seed, implementation) for r in regime]
        parts = [r for r in parts if r is not None]
        if not parts:
            return None
        if len(parts) == 1:
            return parts[0]
        from mowl.training import ComposedRegime
        return ComposedRegime(*parts)
    kwargs = {}
    if isinstance(regime, tuple):
        regime, kwargs = regime
        kwargs = dict(kwargs or {})
    if isinstance(regime, str):
        if ":" in regime:
            import json
            regime, raw = regime.split(":", 1)
            kwargs = {**json.loads(raw), **kwargs}
        from mowl.catalog import get_regime
        card = get_regime(regime)
        if card.kind.value != "training":
            raise ValueError(f"Regime {regime!r} is an {card.kind.value} regime, not a "
                             f"training regime.")
        regime = card.load()
    if isinstance(regime, type) and implementation is not None:
        # An implementation may train with a regime class of its own for the same name.
        regime = implementation.regime_class(regime)
    if isinstance(regime, type):
        if seed is not None:
            from mowl.benchmarks.implementations import with_run_seed
            kwargs = with_run_seed(regime, kwargs, seed)
        regime = regime(**kwargs)
    elif kwargs:
        raise ValueError("Constructor arguments were given for a regime instance.")
    return regime


def _regime_key(regime):
    if regime is None:
        return None
    from mowl.training import ComposedRegime
    parts = getattr(regime, "regimes", None)
    if parts and type(regime) is ComposedRegime:
        return [_regime_key(r) for r in parts]
    return getattr(regime, "key", "") or type(regime).__name__


def _read_settings(model, model_kwargs):
    settings = dict(model_kwargs)
    for name in ("embed_dim", "learning_rate", "margin", "batch_size", "reg_norm",
                 "reg_factor", "delta", "gamma", "n_components", "seed"):
        value = getattr(model, name, None)
        if isinstance(value, (int, float, str, bool)) and name not in settings:
            settings[name] = value
    settings["model_class"] = f"{type(model).__module__}.{type(model).__name__}"
    return settings


def resolve_recipe(benchmark, method, recipe):
    """The :class:`~mowl.benchmarks.recipes.Recipe` that a ``recipe`` argument of
    :func:`run_benchmark` names.

    :param recipe: ``True``, the key of the publication, ``"key:name"`` (one publication,
        several recipes -- e.g. the model variants of DELE), or a :class:`Recipe`.
    """
    from mowl.benchmarks.recipes import Recipe, get_recipe
    if isinstance(recipe, Recipe):
        return recipe
    if method is None:
        raise ValueError("A recipe needs a catalogue method key.")
    if recipe is True:
        return get_recipe(benchmark, method, None)
    source, _, name = recipe.partition(":")
    return get_recipe(benchmark, method, source or None, name or None)


def run_benchmark(benchmark, method=None, model_factory=None, model_kwargs=None,
                  epochs=None, seed=0, device="cpu", training_regime=None, dataset=None,
                  dataset_kwargs=None, protocol_kwargs=None, validate_every=None,
                  select_best=True, workdir=None, jvm_memory="8g", deterministic=True,
                  recipe=None, implementation="parts"):
    """Train a method on a benchmark and evaluate it with the benchmark's protocol.

    :param benchmark: A :class:`~mowl.benchmarks.Benchmark` or its key.
    :param method: Catalogue key of the method (``mowl.catalog.get_method(key).load()``).
    :param model_factory: Alternatively, ``model_factory(dataset) -> model``; called after
        seeding. ``model_kwargs`` are then ignored.
    :param model_kwargs: Keyword arguments of the model class (e.g. ``embed_dim``).
    :param epochs: Training epochs (default: the recipe's, otherwise 100).
    :param seed: Random seed (Python, NumPy, PyTorch).
    :param device: Torch device, e.g. ``"cuda"``.
    :param training_regime: A :class:`mowl.training.TrainingRegime` (instance, class or
        catalogue key) assigned to ``model.training_regime`` before training.
    :param dataset: An already loaded dataset (skips loading; ``dataset_kwargs`` are then
        only recorded).
    :param dataset_kwargs: Overrides of the benchmark's dataset arguments.
    :param protocol_kwargs: Overrides of the benchmark's protocol arguments.
    :param validate_every: Passed to ``model.train`` when supported.
    :param select_best: Reload the parameters with the best validation value (by
        ``validation_criterion`` of the model; ``model_kwargs`` may set it) after
        training (EL models with a validation set).
    :param workdir: Directory for model checkpoints (default: a new directory under
        the system temporary directory, which honours ``$TMPDIR``).
    :param jvm_memory: Heap size if the runner has to start the JVM.
    :param deterministic: Request deterministic PyTorch algorithms.
    :param recipe: Train as a publication did: ``True`` (the only recipe of the method on
        the benchmark), the key of the publication (``"source"``), the key and a recipe
        name (``"source:name"``, when one publication has several recipes for the pair,
        e.g. the model variants of DELE), or a
        :class:`~mowl.benchmarks.recipes.Recipe`. The recipe supplies model arguments,
        training regime, epochs and validation interval; arguments given explicitly
        take precedence (``model_kwargs`` are merged).
    :param implementation: Which implementation of ``method`` to run: ``"parts"`` (the
        default, what users get) or the name of another registered
        :class:`~mowl.benchmarks.implementations.Implementation` (``"baseline"`` once
        :mod:`mowl.baselines.benchmarks` is imported).
    :rtype: :class:`BenchmarkResult`
    """
    return run_benchmark_with_model(
        benchmark, method=method, model_factory=model_factory, model_kwargs=model_kwargs,
        epochs=epochs, seed=seed, device=device, training_regime=training_regime,
        dataset=dataset, dataset_kwargs=dataset_kwargs, protocol_kwargs=protocol_kwargs,
        validate_every=validate_every, select_best=select_best, workdir=workdir,
        jvm_memory=jvm_memory, deterministic=deterministic, recipe=recipe,
        implementation=implementation)[0]


def run_benchmark_with_model(benchmark, method=None, model_factory=None, model_kwargs=None,
                             epochs=None, seed=0, device="cpu", training_regime=None,
                             dataset=None, dataset_kwargs=None, protocol_kwargs=None,
                             validate_every=None, select_best=True, workdir=None,
                             jvm_memory="8g", deterministic=True, recipe=None,
                             implementation="parts"):
    """:func:`run_benchmark`, returning the trained model and the dataset as well.

    :return: ``(result, model, dataset)``: the :class:`BenchmarkResult`, the trained model
        and the dataset it was trained on (as prepared by the benchmark's protocol).
    """
    from mowl.benchmarks.implementations import get_implementation
    t0 = time.time()
    impl = get_implementation(implementation)
    if model_factory is not None and impl.name != "parts":
        raise ValueError("A model_factory replaces the implementation; give one or the "
                         "other.")
    if method is not None and model_factory is None and not impl.supports(method):
        raise UnsupportedModelError(impl.unsupported_reason(method))
    given = {"model_kwargs": dict(model_kwargs or {}), "epochs": epochs,
             "training_regime": training_regime, "validate_every": validate_every}
    if not isinstance(benchmark, Benchmark):
        benchmark = get_benchmark(benchmark)
    if method is None and model_factory is None:
        raise ValueError("Give a catalogue method key or a model_factory.")
    model_kwargs = dict(model_kwargs or {})
    if recipe is not None and recipe is not False:
        recipe = resolve_recipe(benchmark.key, method, recipe)
        model_kwargs = {**recipe.model_kwargs, **model_kwargs}
        if training_regime is None:
            training_regime = recipe.regime_spec()
        if epochs is None:
            epochs = recipe.epochs
        if validate_every is None:
            validate_every = recipe.validate_every
        recipe_info = {"source": recipe.source.key, "name": recipe.name,
                       "epochs": recipe.epochs, "training_regime": recipe.training_regime,
                       "regime_kwargs": dict(recipe.regime_kwargs),
                       "validate_every": recipe.validate_every,
                       "model_kwargs": dict(recipe.model_kwargs)}
    else:
        recipe = None
        recipe_info = None
    if epochs is None:
        epochs = 100
    _ensure_jvm(jvm_memory)
    set_seed(seed, deterministic)

    d_kwargs = {**benchmark.dataset_kwargs, **(dataset_kwargs or {})}
    p_kwargs = {**benchmark.protocol_kwargs, **(protocol_kwargs or {})}
    protocol = Benchmark._resolve(benchmark.protocol)(benchmark, **p_kwargs)
    regime_obj = getattr(protocol, "regime", None)
    if regime_obj is not None and hasattr(regime_obj, "device"):
        regime_obj.device = device
    if dataset is None:
        dataset = Benchmark._resolve(benchmark.dataset)(**d_kwargs)
    dataset = protocol.prepare(dataset)
    t_load = time.time()

    if workdir is None:
        base = tempfile.gettempdir()
        os.makedirs(base, exist_ok=True)
        workdir = tempfile.mkdtemp(prefix=f"mowl_bench_{benchmark.key}_", dir=base)
    os.makedirs(workdir, exist_ok=True)
    model_filepath = os.path.join(workdir, f"model_seed{seed}.pt")

    # (not a constructor argument: what the built-in validation keeps the best state by)
    validation_criterion = model_kwargs.pop("validation_criterion", None)
    set_seed(seed, deterministic)
    if model_factory is not None:
        model = model_factory(dataset)
        family = family_for(type(model))
    else:
        from mowl.catalog import get_method
        card = get_method(method)
        model, family = impl.build(card, dataset, model_kwargs, device, model_filepath,
                                   seed=seed)

    if validation_criterion is not None:
        if not hasattr(model, "validation_criterion"):
            raise ValueError(f"{type(model).__name__} has no built-in validation to choose a "
                             f"criterion for.")
        model.validation_criterion = validation_criterion
    regime = resolve_training_regime(training_regime, seed=seed, implementation=impl)
    if regime is not None:
        impl.attach_regime(model, regime)
    elif getattr(model, "training_regime", None) is not None:
        regime = model.training_regime

    initialise = getattr(protocol, "initialise_model", None)
    if initialise is not None:
        initialise(model, dataset)

    train_info = family.train(model, epochs, protocol, validate_every=validate_every,
                              select_best=select_best)
    t_train = time.time()
    metrics = protocol.evaluate(model, dataset)
    t_eval = time.time()

    info = dict(getattr(protocol, "info", {}) or {})
    info["training"] = train_info
    result = BenchmarkResult(
        benchmark=benchmark.key, method=method, metrics=_jsonable(metrics),
        settings=_jsonable(_read_settings(model, model_kwargs)), epochs=epochs, seed=seed,
        device=device, evaluation_regime=getattr(protocol, "key", ""),
        training_regime=_regime_key(regime), dataset_kwargs=_jsonable(d_kwargs),
        protocol_kwargs=_jsonable(p_kwargs), protocol_info=_jsonable(info),
        wall_time={"load": t_load - t0, "train": t_train - t_load,
                   "evaluate": t_eval - t_train, "total": t_eval - t0},
        environment=environment(),
        created=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        implementation=_jsonable(impl.describe(model)), recipe=_jsonable(recipe_info),
        options=_jsonable({
            "given": given, "model_kwargs": model_kwargs, "epochs": epochs,
            "seed": seed, "device": device, "training_regime": training_regime,
            "validate_every": validate_every, "select_best": select_best,
            "deterministic": deterministic, "dataset_kwargs": d_kwargs,
            "protocol_kwargs": p_kwargs, "jvm_memory": jvm_memory, "workdir": workdir,
            "model_factory": None if model_factory is None else repr(model_factory)}))
    return result, model, dataset


# ---------------------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------------------

def _norm_regime(value):
    if value is None or value == () or value == []:
        return None
    if isinstance(value, (list, tuple)):
        return tuple(_norm_regime(v) for v in value)
    return value


def settings_differences(ours, published):
    """Settings stated by the publication that differ from (or are absent in) ours, as
    ``{name: (ours, published)}``."""
    diff = {}
    for key, value in published.items():
        if key == "training_regime":
            continue
        mine = ours.get(key)
        if mine != value:
            diff[key] = (mine, value)
    return diff


def compare(result, method=None, published=None):
    """Compare a result with the published numbers of its benchmark.

    :param result: A :class:`BenchmarkResult` (or a dict / path to a saved one).
    :param method: Compare against this method's published results (default: the
        result's method; all methods if the result has none).
    :param published: An explicit list of :class:`~mowl.benchmarks.PublishedResult` to
        compare with (default: from the registry).
    :return: One dict per (published result, subset, metric) with keys ``benchmark``,
        ``method``, ``subset``, ``metric``, ``ours``, ``published``, ``difference``
        (ours - published), ``higher_is_better``, ``source``, ``locator``,
        ``settings_differences``, ``training_regime`` (ours, published),
        ``regime_match``, ``evaluation_regime`` and ``note``.
    """
    if isinstance(result, str):
        result = BenchmarkResult.load(result)
    elif isinstance(result, dict):
        result = BenchmarkResult.from_dict(result)
    bench = get_benchmark(result.benchmark)
    method = method if method is not None else result.method
    if published is None:
        published = published_results(benchmark=result.benchmark, method=method)
    ours_regime = _norm_regime(result.training_regime)
    rows = []
    for pub in published:
        theirs_regime = _norm_regime(pub.settings.get("training_regime"))
        ours_subset = result.metrics.get(pub.subset, {})
        for metric in bench.metrics:
            if metric not in pub.metrics:
                continue
            ours = ours_subset.get(metric)
            value = pub.metrics[metric]
            rows.append({
                "benchmark": result.benchmark, "method": pub.method, "subset": pub.subset,
                "metric": metric, "ours": ours, "published": value,
                "difference": None if ours is None else ours - value,
                "higher_is_better": bench.higher_is_better.get(metric),
                "source": pub.source.key, "locator": pub.locator,
                "settings_differences": settings_differences(result.settings, pub.settings),
                "training_regime": (ours_regime, theirs_regime),
                "regime_match": ours_regime == theirs_regime,
                "evaluation_regime": result.evaluation_regime,
                "note": pub.note,
            })
    return rows
