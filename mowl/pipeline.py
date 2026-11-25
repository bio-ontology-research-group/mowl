"""Train and evaluate a method in one call.

:func:`pipeline` takes the name of a method and either the name of a benchmark or a
dataset of your own, builds the model, trains it, evaluates it and returns the trained
model together with the metrics.

.. testcode::

   from mowl.pipeline import pipeline

   result = pipeline("elbe", benchmark="helly-tk", epochs=2,
                     model_kwargs={"embed_dim": 2})
   print(type(result.model).__name__, sorted(result.metrics))

.. testoutput::
   :options: +ELLIPSIS

   ...ELBE ['all']

It is a thin layer over :func:`mowl.benchmarks.runner.run_benchmark`: the methods are those
of :func:`mowl.catalog.list_methods`, the benchmarks those of
:func:`mowl.benchmarks.list_benchmarks`, and every argument of ``run_benchmark`` can be given.
"""

import os
from dataclasses import dataclass

from mowl.benchmarks.protocols.base import Protocol

__all__ = ["pipeline", "PipelineResult", "TrainOnlyProtocol"]


class TrainOnlyProtocol(Protocol):
    """The protocol of :func:`pipeline` when no evaluation is asked for: no metrics."""

    key = ""

    def evaluate(self, model, dataset):
        return {}


@dataclass
class PipelineResult:
    """What :func:`pipeline` returns.

    :param model: The trained model.
    :param dataset: The dataset the model was trained on.
    :param result: The :class:`mowl.benchmarks.runner.BenchmarkResult` of the run: metrics,
        settings, seed, timings and software versions.
    """

    model: object
    dataset: object
    result: object

    @property
    def metrics(self):
        """``{subset: {metric: value}}``; empty when the run had no evaluation."""
        return self.result.metrics

    def metric(self, name, subset=None):
        """The value of one metric.

        :param name: Name of the metric, e.g. ``"mrr"``.
        :param subset: The subset of the results to read it from (default: the only
            subset, or ``"combined"`` / ``"all"`` if the protocol reports one).
        """
        metrics = self.metrics
        if subset is None:
            if len(metrics) == 1:
                subset = next(iter(metrics))
            else:
                subset = next((s for s in ("combined", "all") if s in metrics), None)
                if subset is None:
                    raise ValueError(f"Give a subset; the run has {sorted(metrics)}.")
        return metrics[subset][name]

    def save_to_directory(self, directory):
        """Write the run to ``directory``.

        ``results.json`` is the :class:`~mowl.benchmarks.runner.BenchmarkResult` (read it
        back with ``BenchmarkResult.load``). ``model.pt`` is the ``state_dict`` of the
        model's PyTorch module, for models that have one (``model.module``); load it into
        a model built with the same arguments, which are in ``results.json`` under
        ``options``.

        :return: The paths written, by name.
        """
        os.makedirs(directory, exist_ok=True)
        written = {"results": os.path.join(directory, "results.json")}
        self.result.save(written["results"])
        module = getattr(self.model, "module", None)
        if module is not None and hasattr(module, "state_dict"):
            import torch as th
            written["model"] = os.path.join(directory, "model.pt")
            th.save(module.state_dict(), written["model"])
        return written


def _training_only():
    """The stand-in for a benchmark when a model is only trained: it measures nothing, so
    it has no task, reference or metric (which a registered benchmark must have)."""
    from dataclasses import dataclass as _dataclass
    from mowl.benchmarks.spec import Benchmark, Task

    @_dataclass(frozen=True)
    class TrainingOnly(Benchmark):
        def __post_init__(self):
            for name in ("completion_tasks", "extra_inference_tasks"):
                object.__setattr__(self, name, ())

    return TrainingOnly(
        key="custom-training", name="Training on a dataset, no evaluation",
        task=Task.AXIOM_PREDICTION, description="Train a model; nothing is measured.",
        dataset="", protocol="mowl.pipeline.TrainOnlyProtocol", metrics=(), references=(),
        subsets=())


def _as_dataset(dataset):
    """A :class:`mowl.datasets.Dataset` from a dataset, a path or a mapping of paths."""
    from mowl.datasets import PathDataset
    if isinstance(dataset, (str, os.PathLike)):
        return PathDataset(os.fspath(dataset))
    if isinstance(dataset, dict):
        unknown = set(dataset) - {"ontology", "validation", "testing"}
        if unknown or "ontology" not in dataset:
            raise ValueError("A dataset given as a mapping has the keys 'ontology' and, "
                             "optionally, 'validation' and 'testing'.")
        return PathDataset(os.fspath(dataset["ontology"]),
                           validation_path=dataset.get("validation"),
                           testing_path=dataset.get("testing"))
    return dataset


def pipeline(method=None, benchmark=None, dataset=None, evaluation=None, model_kwargs=None,
             epochs=None, seed=0, device="cpu", training_regime=None, recipe=None,
             model_factory=None, jvm_memory="8g", **options):
    """Build, train and evaluate a model.

    Give ``benchmark`` to run on a registered benchmark, or ``dataset`` to train on data of
    your own; with ``dataset``, ``evaluation`` names the benchmark whose evaluation is
    applied to it, and without ``evaluation`` the model is only trained.

    :param method: Catalogue key of the method (:func:`mowl.catalog.list_methods`), e.g.
        ``"elem"``, ``"box2el"``, ``"owl2vec_star"``.
    :param benchmark: Key of a benchmark (:func:`mowl.benchmarks.list_benchmarks`) or a
        :class:`~mowl.benchmarks.Benchmark`: its dataset, splits and evaluation are used.
    :param dataset: Your own data: a :class:`mowl.datasets.Dataset`, the path of an
        ontology file, or ``{"ontology": path, "validation": path, "testing": path}``.
    :param evaluation: With ``dataset``: key of the benchmark whose evaluation protocol is
        run on the dataset (the dataset must have the splits that protocol reads, usually
        a testing ontology).
    :param model_kwargs: Keyword arguments of the model class, e.g. ``{"embed_dim": 50}``.
    :param epochs: Training epochs (default: the recipe's, otherwise 100).
    :param seed: Random seed.
    :param device: Torch device, e.g. ``"cuda"``.
    :param training_regime: A training regime (instance, class or catalogue key).
    :param recipe: With ``benchmark``: train as a publication did (``True`` or the key of
        the publication, see :func:`mowl.benchmarks.list_recipes`).
    :param model_factory: Instead of ``method``: ``model_factory(dataset) -> model``.
    :param jvm_memory: Heap size if the JVM has to be started.
    :param options: Further arguments of :func:`mowl.benchmarks.runner.run_benchmark`
        (``validate_every``, ``select_best``, ``workdir``, ``protocol_kwargs``, ...).
    :rtype: :class:`PipelineResult`
    """
    from mowl.benchmarks.runner import _ensure_jvm, run_benchmark_with_model
    if (benchmark is None) == (dataset is None):
        raise ValueError("Give either a benchmark or a dataset.")
    if benchmark is not None and evaluation is not None:
        raise ValueError("A benchmark has its own evaluation; 'evaluation' goes with a "
                         "dataset of your own.")
    if method is None and model_factory is None:
        raise ValueError("Give a catalogue method key or a model_factory.")
    if dataset is not None:
        if recipe is not None and evaluation is None:
            raise ValueError("A recipe belongs to a benchmark; give 'evaluation' to use "
                             "the recipe of that benchmark on your dataset.")
        _ensure_jvm(jvm_memory)
        dataset = _as_dataset(dataset)
        if evaluation is None:
            benchmark = _training_only()
        else:
            benchmark = evaluation
    result, model, dataset = run_benchmark_with_model(
        benchmark, method=method, model_factory=model_factory, model_kwargs=model_kwargs,
        epochs=epochs, seed=seed, device=device, training_regime=training_regime,
        dataset=dataset, recipe=recipe, jvm_memory=jvm_memory, **options)
    return PipelineResult(model=model, dataset=dataset, result=result)
