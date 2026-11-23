"""Data model of the mOWL benchmark set.

A :class:`Benchmark` names a dataset, a task and an evaluation protocol.
A :class:`PublishedResult` records one row of a table of a publication, so results
obtained with mOWL can be compared against the literature.

Like the method catalogue, these are plain data and do not need the JVM.
"""

from dataclasses import dataclass, field
from enum import Enum
import importlib
from typing import Optional

from mowl.catalog.card import Reference
from mowl.logic.tasks import COMPLETION_TASKS, TASKS, Target


class Task(str, Enum):
    """Coarse groups of benchmarks, as the literature names them.

    The precise statement of what a benchmark measures is its
    :attr:`Benchmark.completion_tasks` (the form of the ranked axioms,
    :data:`mowl.logic.COMPLETION_TASKS`), its :attr:`Benchmark.inference_tasks` (the
    reasoning tasks behind them, :data:`mowl.logic.TASKS`) and its
    :attr:`Benchmark.target` (:class:`mowl.logic.Target`). Each group has a default for
    these (:data:`TASK_DEFAULTS`); a benchmark states them itself where the default does
    not describe it."""

    #: Predict held-out axioms (a random split of the axioms of an ontology).
    AXIOM_PREDICTION = "axiom_prediction"
    #: Rank the entailed axioms (deductive closure) of the training ontology above
    #: non-entailed ones ("approximate reasoning").
    DEDUCTIVE_INFERENCE = "deductive_inference"
    #: Predict held-out role assertions or ``C ⊑ ∃r.D`` links (e.g. protein-protein
    #: interactions, gene-disease associations).
    LINK_PREDICTION = "link_prediction"
    #: Predict held-out class memberships (e.g. protein functions).
    MEMBERSHIP_PREDICTION = "membership_prediction"
    #: Separate entailed, contradicted and undetermined axioms (logical faithfulness).
    FAITHFULNESS = "faithfulness"
    #: Find an embedding that is an exact model of the ontology (expressivity benchmarks,
    #: such as the Helly family :math:`T_k`).
    EXACT_MODEL = "exact_model"
    #: Rank the answers of logical queries (paths, intersections, unions, negations) that
    #: need an edge missing from the training graph.
    QUERY_ANSWERING = "query_answering"


#: For each :class:`Task`: the completion tasks, the inference tasks beyond those the
#: completion tasks imply, and the target of a benchmark that does not state its own.
TASK_DEFAULTS = {
    Task.AXIOM_PREDICTION: (("subsumption-prediction",), (), Target.UNDETERMINED),
    Task.DEDUCTIVE_INFERENCE: (("subsumption-prediction",), (), Target.ENTAILED),
    Task.LINK_PREDICTION: (("link-prediction",), (), Target.UNDETERMINED),
    Task.MEMBERSHIP_PREDICTION: (("type-prediction",), (), Target.UNDETERMINED),
    Task.FAITHFULNESS: (("axiom-ranking",), (), Target.STATUS),
    # An exact model satisfies every axiom: each check is a subsumption test on the
    # geometry, with the axioms of the ontology (all entailed) as the test set.
    Task.EXACT_MODEL: ((), ("subsumption",), Target.ENTAILED),
    Task.QUERY_ANSWERING: (("multi-hop-query-answering",), (), Target.UNDETERMINED),
}


@dataclass(frozen=True)
class Benchmark:
    """A benchmark: dataset, task and protocol.

    :param key: Unique lower-case identifier, e.g. ``"box2el-galen-prediction"``.
    :param name: Display name.
    :param task: The :class:`Task`.
    :param description: What is measured and how (candidate set, filtering, splits).
    :param dataset: Dotted path of the dataset class (a :class:`mowl.datasets.Dataset`
        subclass or a factory returning one).
    :param protocol: Dotted path of the :class:`mowl.benchmarks.protocols.Protocol`
        subclass that evaluates a trained model on this benchmark.
    :param metrics: Names of the metrics the protocol reports, in display order.
    :param higher_is_better: For each metric name, whether larger values are better.
    :param subsets: Names of the result subsets the protocol reports (e.g. normal forms
        ``"gci0"``, ..., and ``"combined"``). ``("all",)`` if there is only one.
    :param references: Publications that define the benchmark.
    :param dataset_kwargs: Keyword arguments for the dataset class.
    :param protocol_kwargs: Keyword arguments for the protocol class.
    :param notes: Caveats (e.g. how the data differ from the original release).
    :param evaluation_regime: Key of the evaluation regime card
        (:func:`mowl.catalog.get_regime`) of the regime the protocol runs.
    :param completion_tasks: Keys of the completion tasks
        (:data:`mowl.logic.COMPLETION_TASKS`) the benchmark scores, in the form the
        candidates actually have in the data. Default: that of ``task``
        (:data:`TASK_DEFAULTS`).
    :param encodes: Keys of completion tasks that the benchmark poses in another form.
        Example: protein-protein interaction prediction with proteins as classes is
        scored as ``C ⊑ ∃r.?`` (``"existential-prediction"``) and encodes
        ``"link-prediction"``.
    :param extra_inference_tasks: Keys of inference tasks (:data:`mowl.logic.TASKS`)
        beyond those of the completion tasks.
    :param target: Which axioms are the correct answers (:class:`mowl.logic.Target` or
        its value). Default: that of ``task``.
    """

    key: str
    name: str
    task: Task
    description: str
    dataset: str
    protocol: str
    metrics: tuple
    references: tuple
    higher_is_better: dict = field(default_factory=dict)
    subsets: tuple = ("all",)
    dataset_kwargs: dict = field(default_factory=dict)
    protocol_kwargs: dict = field(default_factory=dict)
    notes: str = ""
    evaluation_regime: str = ""
    completion_tasks: tuple = None
    encodes: tuple = ()
    extra_inference_tasks: tuple = None
    target: Target = None
    #: Abilities a method needs beyond the inference tasks: ``"unseen names"`` (the test
    #: data name classes, individuals or entities that training did not see).
    requires: tuple = ()

    @property
    def assumption(self):
        """The :class:`mowl.logic.WorldAssumption` of the benchmark's evaluation regime
        (which candidates count as wrong), or ``None`` if the regime does not state one."""
        if not self.evaluation_regime:
            return None
        from mowl.catalog import get_regime
        from mowl.logic import WorldAssumption
        value = get_regime(self.evaluation_regime).assumption
        return WorldAssumption(value) if value else None

    def __post_init__(self):
        if not self.key or self.key != self.key.lower() or " " in self.key:
            raise ValueError(f"Benchmark key must be a lower-case identifier: {self.key!r}")
        if not isinstance(self.task, Task):
            raise TypeError("task must be a mowl.benchmarks.Task.")
        completion, inference, target = TASK_DEFAULTS[self.task]
        if self.completion_tasks is None:
            object.__setattr__(self, "completion_tasks", completion)
        if self.extra_inference_tasks is None:
            object.__setattr__(self, "extra_inference_tasks", inference)
        object.__setattr__(self, "target",
                           target if self.target is None else Target(self.target))
        for name in ("completion_tasks", "encodes", "extra_inference_tasks"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        for t in self.completion_tasks + self.encodes:
            if t not in COMPLETION_TASKS:
                raise ValueError(f"Benchmark {self.key!r} names the unknown completion task "
                                 f"{t!r}. Known: {sorted(COMPLETION_TASKS)}")
        for t in self.extra_inference_tasks:
            if t not in TASKS:
                raise ValueError(f"Benchmark {self.key!r} names the unknown inference task "
                                 f"{t!r}. Known: {sorted(TASKS)}")
        if not self.inference_tasks:
            raise ValueError(f"Benchmark {self.key!r} is not linked to any inference task.")
        if not self.references:
            raise ValueError(f"Benchmark {self.key!r} needs at least one reference.")
        for r in self.references:
            if not isinstance(r, Reference):
                raise TypeError("references must contain mowl.catalog.Reference objects.")

    @property
    def inference_tasks(self):
        """Keys of the inference tasks (:data:`mowl.logic.TASKS`) that decide the
        candidates of the benchmark: those of its completion tasks, then the extra ones.
        A method can be run on the benchmark if it supports them (see
        :func:`mowl.benchmarks.methods_for`)."""
        out = []
        for t in self.completion_tasks:
            out.append(COMPLETION_TASKS[t].inference_task)
        out.extend(self.extra_inference_tasks)
        return tuple(dict.fromkeys(out))

    def measures(self, task):
        """Whether ``task`` (the key of a completion task or of an inference task) is
        what the benchmark scores, directly or in another form (``encodes``)."""
        if task in self.completion_tasks or task in self.encodes:
            return True
        if task in self.inference_tasks:
            return True
        return any(COMPLETION_TASKS[t].inference_task == task for t in self.encodes)

    @staticmethod
    def _resolve(path):
        module_name, _, attr = path.rpartition(".")
        return getattr(importlib.import_module(module_name), attr)

    def load_dataset(self):
        """Instantiate the dataset (needs the JVM; may download data)."""
        return self._resolve(self.dataset)(**self.dataset_kwargs)

    def load_protocol(self):
        """Instantiate the evaluation protocol."""
        return self._resolve(self.protocol)(self, **self.protocol_kwargs)


@dataclass(frozen=True)
class PublishedResult:
    """One published result: a method on a benchmark, as printed in a paper.

    Only record numbers that were read from the publication itself.

    :param benchmark: Key of the :class:`Benchmark`.
    :param method: Key of the method in :mod:`mowl.catalog`.
    :param metrics: Mapping from metric name to the published value.
    :param source: The publication the numbers are printed in. This is often not the
        paper that introduced the method (baselines are re-run by later papers).
    :param locator: Where in the source, e.g. ``"Table 3"``.
    :param subset: Result subset (see :attr:`Benchmark.subsets`).
    :param settings: Hyperparameters stated by the source (``{"embed_dim": 200}``, ...).
    :param note: Anything a reader must know to interpret the comparison.
    :param recipe: Name of the :class:`~mowl.benchmarks.recipes.Recipe` (of the same source,
        benchmark and method) whose settings these numbers were obtained with. Needed when
        a source reports several variants of one method on one benchmark: a run is then
        compared with the numbers of its own variant. ``None``: not tied to a named recipe
        (the comparison falls back on the ``settings``).
    """

    benchmark: str
    method: str
    metrics: dict
    source: Reference
    locator: str
    subset: str = "all"
    settings: dict = field(default_factory=dict)
    note: str = ""
    recipe: Optional[str] = None

    def __post_init__(self):
        if not isinstance(self.source, Reference):
            raise TypeError("source must be a mowl.catalog.Reference.")
        if not self.locator:
            raise ValueError("A PublishedResult needs a locator (table or section).")
        if not self.metrics:
            raise ValueError("A PublishedResult needs at least one metric.")
