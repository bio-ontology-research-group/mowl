"""Data model of the mOWL benchmark set.

A :class:`Benchmark` names a dataset, a task and an evaluation protocol.
A :class:`PublishedResult` records one row of a table of a publication, so results
obtained with mOWL can be compared against the literature.

Like the method catalogue, these are plain data and do not need the JVM.
"""

from dataclasses import dataclass, field
from enum import Enum
import importlib

from mowl.catalog.card import Reference


class Task(str, Enum):
    """Evaluation tasks."""

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

    def __post_init__(self):
        if not self.key or self.key != self.key.lower() or " " in self.key:
            raise ValueError(f"Benchmark key must be a lower-case identifier: {self.key!r}")
        if not isinstance(self.task, Task):
            raise TypeError("task must be a mowl.benchmarks.Task.")
        if not self.references:
            raise ValueError(f"Benchmark {self.key!r} needs at least one reference.")
        for r in self.references:
            if not isinstance(r, Reference):
                raise TypeError("references must contain mowl.catalog.Reference objects.")

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
    """

    benchmark: str
    method: str
    metrics: dict
    source: Reference
    locator: str
    subset: str = "all"
    settings: dict = field(default_factory=dict)
    note: str = ""

    def __post_init__(self):
        if not isinstance(self.source, Reference):
            raise TypeError("source must be a mowl.catalog.Reference.")
        if not self.locator:
            raise ValueError("A PublishedResult needs a locator (table or section).")
        if not self.metrics:
            raise ValueError("A PublishedResult needs at least one metric.")
