"""The joint mOWL benchmarks: every method that supports a task, on the same data, ranked
the same way.

The benchmarks of the other modules reproduce the evaluation of one publication each,
with its candidates, tie handling and scores. These use one regime
(``common-filtered-ranking``) on the data most publications share, and take the score
from whatever the model declares (:mod:`mowl.evaluation.model_scoring`), so that box
models, cone models, lattice models, knowledge graph embeddings and Word2Vec-based
methods can be compared directly. They have no published results.
"""

from mowl.benchmarks.spec import Benchmark, Task
from mowl.benchmarks.entries._refs import JACKERMEIER2024
from mowl.logic.tasks import Target

_P = "mowl.benchmarks.protocols.bindings.CommonRankingProtocol"
_METRICS = ("f_mrr", "f_hits@1", "f_hits@10", "f_hits@100", "f_mr", "f_med", "f_auc")
_HIB = {m: m not in ("f_mr", "f_med") for m in _METRICS}

_RANKING = (
    "the class D is ranked among all classes of the ontology (C removed) after removing "
    "the other classes known to be correct for the query (training, validation and test "
    "axioms of the same form); tied candidates share the average rank. Filtered MRR, "
    "Hits@1/10/100, mean rank, median rank and AUC.")

_NOTES = (
    "Scores are what each model declares (recorded as 'scoring' in the result): the "
    "centre distance or the loss of normal-form EL models, the order violation of CatE, "
    "the axiom cost of ALC and quantum-logic models, the edge score of a knowledge graph "
    "embedding, or the pair classifier head for models that give vectors only. No "
    "publication reports numbers under this regime; compare runs with each other. ")

_ONTOLOGIES = (
    ("galen", "GALEN", "GALENJackermeier2024Dataset", "GALEN"),
    ("go", "GO", "GOJackermeier2024Dataset", "GO"),
    ("anatomy", "Anatomy", "AnatomyJackermeier2024Dataset", "ANATOMY"),
)

BENCHMARKS = []

for _name, _label, _cls, _onto in _ONTOLOGIES:
    BENCHMARKS.append(Benchmark(
        key=f"common-{_name}-subsumption-prediction",
        name=f"Joint benchmark: subsumption prediction ({_label})",
        task=Task.AXIOM_PREDICTION,
        evaluation_regime="common-filtered-ranking",
        description=(
            "Held-out subsumptions between named classes. Data: the 80/10/10 split of the "
            "Box2EL release. For every test axiom C ⊑ D, " + _RANKING),
        dataset=f"mowl.datasets.builtin.jackermeier2024.{_cls}",
        protocol=_P,
        protocol_kwargs={"gci_name": "gci0"},
        metrics=_METRICS, higher_is_better=_HIB, references=(JACKERMEIER2024,),
        notes=_NOTES + "A held-out axiom may be entailed by the training axioms.",
    ))
    BENCHMARKS.append(Benchmark(
        key=f"common-{_name}-existential-prediction",
        name=f"Joint benchmark: existential restriction prediction ({_label})",
        task=Task.AXIOM_PREDICTION,
        evaluation_regime="common-filtered-ranking",
        completion_tasks=("existential-prediction",),
        description=(
            "Held-out axioms C ⊑ ∃r.D. Data: the 80/10/10 split of the Box2EL release. "
            "For every test axiom, " + _RANKING),
        dataset=f"mowl.datasets.builtin.jackermeier2024.{_cls}",
        protocol=_P,
        protocol_kwargs={"gci_name": "gci2"},
        metrics=_METRICS, higher_is_better=_HIB, references=(JACKERMEIER2024,),
        notes=_NOTES + ("Only methods whose expressivity has existential restrictions "
                        "take part. CatE scores ∃r.D only for fillers that are elements "
                        "of its lattice (model argument 'existentials')."),
    ))
    BENCHMARKS.append(Benchmark(
        key=f"common-{_name}-subsumption-inference",
        name=f"Joint benchmark: subsumption inference ({_label})",
        task=Task.DEDUCTIVE_INFERENCE,
        evaluation_regime="common-filtered-ranking",
        description=(
            "Subsumptions between named classes that are entailed by the training "
            "ontology and not stated in it. Data: the inferences of the Box2EL release "
            "(the full ontology for training, 90% of the inferred subsumptions for "
            "testing, 10% for validation). For every test axiom C ⊑ D, " + _RANKING),
        dataset="mowl.datasets.builtin.jackermeier2024_reference."
                "Jackermeier2024InferencesDataset",
        dataset_kwargs={"ontology": _onto},
        protocol=_P,
        protocol_kwargs={"gci_name": "gci0"},
        metrics=_METRICS, higher_is_better=_HIB, references=(JACKERMEIER2024,),
        target=Target.ENTAILED,
        notes=_NOTES,
    ))

for _org in ("yeast", "human"):
    BENCHMARKS.append(Benchmark(
        key=f"common-{_org}-ppi",
        name=f"Joint benchmark: protein-protein interaction prediction ({_org})",
        task=Task.LINK_PREDICTION,
        evaluation_regime="common-filtered-ranking",
        completion_tasks=("existential-prediction",),
        encodes=("link-prediction",),
        description=(
            "Held-out interactions, with proteins as classes: P1 ⊑ ∃interacts.P2. Data: "
            "the PPI data of the Box2EL release (from ELEmbeddings). For every test "
            "axiom, the protein P2 is ranked among all proteins (P1 removed) after "
            "removing the other proteins known to interact with P1 (training, validation "
            "and test interactions); tied candidates share the average rank. Filtered "
            "MRR, Hits@1/10/100, mean rank, median rank and AUC."),
        dataset="mowl.datasets.builtin.jackermeier2024_reference.Jackermeier2024PPIDataset",
        dataset_kwargs={"organism": _org},
        protocol=_P,
        protocol_kwargs={"gci_name": "gci2", "relation": "http://interacts",
                         "candidates": "evaluation_classes"},
        metrics=_METRICS, higher_is_better=_HIB, references=(JACKERMEIER2024,),
        notes=_NOTES,
    ))
