"""Benchmarks of OnT (Yang, Chen, He, Gao, Horrocks, "Language Models as Ontology Encoders",
ISWC 2025): GALEN, GO and ANATOMY under OnT's own normalisation and splits."""

from mowl.benchmarks.spec import Benchmark, Task
from mowl.catalog.card import Reference

YANG2025ONT = Reference(
    "yang2025ont",
    "Yang, H., Chen, J., He, Y., Gao, Y., Horrocks, I. (2025). Language Models as Ontology "
    "Encoders. In The Semantic Web - ISWC 2025. arXiv:2507.14334.",
    "https://arxiv.org/abs/2507.14334")

_P = "mowl.benchmarks.protocols.ont."
_D = "mowl.datasets.builtin.yang2025ont.Yang2025OnTDataset"
_METRICS = ("hits@1", "hits@10", "hits@100", "med", "mrr", "mr", "auc")
_HIB = {"hits@1": True, "hits@10": True, "hits@100": True, "med": False, "mrr": True,
        "mr": False, "auc": True}

_COMMON_NOTES = (
    "Data: the authors' release on Zenodo (record 17314434), converted to OWL with the "
    "release's concept texts as rdfs:label (mowl.datasets.builtin.yang2025ont). The "
    "ontologies are normalised by OnT's own name-preserving normaliser, so concepts, "
    "axioms and splits differ from the box2el-* benchmarks, and results are not "
    "comparable with them. Train text models with verbalisation_style='release' to "
    "reproduce the texts of the release. The paper's metric table reports H@k and MRR as "
    "percentages rounded to integers; mOWL stores fractions. Trivial answers: OnT's "
    "normaliser names every complex concept by its verbalisation and keeps the "
    "definitional axioms, so in the test sets of the release the answer concept has "
    "exactly the query's text for almost all NF4 axioms (GALEN 787/788, ANATOMY 949/951, "
    "GO 10,245/10,245), for 787/3,165, 949/4,972 and 10,245/29,865 NF3 axioms and "
    "178/1,462, 174/1,543 and 329/3,014 NF2 axioms (counted on the release). A text "
    "encoder ranks these answers first (distance 0) without training; the paper does "
    "not discuss this.")

_LABELS = (("galen", "GALEN"), ("go", "GO"), ("anatomy", "ANATOMY"))

_PREDICTION = [
    Benchmark(
        key=f"ont-{name}-prediction",
        name=f"OnT axiom prediction ({label})",
        task=Task.AXIOM_PREDICTION,
        evaluation_regime="ont-normal-form-ranking",
        completion_tasks=("subsumption-prediction", "existential-prediction"),
        description=(
            "Random 80/10/10 split of the normalised axioms (Section 5.1, Table 1). For "
            "every test axiom one concept is ranked among all concepts of the normalised "
            "ontology (including the named concepts introduced by normalisation, without "
            "removing trivial answers): NF1 C ⊑ ?D, NF2 C ⊓ D ⊑ ?E, NF3 ?C ⊑ ∃r.D, NF4 "
            "∃r.C ⊑ ?D. Text models score with OnT Eq. 8, the complex side embedded from "
            "its verbalisation, λ chosen on validation. Raw ranks, ties by position; "
            "Hits@1/10/100, median rank, MRR, mean rank and ROC AUC per normal form and "
            "combined (Table 2 reports the combined ranking)."),
        dataset=_D,
        dataset_kwargs={"ontology": label, "task": "prediction"},
        protocol=_P + "OnTPredictionProtocol",
        metrics=_METRICS,
        higher_is_better=_HIB,
        subsets=("gci0", "gci1", "gci2", "gci3", "combined"),
        references=(YANG2025ONT,),
        notes=_COMMON_NOTES,
    )
    for name, label in _LABELS
]

_INFERENCE = [
    Benchmark(
        key=f"ont-{name}-inference",
        name=f"OnT axiom inference ({label})",
        task=Task.DEDUCTIVE_INFERENCE,
        evaluation_regime="ont-inferred-subsumption-ranking",
        description=(
            "Train on the whole normalised ontology; rank the superclass of every "
            "inferred atomic subsumption C ⊑ D (NF1) among all concepts, as NF1 of the "
            "prediction benchmark; 1,000 inferred subsumptions for validation (Section "
            "5.1, Section 5.3, Table 3)."),
        dataset=_D,
        dataset_kwargs={"ontology": label, "task": "inference"},
        protocol=_P + "OnTInferenceProtocol",
        metrics=_METRICS,
        higher_is_better=_HIB,
        references=(YANG2025ONT,),
        notes=_COMMON_NOTES + (
            " The test set has the inferred NF1 counts of Table 1; none of its axioms is "
            "an asserted NF1 training axiom and none is trivial (C ⊑ C), checked on the "
            "GALEN and ANATOMY releases."),
    )
    for name, label in _LABELS
]

BENCHMARKS = _PREDICTION + _INFERENCE
