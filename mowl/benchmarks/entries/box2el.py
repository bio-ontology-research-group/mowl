"""Benchmarks of Box\\ :sup:`2`\\ EL (Jackermeier, Chen, Horrocks, WWW 2024)."""

from mowl.benchmarks.spec import Benchmark, Task
from mowl.benchmarks.entries._refs import JACKERMEIER2024, YANG2025

_P = "mowl.benchmarks.protocols.bindings."
_RANK_METRICS = ("hits@1", "hits@10", "hits@100", "med", "mrr", "mr", "auc")
_HIB = {"hits@1": True, "hits@10": True, "hits@100": True, "med": False, "mrr": True,
        "mr": False, "auc": True}

_PREDICTION_DESCRIPTION = (
    "Subsumption prediction on the 80/10/10 split per normal form of the reference "
    "release. For every test axiom one atomic concept is replaced by every class of the "
    "ontology and the true axiom is ranked by score: NF1 C ⊑ ?D (C itself removed from "
    "the candidates), NF2 C ⊓ D ⊑ ?E (C and D removed), NF3 ?C ⊑ ∃r.D, NF4 ∃r.C ⊑ ?D. "
    "Raw (unfiltered) ranks, ties broken by candidate position; Hits@1/10/100, median "
    "rank, MRR, mean rank and the ROC AUC of the reference code, per normal form "
    "(gci0..gci3 = NF1..NF4) and over all normal forms combined. Disjointness axioms "
    "are not evaluated.")

_PREDICTION_NOTES = (
    "Scores: the centre-distance scores of the paper's Appendix F are used for Box2EL, "
    "ELBE and ELEmbeddings ('auto' scoring); other models are scored by their own loss, "
    "which the result records as scoring='loss' (not comparable with the published "
    "numbers in a strict sense). The paper selects the model with the best validation "
    "performance every 100 epochs and averages 5 seeds; mOWL's EL training loop keeps "
    "the parameters with the lowest validation loss of eval_gci_name. TransBox's numbers "
    "(Yang et al. 2025) use a modified score for NF2 when Box(C) ∩ Box(D) is empty.")

_PREDICTION = [
    Benchmark(
        key=f"box2el-{name}-prediction",
        name=f"Box2EL subsumption prediction ({label})",
        task=Task.AXIOM_PREDICTION,
        description=_PREDICTION_DESCRIPTION,
        dataset=f"mowl.datasets.builtin.jackermeier2024.{cls}",
        protocol=_P + "Box2ELPredictionProtocol",
        metrics=_RANK_METRICS,
        higher_is_better=_HIB,
        subsets=("gci0", "gci1", "gci2", "gci3", "combined"),
        references=(JACKERMEIER2024, YANG2025),
        notes=_PREDICTION_NOTES + extra,
    )
    for name, label, cls, extra in (
        ("galen", "GALEN", "GALENJackermeier2024Dataset", ""),
        ("go", "GO", "GOJackermeier2024Dataset", ""),
        ("anatomy", "Anatomy", "AnatomyJackermeier2024Dataset",
         " Class IRIs of this dataset are opaque (see the dataset class)."),
    )
]

_INFERENCE = [
    Benchmark(
        key=f"box2el-{name}-inference",
        name=f"Box2EL approximating deductive reasoning ({label})",
        task=Task.DEDUCTIVE_INFERENCE,
        description=(
            "Train on the full normalised ontology (all asserted axioms); rank the atomic "
            "subsumptions C ⊑ D inferred by an EL++ reasoner but not asserted (90% test, "
            "10% validation, from the reference release). Ranking as NF1 of the "
            "prediction benchmark: ?D among all classes, C removed, raw ranks, ties by "
            "position; Hits@1/10/100, median rank, MRR, mean rank, AUC (Section 4.4, "
            "Table 6)."),
        dataset="mowl.datasets.builtin.jackermeier2024_reference."
                "Jackermeier2024InferencesDataset",
        dataset_kwargs={"ontology": onto},
        protocol=_P + "InferredSubsumptionProtocol",
        metrics=_RANK_METRICS,
        higher_is_better=_HIB,
        references=(JACKERMEIER2024,),
        notes=("Data: data.zip of KRR-Oxford/BoxSquaredEL (commit a715080), "
               "<ONT>/inferences/. mOWL loads the training ontology with the OWL API; "
               "the reference loader misreads the 21 EquivalentClasses/ObjectOneOf and 63 "
               "ObjectPropertyRange axioms of ANATOMY as atomic subsumptions, so the class "
               "count differs slightly there. Scoring as in the prediction benchmarks."),
    )
    for name, label, onto in (("galen", "GALEN", "GALEN"), ("go", "GO", "GO"),
                              ("anatomy", "Anatomy", "ANATOMY"))
]

_PPI_METRICS = ("hits@10", "f_hits@10", "hits@100", "f_hits@100", "mr", "f_mr", "auc",
                "f_auc")

_PPI = [
    Benchmark(
        key=f"box2el-{org}-ppi",
        name=f"Box2EL PPI prediction ({org})",
        task=Task.LINK_PREDICTION,
        description=(
            "For every test interaction P1 ⊑ ∃interacts.P2, rank P1 among all proteins of "
            "the dataset (scored with r and P2 fixed), ties by position. Filtered metrics "
            "(prefix f_) move the training interactions of the ranked candidates to the "
            "end. AUC: area under the cumulative rank curve divided by the number of "
            "proteins (reference evaluate_ppi.py). Section 4.3, Table 5."),
        dataset="mowl.datasets.builtin.jackermeier2024_reference.Jackermeier2024PPIDataset",
        dataset_kwargs={"organism": org},
        protocol=_P + "PPIRankingProtocol",
        protocol_kwargs={"relation": "http://interacts"},
        metrics=_PPI_METRICS,
        higher_is_better={m: not (m.endswith("mr")) for m in _PPI_METRICS},
        references=(JACKERMEIER2024,),
        notes=("Data: data.zip of KRR-Oxford/BoxSquaredEL, data/PPI/<organism> (the "
               "ELEmbeddings PPI data; 21,560 yeast test links). This is not mOWL's "
               "PPIYeastDataset/PPIHumanDataset (12,040 yeast test axioms), so those are "
               "not used here. The reference filter indexes its training matrix by "
               "(P2, candidate); mOWL filters training axioms (candidate, interacts, P2). "
               "Both are the same because the data are closed under symmetry. The "
               "baseline numbers of Table 5 are copied by the paper from Peng et al. "
               "(ELBE) and Xiong et al. (BoxEL)."),
    )
    for org in ("yeast", "human")
]

BENCHMARKS = _PREDICTION + _INFERENCE + _PPI
