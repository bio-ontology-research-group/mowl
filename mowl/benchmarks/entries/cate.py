"""Benchmarks of CatE (Zhapa-Camacho and Hoehndorf, "Lattice-based ALC ontology
embeddings with saturation", Tables 2-4)."""

from mowl.benchmarks.spec import Benchmark, Task
from mowl.benchmarks.entries._refs import ZHAPA2023

_P = "mowl.benchmarks.protocols.bindings.CatEProtocol"


def _hib(metrics):
    return {m: not m.endswith("mr") or m.endswith("mrr") for m in metrics}


_FULL = ("f_mr", "f_mrr", "f_hits@3", "f_hits@10", "f_hits@100", "f_auc")
_TBOX = ("f_mr", "f_hits@10", "f_hits@100", "f_auc")

_COMMON = ("Filtered metrics only (Section 4.1): candidates that are known positives "
           "are moved to the end of the ranking before the rank is taken. Ties by "
           "position; AUC as mOWL's compute_rank_roc.")
_NOTES = ("CatE's paper does not specify tie handling, AUC variant or score function for "
          "the baselines beyond 'implementations provided in mOWL'; mOWL's evaluator "
          "conventions are used (model loss as score, ties by position, Riemann AUC).")

BENCHMARKS = [
    Benchmark(
        key="cate-ore1-membership", name="CatE class membership prediction (ORE1)",
        task=Task.MEMBERSHIP_PREDICTION,
        description=("Rank every test assertion C(a) against C'(a) for every named concept "
                     "C'; filtered by the training assertions and by their deductive "
                     "closure (C(a) asserted entails D(a) for every superclass D). "
                     f"{_COMMON} Table 2."),
        dataset="mowl.datasets.builtin.zhapa2023.CatEORE1Dataset",
        protocol=_P,
        protocol_kwargs={"gci_name": "class_assertion", "closure": True,
                         "candidates": "classes"},
        metrics=_FULL, higher_is_better=_hib(_FULL), references=(ZHAPA2023,),
        notes=("Data: Zenodo record 13766937 (use_cases.tar.gz, use_cases/ore1); 15,157 test "
               "assertions as in Table 1. The test file also contains 2,342 SubClassOf "
               "axioms, which are not queries. The closure filter is computed with ELK "
               "from the training ontology. " + _NOTES)),
    Benchmark(
        key="cate-go-completion", name="CatE TBox completion (GO)",
        task=Task.AXIOM_PREDICTION,
        description=("Rank every test axiom C ⊑ D against C ⊑ D' for every named concept "
                     "D'; filtered by the training subsumptions. " + _COMMON + " Table 3."),
        dataset="mowl.datasets.builtin.subsumption.GOSubsumptionDataset",
        protocol=_P, protocol_kwargs={"gci_name": "gci0", "candidates": "classes"},
        metrics=_TBOX, higher_is_better=_hib(_TBOX), references=(ZHAPA2023,),
        notes=("Verified: GOSubsumptionDataset has 7,260 validation and 14,521 test "
               "SubClassOf axioms, the counts of Table 1. The Zenodo release also has a "
               "'test_cleaned.csv' with 11,893 pairs whose role in Table 3 is not stated; "
               "the full test set is used. " + _NOTES)),
    Benchmark(
        key="cate-foodon-completion", name="CatE TBox completion (FoodOn)",
        task=Task.AXIOM_PREDICTION,
        description=("Rank every test axiom C ⊑ D against C ⊑ D' for every named concept "
                     "D'; filtered by the training subsumptions. " + _COMMON + " Table 3."),
        dataset="mowl.datasets.builtin.subsumption.FoodOnSubsumptionDataset",
        protocol=_P, protocol_kwargs={"gci_name": "gci0", "candidates": "classes"},
        metrics=_TBOX, higher_is_better=_hib(_TBOX), references=(ZHAPA2023,),
        notes=("Verified: FoodOnSubsumptionDataset has 2,977 validation and 5,957 test "
               "SubClassOf axioms, the counts of Table 1 (the Zenodo 'test_cleaned.csv' "
               "has 3,632 pairs; see the GO benchmark). " + _NOTES)),
    Benchmark(
        key="cate-yeast-ppi", name="CatE PPI prediction (yeast)",
        task=Task.LINK_PREDICTION,
        description=("Rank the tail P2 of every test axiom P1 ⊑ ∃interacts_with.P2 among "
                     "the yeast proteins; filtered by the training interactions. "
                     f"{_COMMON} Table 4 (left: PPI axioms)."),
        dataset="mowl.datasets.builtin.ppi_yeast.PPIYeastDataset",
        protocol=_P,
        protocol_kwargs={"gci_name": "gci2", "relation": "http://interacts_with"},
        metrics=_FULL, higher_is_better=_hib(_FULL), references=(ZHAPA2023,),
        notes=("CatE uses 'the PPI dataset provided in [mOWL]'; verified: the Zenodo "
               "release's ppi/test.owl is byte-identical to PPIYeastDataset's test.owl "
               "(12,040 interactions). The right half of Table 4 (GO subsumptions learned "
               "during training) is not part of this benchmark. " + _NOTES)),
]
