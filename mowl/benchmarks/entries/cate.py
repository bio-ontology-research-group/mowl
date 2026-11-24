"""Benchmarks of CatE (Zhapa-Camacho and Hoehndorf, "Lattice-based ALC ontology
embeddings with saturation", Tables 2-4).

The evaluation regimes follow the reference code, not the paper's prose; the
differences are documented on the regime cards ('cate-filtered-ranking',
'cate-ppi-ranking') and in the notes of every benchmark.
"""

from mowl.benchmarks.spec import Benchmark, Task
from mowl.benchmarks.entries._refs import ZHAPA2023

_P = "mowl.benchmarks.protocols.bindings.CatEProtocol"
_PPPI = "mowl.benchmarks.protocols.bindings.CatEPPIProtocol"


def _hib(metrics):
    return {m: not m.endswith("mr") or m.endswith("mrr") for m in metrics}


_FULL = ("f_mr", "f_mrr", "f_hits@3", "f_hits@10", "f_hits@100", "f_auc")
_TBOX = ("f_mr", "f_hits@10", "f_hits@100", "f_auc")

_COMMON = ("The evaluation is the reference code's (Section 4.1 and src/*.py): "
           "filtered metrics only, ties by position, Hits@k with rank < k, the "
           "trapezoid AUC of the code's compute_rank_roc, candidates = all named "
           "classes (including owl:Thing and owl:Nothing).")

_NOTES = ("CatE's paper does not specify tie handling, the AUC variant or the scores "
          "of the baselines beyond 'implementations provided in mOWL'; the "
          "conventions of the reference code are used (order-violation scores, ties "
          "by position, Hits@k with rank < k, trapezoid AUC).")

BENCHMARKS = [
    Benchmark(
        key="cate-ore1-membership", name="CatE class membership prediction (ORE1)",
        task=Task.MEMBERSHIP_PREDICTION,
        description=("Rank every test assertion C(a) against C'(a) for every named "
                     "concept C'; filtered by the training assertions. ",
                     _COMMON + " Table 2."),
        dataset="mowl.datasets.builtin.zhapa2023.CatEORE1Dataset",
        protocol=_P,
        protocol_kwargs={"gci_name": "class_assertion", "closure": False,
                         "candidates": "classes"},
        metrics=_FULL, higher_is_better=_hib(_FULL), references=(ZHAPA2023,),
        notes=("Data: Zenodo record 13766937 (use_cases.tar.gz, use_cases/ore1); 15,157 "
               "test assertions as in Table 1. The test ontology also contains 2,342 "
               "SubClassOf axioms, which are not queries. The reference code filters "
               "only the training assertions (src/cat_membership.py, "
               "get_filtering_labels reads the training lattice); the paper, Section "
               "4.1, additionally states the deductive closure of the training set for "
               "C(a), which the code does not do - the code is followed. " + _NOTES)),
    Benchmark(
        key="cate-go-completion", name="CatE TBox completion (GO)",
        task=Task.AXIOM_PREDICTION,
        description=("Rank every query C ⊑ D against C ⊑ D' for every named concept "
                     "D'; filtered by the training subsumptions and their deductive "
                     "closure (the release's inferred_ancestors); a query's head is "
                     "not a candidate of itself. " + _COMMON + " Table 3."),
        dataset="mowl.datasets.builtin.zhapa2023.CatEGOSubsumptionDataset",
        protocol=_P,
        protocol_kwargs={"gci_name": "gci0", "candidates": "classes", "closure": True,
                         "queries": "dataset:cate_test_query_pairs",
                         "exclude_head": True},
        metrics=_TBOX, higher_is_better=_hib(_TBOX), references=(ZHAPA2023,),
        notes=("Queries: the release's test_cleaned.csv (11,893 of the 14,521 test "
               "SubClassOf axioms, the counts the paper's Table 1 gives for the test "
               "set; the reference run script evaluates the cleaned list, and the "
               "cleaning script is not part of the release). Model selection uses the "
               "release's valid_cleaned.csv (5,935 pairs). Training ontology and "
               "filtering closure verified byte-identical to the release's "
               "go.train.owl and inferred_ancestors.txt. " + _NOTES)),
    Benchmark(
        key="cate-foodon-completion", name="CatE TBox completion (FoodOn)",
        task=Task.AXIOM_PREDICTION,
        description=("Rank every query C ⊑ D against C ⊑ D' for every named concept "
                     "D'; filtered by the training subsumptions and their deductive "
                     "closure (the release's inferred_ancestors); a query's head is "
                     "not a candidate of itself. " + _COMMON + " Table 3."),
        dataset="mowl.datasets.builtin.zhapa2023.CatEFoodOnSubsumptionDataset",
        protocol=_P,
        protocol_kwargs={"gci_name": "gci0", "candidates": "classes", "closure": True,
                         "queries": "dataset:cate_test_query_pairs",
                         "exclude_head": True},
        metrics=_TBOX, higher_is_better=_hib(_TBOX), references=(ZHAPA2023,),
        notes=("Queries: the release's test_cleaned.csv (3,632 of the 5,957 test "
               "SubClassOf axioms; see the GO benchmark). Model selection uses the "
               "release's valid_cleaned.csv (1,888 pairs). Training ontology and "
               "filtering closure verified byte-identical to the release's "
               "foodon-merged.train.owl and inferred_ancestors.txt. " + _NOTES)),
    Benchmark(
        key="cate-yeast-ppi", name="CatE PPI prediction (yeast)",
        task=Task.LINK_PREDICTION,
        description=("For every test axiom P1 ⊑ ∃interacts.P2, rank P1 among the yeast "
                     "proteins with P2 fixed and P2 with P1 fixed, and average over "
                     "the two directions; filtered by the training and validation "
                     "interactions. Ties by position, Hits@k with rank ≤ k, the "
                     "trapezoid AUC (src/evaluators.py, PPIEvaluator). Table 4 "
                     "(left: PPI axioms)."),
        dataset="mowl.datasets.builtin.ppi_yeast.PPIYeastDataset",
        protocol=_PPPI,
        protocol_kwargs={"relation": "http://interacts_with"},
        metrics=_FULL, higher_is_better=_hib(_FULL), references=(ZHAPA2023,),
        notes=("CatE uses 'the PPI dataset provided in [mOWL]'; verified: the Zenodo "
               "release's ppi/test.owl is byte-identical to PPIYeastDataset's test.owl "
               "(12,040 interactions). The published numbers come from the mOWL-style "
               "PPI evaluation of src/evaluators.py (both directions, rank ≤ k, "
               "trapezoid AUC), which the released run script calls; the one-direction "
               "variant inside src/cat_ppi.py was not used. The right half of Table 4 "
               "(GO subsumptions in the ELK closure) is not part of this benchmark. ",
               _NOTES)),
]
