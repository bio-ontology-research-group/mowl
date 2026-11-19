"""Benchmarks of DELE (Mashkova, Zhapa-Camacho, Hoehndorf; Tables 3-6)."""

from mowl.benchmarks.spec import Benchmark, Task
from mowl.benchmarks.entries._refs import MASHKOVA2024

_P = "mowl.benchmarks.protocols.bindings.DELEProtocol"
_DS = "mowl.datasets.builtin.mashkova2024.DELEDataset"

_RAW = ("hits@10", "hits@100", "macro_mr", "micro_mr", "macro_auc", "micro_auc")
_FILTERED = ("hits@10", "f_hits@10", "hits@100", "f_hits@100", "macro_mr", "f_macro_mr",
             "micro_mr", "f_micro_mr", "macro_auc", "f_macro_auc", "micro_auc",
             "f_micro_auc")


def _hib(metrics):
    return {m: not m.endswith("mr") for m in metrics}


_COMMON = (
    "Every test axiom is a query; its tail D is ranked among the candidate classes "
    "(scores: the model's loss for the axiom type, ties averaged). macro_mr/macro_auc "
    "are computed over test axioms (AUC as mOWL's compute_rank_roc), micro_mr/micro_auc "
    "average the metric per head class (Eqs. 19-22). Evaluated by "
    "mowl.evaluation.deductive.DELEEvaluation.")
_FILTER = (
    " Filtered metrics (prefix f_) move the training axioms and the deductive closure of "
    "training and test axioms (other than the true answer) to the end of the ranking.")
_CLOSURE_NOTE = (
    "The deductive closure is DELE's (mowl.training.DeductiveClosure, see the "
    "'dele-evaluation' regime card for its completeness). DELE trains with "
    "ReduceLROnPlateau, early stopping (patience 20) and "
    "selects hyperparameters per model (Appendix D); the 'l' and 'l+n' variants are the "
    "DELE training regimes (all negative losses; negatives filtered by the closure).")

BENCHMARKS = [
    Benchmark(
        key="dele-yeast-ppi", name="DELE PPI prediction (yeast)",
        task=Task.LINK_PREDICTION,
        description=("Predict {P1} ⊑ ∃interacts_with.{P2}; candidates: yeast proteins. "
                     f"{_COMMON} Only non-filtered metrics are published (Table 4)."),
        dataset="mowl.datasets.builtin.ppi_yeast.PPIYeastDataset",
        protocol=_P,
        protocol_kwargs={"gci_name": "gci2", "relation": "http://interacts_with",
                         "filter_with": None, "remove_entailed_test": False},
        metrics=_RAW, higher_is_better=_hib(_RAW), references=(MASHKOVA2024,),
        notes=("DELE uses mOWL's PPIYeastDataset (Section 6.1.1). Verified: the test and "
               "validation files of PPIYeastDataset are byte-identical to "
               "data/go_string/{test,valid}_yeast.owl of the DELE release; the training "
               "files differ in size (DELE 93 MB, mOWL 198 MB) and were not compared "
               "axiom by axiom. " + _CLOSURE_NOTE)),
    Benchmark(
        key="dele-human-ppi", name="DELE PPI prediction (human)",
        task=Task.LINK_PREDICTION,
        description="The DELE PPI protocol applied to human proteins. " + _COMMON,
        dataset="mowl.datasets.builtin.ppi_yeast.PPIHumanDataset",
        protocol=_P,
        protocol_kwargs={"gci_name": "gci2", "relation": "http://interacts_with",
                         "filter_with": None, "remove_entailed_test": False},
        metrics=_RAW, higher_is_better=_hib(_RAW), references=(MASHKOVA2024,),
        notes=("The DELE paper reports yeast only; no published numbers exist for human. "
               f"{_CLOSURE_NOTE}")),
    Benchmark(
        key="dele-yeast-function", name="DELE protein function prediction (yeast)",
        task=Task.MEMBERSHIP_PREDICTION,
        description=("Predict {P} ⊑ ∃has_function.{GO}; heads: yeast proteins, candidates: "
                     "GO classes. Entailed axioms were removed from the test set by the "
                     "authors. " + _COMMON + _FILTER + " Table 3."),
        dataset=_DS, dataset_kwargs={"task": "yeast_function"},
        protocol=_P,
        protocol_kwargs={"gci_name": "gci2", "relation": "http://has_function"},
        metrics=_FILTERED, higher_is_better=_hib(_FILTERED), references=(MASHKOVA2024,),
        notes=("Data: DELE release, data/go_string/*_yeast_hf.owl (commit 5974c7d). "
               f"{_CLOSURE_NOTE}")),
    Benchmark(
        key="dele-human-function", name="DELE protein function prediction (human)",
        task=Task.MEMBERSHIP_PREDICTION,
        description=("The DELE function prediction protocol for human proteins. "
                     f"{_COMMON}{_FILTER}"),
        dataset=_DS, dataset_kwargs={"task": "human_function"},
        protocol=_P,
        protocol_kwargs={"gci_name": "gci2", "relation": "http://has_function"},
        metrics=_FILTERED, higher_is_better=_hib(_FILTERED), references=(MASHKOVA2024,),
        notes=("No data: neither the DELE paper nor its release has a human function "
               "dataset. Loading this benchmark raises an error unless the dataset is "
               "given explicit files (run with --dataset-param paths=...). No published "
               "numbers. " + _CLOSURE_NOTE)),
    Benchmark(
        key="dele-foodon-subsumption", name="DELE subsumption prediction (FoodOn)",
        task=Task.AXIOM_PREDICTION,
        description=("Predict atomic subsumptions A ⊑ B; candidates: all classes. "
                     f"{_COMMON}{_FILTER} Table 5."),
        dataset=_DS, dataset_kwargs={"task": "foodon"},
        protocol=_P, protocol_kwargs={"gci_name": "gci0"},
        metrics=_FILTERED, higher_is_better=_hib(_FILTERED), references=(MASHKOVA2024,),
        notes=("Data: DELE release, data/food_ontology (commit 5974c7d). Verified to differ "
               "from mOWL's FoodOnSubsumptionDataset (5,752 vs 5,957 test axioms, because "
               "DELE removes entailed and normalisation-overlapping axioms), so the "
               "release is used. " + _CLOSURE_NOTE)),
    Benchmark(
        key="dele-galen-subsumption", name="DELE subsumption prediction (GALEN)",
        task=Task.AXIOM_PREDICTION,
        description=("Predict atomic subsumptions A ⊑ B on a random 90:5:5 split of GALEN; "
                     "candidates: all classes. " + _COMMON + _FILTER + " Table 6."),
        dataset=_DS, dataset_kwargs={"task": "galen"},
        protocol=_P, protocol_kwargs={"gci_name": "gci0"},
        metrics=_FILTERED, higher_is_better=_hib(_FILTERED), references=(MASHKOVA2024,),
        notes=("Data: DELE release, data/galen_ontology (667 test axioms, matching "
               "Appendix C). Not the Box2EL GALEN split. " + _CLOSURE_NOTE)),
]
