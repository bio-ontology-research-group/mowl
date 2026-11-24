"""Faithfulness benchmarks of Mashkova & Hoehndorf (2026): Pizza and the GO-plus slice."""

from mowl.benchmarks.spec import Benchmark, Task
from mowl.catalog.card import Reference

MASHKOVA2026 = Reference(
    "mashkova2026",
    "Mashkova, O., Hoehndorf, R. (2026). A hierarchy of faithfulness criteria for "
    "knowledge base completion. arXiv:2609.27863.",
    "https://arxiv.org/abs/2609.27863")

NORMAL_FORMS = ("gci0", "gci1", "gci2", "gci3", "gci1_bot")
METRICS = ("auc_pm", "adm@10", "fvr", "pair_fvr", "rho_p", "mrr", "hits@10")
HIGHER_IS_BETTER = {"auc_pm": True, "adm@10": True, "fvr": False, "pair_fvr": False,
                    "rho_p": True, "mrr": True, "hits@10": True}

_DESCRIPTION = (
    "Faithfulness hierarchy (discrimination, admissibility, monotonic and probabilistic "
    "logical faithfulness) on reasoner-generated test sets of entailed (T+), "
    "contradictory (T-) and undetermined (T?) axioms, per EL normal form. Each normal "
    "form has its own split: train on train.owl of the split, score T+, T- and T? with "
    "the model's axiom scores. AUC+/- separates T+ from T- (L0); Adm@10 is the fraction "
    "of the 10 best completions of each T+ query that are not in T- (L1); FVR is the "
    "rate of triples in T- x T? x T+ violating f(a-) < f(a?) < f(a+) and Pair-FVR the "
    "rate of pairs with f(a?) <= f(a-) (L2); rho_P is the Spearman correlation between "
    "scores and the relative model count P(a | O) (|Delta| = 2, published values) on T? "
    "(L3); MRR and Hits@10 rank the superclass slot of T+ over all classes (unfiltered, "
    "ties averaged).")

_NOTES = (
    "One run evaluates one normal form, selected with dataset_kwargs['normal_form'] "
    "(default gci0); subsets lists all five. Data: bio-ontology-research-group/kbc at a "
    "pinned commit (mowl.datasets.builtin.faithfulness), kept as released. The paper's "
    "Table 2 counts are smaller than the released files for some splits (e.g. Pizza "
    "GCI0: 166 entailed / 936 contradictory axioms in the files, 97 / 914 in Table 2; 22 "
    "of the 936 are C SubClassOf owl:Thing), presumably because the reference evaluator "
    "drops axioms whose classes have no embedding; mOWL likewise drops axioms the model "
    "cannot score. Published numbers are means over seeds (standard deviations in the "
    "notes of each result).")

BENCHMARKS = [
    Benchmark(
        key="faithfulness-pizza",
        name="Faithfulness hierarchy on the Pizza ontology",
        task=Task.FAITHFULNESS,
        description=_DESCRIPTION,
        dataset="mowl.datasets.builtin.faithfulness.PizzaFaithfulnessDataset",
        dataset_kwargs={"normal_form": "gci0"},
        protocol="mowl.benchmarks.protocols.faithfulness.FaithfulnessProtocol",
        metrics=METRICS,
        higher_is_better=HIGHER_IS_BETTER,
        subsets=NORMAL_FORMS,
        references=(MASHKOVA2026,),
        notes=_NOTES + " All five normal forms have a contradictory set."),
    Benchmark(
        key="faithfulness-goplus",
        name="Faithfulness hierarchy on the GO-plus slice",
        task=Task.FAITHFULNESS,
        description=_DESCRIPTION,
        dataset="mowl.datasets.builtin.faithfulness.GOPlusFaithfulnessDataset",
        dataset_kwargs={"normal_form": "gci0"},
        protocol="mowl.benchmarks.protocols.faithfulness.FaithfulnessProtocol",
        metrics=METRICS,
        higher_is_better=HIGHER_IS_BETTER,
        subsets=NORMAL_FORMS,
        references=(MASHKOVA2026,),
        notes=_NOTES + " Contradictory sets exist only for gci0 and gci1_bot; for gci1, "
        "gci2 and gci3 only rho_p, mrr and hits@10 are defined (the others are None)."),
]
