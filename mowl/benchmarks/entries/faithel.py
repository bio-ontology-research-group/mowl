"""The family knowledge base of the FaithEL code (Lacerda, Ozaki and Guimarães, 2024)."""

from mowl.benchmarks.spec import Benchmark, Task
from mowl.benchmarks.entries._refs import LACERDA2024FAITHEL

_METRICS = ("hits@1", "hits@3", "hits@10", "hits@100", "mrr")

BENCHMARKS = [
    Benchmark(
        key="faithel-family-assertions",
        name="FaithEL family knowledge base: concept and role assertion queries",
        task=Task.MEMBERSHIP_PREDICTION,
        evaluation_regime="assertion-query-hits",
        completion_tasks=("type-prediction", "link-prediction"),
        description=(
            "The family knowledge base of the authors' code (469 individuals, 10 classes, 7 "
            "roles, from Wikidata). The entailed class assertions (HermiT) and the entailed "
            "role assertions are split 70/30 at random; the TBox and the training "
            "assertions are the training ontology. Every class of a test class assertion "
            "is a query C(?) and every (a, r) of a test role assertion a query r(a, ?); "
            "all individuals are ranked, and a query is a hit at k if any of its test "
            "answers is among the k best (hits@1, 3, 10, 100, as in the code)."),
        dataset="mowl.datasets.builtin.faithel_family.FaithELFamilyDataset",
        protocol="mowl.benchmarks.protocols.bindings.AssertionQueryProtocol",
        metrics=_METRICS, subsets=("concept", "role"),
        higher_is_better={m: True for m in _METRICS},
        references=(LACERDA2024FAITHEL,),
        notes=(
            "No published numbers are recorded: the RuleML+RR paper is behind a login and "
            "could not be read, and the repository (no licence) contains only a notebook "
            "that plots hits@k per epoch without stored output. The split is mOWL's, not "
            "PyTorch's random_split of the code; the code reads the class assertions as "
            "the indirect named-class types after classification with HermiT (owlready2). "
            "Which numbers the paper reports (and for which split) is unknown.")),
]
