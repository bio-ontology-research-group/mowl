"""Benchmarks of the two-view ontology embedding literature: concepts and instances.

EIKE (Wang et al. 2024) on YAGO39K and M-YAGO39K (Lv et al. 2018): triple classification of
InstanceOf, SubClassOf and relational triples against released negatives, and link prediction
on relational triples. Concept2Box (Huang et al. 2023) on the DBpedia data of JOIE (Hao et al.
2019): knowledge graph completion in the instance view and the ontology view, and concept
linking.

Not here, and why: EIKE's third dataset DB99K-242 has no released splits or negatives (the
repository of EIKE ships another DBpedia, "new-dbpedia", which is not the one of the paper);
the Recipe dataset of Concept2Box was built from Amazon.com listings and is not public.
"""

from mowl.benchmarks.spec import Benchmark, Task
from mowl.benchmarks.entries._refs import WANG2024EIKE, HUANG2023, LV2018TRANSC, HAO2019JOIE

_TC = "mowl.benchmarks.protocols.bindings.TripleClassificationProtocol"
_TC_METRICS = ("accuracy", "precision", "recall", "f1")
_YAGO = "mowl.datasets.builtin.yago39k.YAGO39KDataset"
_JOIE = "mowl.datasets.builtin.joie_dbpedia.JOIEDBpediaDataset"

_TC_NOTES = (
    "Metrics are fractions (the paper prints %). The negatives are the ones released with "
    "the data, one per positive; a threshold per relation is tuned on the validation "
    "positives and negatives (accuracy, 100 steps, as the authors' code). The paper trains "
    "all three kinds of triples together and reports each; the benchmark trains on whatever "
    "the dataset's training ontology holds, so the three benchmarks of a variant are one "
    "training run (one result file per run is enough, the evaluations differ only in the "
    "form). The paper's best settings (Appendix) differ per variant: see the recipes. mOWL "
    "has no Bernoulli corruption and no L1 norm in the relational score, so some published "
    "variants cannot be reproduced exactly.")

_FORM = {"instanceof": ("class_assertion", "instance-checking", "InstanceOf"),
         "subclassof": ("gci0", "subsumption", "SubClassOf"),
         "relational": ("object_property_assertion", "role-assertion", "relational")}


def _tc(variant, kind):
    form, task, label = _FORM[kind]
    name = "M-YAGO39K" if variant == "m-yago39k" else "YAGO39K"
    return Benchmark(
        key=f"{variant}-{kind}-classification",
        name=f"{name} {label} triple classification",
        task=Task.AXIOM_PREDICTION, evaluation_regime="triple-classification-thresholds",
        completion_tasks=(), extra_inference_tasks=(task,), target="inductive prediction",
        description=(
            f"{name} as an ontology with individuals (instances are individuals, concepts "
            f"classes, relations object properties): classify the held-out {label} triples "
            "against the negatives of the release with one cost threshold per relation, tuned "
            "on the validation triples (EIKE, Section 5.3)."),
        dataset=_YAGO, dataset_kwargs={"variant": variant}, protocol=_TC,
        protocol_kwargs={"form": form, "negatives": "given"},
        metrics=_TC_METRICS, higher_is_better={m: True for m in _TC_METRICS},
        references=(WANG2024EIKE, LV2018TRANSC), notes=_TC_NOTES)


_LP_METRICS = ("mrr", "mr", "hits@1", "hits@3", "hits@10",
               "f_mrr", "f_mr", "f_hits@1", "f_hits@3", "f_hits@10")

BENCHMARKS = [
    _tc("yago39k", "instanceof"), _tc("yago39k", "subclassof"),
    _tc("yago39k", "relational"),
    _tc("m-yago39k", "instanceof"), _tc("m-yago39k", "subclassof"),
    Benchmark(
        key="yago39k-link-prediction", name="YAGO39K link prediction",
        task=Task.LINK_PREDICTION, evaluation_regime="assertion-link-prediction",
        completion_tasks=("link-prediction",), target="inductive prediction",
        description=(
            "Link prediction on the relational triples of YAGO39K (9,364 test triples): every "
            "test assertion r(a, b) is ranked as the object of r(a, ?) and as the subject of "
            "r(?, b) among all 39,374 instances; raw ranks, and filtered ranks without the "
            "known training, validation and test triples; MRR, MR and Hits@1/3/10 over both "
            "directions (EIKE, Section 5.4; TransC)."),
        dataset=_YAGO, dataset_kwargs={"variant": "yago39k"},
        protocol="mowl.benchmarks.protocols.bindings.AssertionLinkPredictionProtocol",
        protocol_kwargs={"max_rows": 2 ** 19},
        metrics=_LP_METRICS, subsets=("all", "head", "tail"),
        higher_is_better={m: not m.endswith("mr") or m.endswith("mrr") for m in _LP_METRICS},
        references=(WANG2024EIKE, LV2018TRANSC),
        notes=("Hits@k and MRR are fractions (the paper prints Hits in %). Tied candidates "
               "share the average rank. The paper's best link prediction settings differ from "
               "its triple classification settings: see the recipes. Evaluating costs 2 x "
               "9,364 x 39,374 = 7.4e8 scored candidate axioms.")),
    Benchmark(
        key="joie-dbpedia-instance-view-completion",
        name="DBpedia (JOIE) instance-view knowledge graph completion",
        task=Task.LINK_PREDICTION, evaluation_regime="two-view-kg-completion",
        completion_tasks=("link-prediction",), target="inductive prediction",
        description=(
            "The instance view of the DBpedia data of JOIE as role assertions between "
            "entities, split at random 80% / 10% / 10% (seeded; the split of the paper is not "
            "released), all cross-view links known: rank the tail of every test triple "
            "(h, r, ?t) among all entities; MRR, Hits@1 and Hits@10, raw and filtered "
            "(Concept2Box, Section 5.2)."),
        dataset=_JOIE, dataset_kwargs={"task": "kg-completion", "seed": 0},
        protocol="mowl.benchmarks.protocols.bindings.AssertionLinkPredictionProtocol",
        protocol_kwargs={"directions": ["tail"], "hits": [1, 10], "max_rows": 2 ** 19},
        metrics=("mrr", "hits@1", "hits@10", "f_mrr", "f_hits@1", "f_hits@10"),
        higher_is_better={m: True for m in ("mrr", "hits@1", "hits@10", "f_mrr",
                                            "f_hits@1", "f_hits@10")},
        references=(HUANG2023, HAO2019JOIE),
        notes=("The paper does not say whether it filters; published numbers are compared "
               "with the raw metrics by default. The release has fewer entities, relations "
               "and triples than the statistics the paper prints (see the dataset class), so "
               "numbers are indicative only. Ranking 65,851 x 99,744 candidate axioms is "
               "6.6e9 scorings.")),
    Benchmark(
        key="joie-dbpedia-ontology-view-completion",
        name="DBpedia (JOIE) ontology-view knowledge graph completion",
        task=Task.LINK_PREDICTION, evaluation_regime="two-view-kg-completion",
        completion_tasks=("existential-prediction",), encodes=("link-prediction",),
        target="inductive prediction",
        description=(
            "The ontology view of the DBpedia data of JOIE (675 distinct triples over 174 concepts, 20 "
            "relations; every triple is C ⊑ ∃r.D), split at random 80% / 10% / 10% (seeded), "
            "all cross-view links known: rank the tail of every test triple among the "
            "concepts of the view; MRR, Hits@1 and Hits@10, raw and filtered (Concept2Box, "
            "Section 5.2)."),
        dataset=_JOIE, dataset_kwargs={"task": "kg-completion", "seed": 0},
        protocol="mowl.benchmarks.protocols.bindings.FilteredRankingProtocol",
        protocol_kwargs={"key": "two-view-kg-completion", "gci_name": "gci2", "target": "tail",
                         "candidates": "dataset:ontology_view_concepts",
                         "filter_by": ["train", "valid", "test"], "ties": "average",
                         "hits": [1, 10], "scoring": "model"},
        metrics=("mrr", "hits@1", "hits@10", "f_mrr", "f_hits@1", "f_hits@10"),
        higher_is_better={m: True for m in ("mrr", "hits@1", "hits@10", "f_mrr",
                                            "f_hits@1", "f_hits@10")},
        references=(HUANG2023, HAO2019JOIE),
        notes=("About 67 test triples: results are noisy. The relation isa is treated like "
               "the other 19 relations, as in the paper. Filtering is not stated in the "
               "paper; published numbers are compared with the raw metrics by default.")),
    Benchmark(
        key="joie-dbpedia-concept-linking",
        name="DBpedia (JOIE) concept linking",
        task=Task.MEMBERSHIP_PREDICTION, evaluation_regime="concept-linking-ranking",
        completion_tasks=("type-prediction",), target="inductive prediction",
        description=(
            "The cross-view links of the DBpedia data of JOIE as class assertions C(a), split "
            "at random 80% / 10% / 10% (seeded), all triples of both views known: rank all "
            "242 concepts for every test entity; MRR, Hits@1 (accuracy) and Hits@3, raw and "
            "filtered (Concept2Box, Section 5.3)."),
        dataset=_JOIE, dataset_kwargs={"task": "concept-linking", "seed": 0},
        protocol="mowl.benchmarks.protocols.bindings.FilteredRankingProtocol",
        protocol_kwargs={"key": "concept-linking-ranking", "gci_name": "class_assertion",
                         "target": "tail",
                         "candidates": "classes", "filter_by": ["train", "valid", "test"],
                         "ties": "average", "hits": [1, 3], "scoring": "model"},
        metrics=("mrr", "hits@1", "hits@3", "f_mrr", "f_hits@1", "f_hits@3"),
        higher_is_better={m: True for m in ("mrr", "hits@1", "hits@3", "f_mrr",
                                            "f_hits@1", "f_hits@3")},
        references=(HUANG2023, HAO2019JOIE),
        notes=("Filtering is not stated in the paper; published numbers are compared with "
               "the raw metrics by default. The release types every entity with exactly one "
               "concept in most cases, so raw and filtered metrics are close.")),
]
