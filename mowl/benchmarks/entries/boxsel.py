"""The Statistical EL benchmarks of the BoxSEL paper (Zhu et al. 2024)."""

from mowl.benchmarks.spec import Benchmark, Task
from mowl.benchmarks.entries._refs import ZHU2024

_METRICS = ("soundness_accuracy", "soundness_error", "approximation_gap",
            "mean_absolute_error")

_DESC = (
    "Approximate probabilistic inference on the {label} Statistical EL ontology (Zhu et al. "
    "2024, Experiments): conditionals (B|A), (B|A1 ⊓ A2), (B|∃r.A), (∃r.A|B) with the "
    "statistical proportions of YAGO3 as probabilities; 30% of the conditionals of the most "
    "general concept are held out as queries; the entailed interval of a query is given by "
    "the probabilistic modus ponens from two premises in the learning set; the estimate is "
    "the volume ratio of the embedding. Soundness accuracy, soundness error and approximation "
    "gap.")
_NOTES = (
    "NOT RUNNABLE WITHOUT DATA: the ontologies of the paper are not released (no code or data "
    "link) and only partly specified (Hybrid: the 18 most frequent roles and the 12 most "
    "frequent concepts in their domains and ranges; Person and Country: the concepts and "
    "roles below wordnet_person_100007846 / wordnet_country_108544813 of YAGO3, the roles "
    "are not named). The dataset reads a file built with "
    "mowl.datasets.builtin.sel_conditionals; the benchmark then measures the same task on "
    "that ontology and the published numbers are not comparable. The published estimate "
    "uses N = 60 embeddings (min and max of their probabilities); a single run gives the "
    "point estimate, use SELProbabilityRegime.evaluate_ensemble for the interval. The "
    "queries evaluated by mOWL are those with premises of the probabilistic modus ponens "
    "(Proposition 1); the counts of the paper's query table are not reproduced.")

BENCHMARKS = [
    Benchmark(
        key=f"boxsel-sel-{name}",
        name=f"Statistical EL {label}: approximate probabilistic inference",
        task=Task.DEDUCTIVE_INFERENCE,
        evaluation_regime="sel-pmp-intervals",
        completion_tasks=(), extra_inference_tasks=("subsumption",),
        description=_DESC.format(label=label),
        dataset="mowl.datasets.builtin.sel_conditionals.SELConditionalDataset",
        dataset_kwargs={"name": name},
        protocol="mowl.benchmarks.protocols.bindings.SELProbabilityProtocol",
        metrics=_METRICS,
        higher_is_better={"soundness_accuracy": True, "soundness_error": False,
                          "approximation_gap": False, "mean_absolute_error": False},
        references=(ZHU2024,), notes=_NOTES)
    for name, label in (("person", "Person"), ("country", "Country"), ("hybrid", "Hybrid"))
]
