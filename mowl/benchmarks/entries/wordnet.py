"""The WordNet noun hierarchy benchmark of the hierarchy embedding literature
(Ganea et al. 2018, Section 5; also Suzuki et al. 2019, Dasgupta et al. 2020 and
Yu et al. 2024)."""

from mowl.benchmarks.spec import Benchmark, Task
from mowl.benchmarks.entries._refs import DASGUPTA2020, GANEA2018, SUZUKI2019, YU2023SHADOW

_DESC = (
    "The hypernymy hierarchy of the WordNet nouns (82,114 synsets; 743,086 edges in the "
    "transitive closure). Training data: the 84,363 basic edges (transitive reduction) plus "
    "{percent}% of the non-basic edges. Validation and test: 5% of the non-basic edges each "
    "(28,838), every positive with 10 fixed negatives from the release. A score threshold is "
    "chosen for the best F1 on the validation pairs and the F1 of that threshold on the test "
    "pairs is reported (Ganea et al. 2018, Section 5, Table 1).")
_NOTES = (
    "The papers report F1 in % at embedding dimensions 5 and 10; the dimension is a model "
    "setting (--param embed_dim=5) and mOWL reports F1 as a fraction. Splits and negatives "
    "are the files of the authors' code (dalab/hyperbolic_cones, data/maxn), read at a fixed "
    "commit; the threshold search follows its eval.py. The test edges are held out from the "
    "training data but are entailed by the full hierarchy; with 0% non-basic edges every "
    "test edge is in the transitive closure of the training edges, so that setting measures "
    "whether the embedding is transitive, not prediction. Published numbers come from "
    "several implementations with different optimisers (Riemannian SGD with burn-in and "
    "pretrained Poincaré embeddings for the cones); mOWL trains full-batch with its "
    "generic loop, see the recipes.")

BENCHMARKS = [
    Benchmark(
        key=f"wordnet-noun-closure-{percent}",
        name=f"WordNet noun hierarchy, {percent}% non-basic edges in training (F1)",
        task=Task.DEDUCTIVE_INFERENCE,
        evaluation_regime="hierarchy-threshold-f1",
        completion_tasks=("subsumption-prediction",),
        description=_DESC.format(percent=percent),
        dataset="mowl.datasets.builtin.wordnet_closure.WordNetNounClosureDataset",
        dataset_kwargs={"percent": percent},
        protocol="mowl.benchmarks.protocols.bindings.EdgeClassificationProtocol",
        protocol_kwargs={"negatives": "given"},
        metrics=("f1", "precision", "recall"),
        higher_is_better={"f1": True, "precision": True, "recall": True},
        references=(GANEA2018, SUZUKI2019, DASGUPTA2020, YU2023SHADOW), notes=_NOTES)
    for percent in (0, 10, 25, 50, 90)
]
