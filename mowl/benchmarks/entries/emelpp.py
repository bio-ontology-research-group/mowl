"""Benchmarks of EmEL++ (Mondal, Bhatia, Mutharaju, AAAI-MAKE 2021, Table 3)."""

from mowl.benchmarks.spec import Benchmark, Task
from mowl.benchmarks.entries._refs import MONDAL2021

_METRICS = ("hits@1", "hits@10", "hits@100", "auc", "med", "p90")

BENCHMARKS = [
    Benchmark(
        key=f"emelpp-{name}-prediction", name=f"EmEL++ subsumption prediction ({label})",
        task=Task.AXIOM_PREDICTION,
        description=("70/20/10 split of the atomic subclass pairs of the normalised "
                     "ontology. For every test pair C ⊑ D, all classes except D are ranked "
                     "by the distance of their centre from D's centre and the rank of C is "
                     "recorded (raw). Hits@1/10/100, AUC, median and 90th percentile rank "
                     "(Sections 4.3-4.4, Table 3, 'Test Data')."),
        dataset="mowl.datasets.builtin.mondal2021.EmELppSubsumptionDataset",
        dataset_kwargs={"ontology": label},
        protocol="mowl.benchmarks.protocols.bindings.EmELppProtocol",
        metrics=_METRICS,
        higher_is_better={"hits@1": True, "hits@10": True, "hits@100": True, "auc": True,
                          "med": False, "p90": False},
        references=(MONDAL2021,),
        notes=("Data: kracr/EmELpp (commit 267f2c7). The reference training script trains "
               "on the whole normalised ontology, which still contains the test pairs "
               "(leakage noted by Jackermeier et al. 2024, footnote 5); mOWL removes the "
               "validation and test pairs from the training ontology, following the "
               "paper's text, so published numbers are not directly comparable. The "
               "reference AUC drops the first rank value before integrating; mOWL "
               "integrates over all ranks. Models without centres are scored by their "
               "gci0 loss. SNOMED CT is not provided (no normalised ontology in the "
               "release; licence)."),
    )
    for name, label in (("galen", "GALEN"), ("go", "GO"), ("anatomy", "ANATOMY"))
]
