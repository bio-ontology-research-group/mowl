"""The masked ABox revision benchmarks of the DF-ALC paper (Wu et al. 2022)."""

from mowl.benchmarks.spec import Benchmark, Task
from mowl.benchmarks.entries._refs import WU2022

MASK_RATES = (0.2, 0.4, 0.6, 0.8)
LABELS = {"family": "Family", "family2": "Family2", "glycordf": "GlycoRDF",
          "nifdys": "Nifdys", "nihss": "Nihss", "ontodm": "Ontodm", "sso": "Sso"}

_DESC = (
    "Masked ABox revision on {label} (Wu et al. 2022, Performance Evaluation, Table 3): the "
    "ABox completed by a reasoner is the ideal grounding (1 if entailed, 0.5 otherwise); "
    "{percent}% of it is replaced by random truth values in (0.2, 0.8); the model starts from "
    "this masked grounding and is trained on the TBox alone; the revised grounding is made "
    "crisp (above 0.5 true, otherwise false, as in the authors' evaluator) and the success "
    "rate is the fraction of TBox axioms it satisfies over the domain of the individuals.")
_NOTES = (
    "The ontologies are the files of the released code (a mirror of an anonymous "
    "repository). The paper's masked grounding (column M) is not a method of mOWL and is "
    "reported by the regime as info['masked_success_rate']; Logical Tensor Networks are not "
    "implemented. The success rate is over the subclass, equivalence and disjointness axioms "
    "of the original ontology (ALC over named entities); other axioms (domain, range, role "
    "axioms) are not counted, so the denominators differ from the paper's. The paper's "
    "conjunctive query answering and the PASCAL-Part semantic image interpretation "
    "experiments are not reproduced: the former needs the query generator (a Java tool of "
    "the code), the latter the Fast R-CNN detections and PASCAL-Part tables, which the "
    "authors' README takes from the code of Donadello et al. (2017) and which were not "
    "obtained.")

BENCHMARKS = [
    Benchmark(
        key=f"dfalc-revision-{name}-mask{int(rate * 100)}",
        name=f"Masked ABox revision, {label}, mask rate {int(rate * 100)}%",
        task=Task.EXACT_MODEL,
        evaluation_regime="masked-abox-revision",
        description=_DESC.format(label=label, percent=int(rate * 100)),
        dataset="mowl.datasets.builtin.dfalc_revision.DFALCRevisionDataset",
        dataset_kwargs={"name": name},
        protocol="mowl.benchmarks.protocols.bindings.MaskedABoxRevisionProtocol",
        protocol_kwargs={"mask_rate": rate, "alpha": 0.8},
        metrics=("success_rate",), higher_is_better={"success_rate": True},
        references=(WU2022,), notes=_NOTES)
    for name, label in LABELS.items() for rate in MASK_RATES
]
