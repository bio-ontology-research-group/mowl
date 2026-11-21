"""Benchmarks of RegD (Yang and Chen, arXiv:2501.17518v3, Section 5.2-5.3, Tables 3-4)."""

from mowl.benchmarks.spec import Benchmark, Task
from mowl.benchmarks.entries._refs import YANG2025_REGD, JACKERMEIER2024

_P = "mowl.benchmarks.protocols.bindings.RegDProtocol"
_NOTES = (
    "RegD reports F1 (%) at embedding dimensions d = 5 and d = 10; the dimension is a "
    "model setting (--param embed_dim=5). mOWL reports F1 as a fraction (published "
    "values are stored as fractions too). The evaluation follows the authors' code "
    "(HuiYang1997/RegD, EXP_ontology/evaluate.py), not the paper's description of it: "
    "the score is relu of the boundary dissimilarity alone, ALL test subsumptions are "
    "evaluated (the paper says 1,000 sampled), negatives are 10 random classes per "
    "query replacing the superclass of C ⊑ D and the subclass of A ⊑ ∃r.B, and the "
    "threshold is the best-F1 threshold of the first 1,000 validation subsumptions, "
    "applied to the test split. For the prediction task the code reads the positive score "
    "of a query A ⊑ ∃r.B from score-matrix column 1, which holds the role index: it scores "
    "the class whose index in the reference classes.json equals the role's index in "
    "relations.json, not A (for GO, part_of -> GO:0008150 biological_process, 45% of the "
    "test queries). The published Table 4 numbers carry this slip; the prediction "
    "benchmarks reproduce it from the reference index files (positive_scores='reference', "
    "reference_ontology) and also report the F1 of the axiom's own score (protocol_info "
    "f1_pair).")
_INFERENCE_DESC = (
    "Train on the full ontology (Box2EL inference data). All inferred subsumptions "
    "C ⊑ D of the test set are queries; for each, the superclass D is replaced by every "
    "class as a candidate and scored by relu of the boundary dissimilarity of C's "
    "region inside the candidate; 10 random candidates per query are negatives; a "
    "query is classified true iff its score is at most the threshold that maximises F1 "
    "on validation queries sampled the same way. F1 (Section 5.2, Table 3).")
_PREDICTION_DESC = (
    "Box2EL 80/10/10 split; queries A ⊑ ∃r.B (NF3, written ∃r.B ≺ ?A in the paper, "
    "which reads C ⊑ D as D ≺ C). The subclass A is replaced by every class as a "
    "candidate and scored by relu of the boundary dissimilarity of the filler's region "
    "translated by -r inside the candidate (the code trains this task with the "
    "reversed inclusion, flip_gci2); 10 random candidates per query are negatives; "
    "the threshold maximises F1 on validation queries. As in the code, the F1's "
    "positive score is the score of the class that the role's reference index names "
    "(see the benchmark notes). F1 (Section 5.3, Table 4).")

_ONTS = (("galen", "GALEN", "GALENJackermeier2024Dataset"),
         ("go", "GO", "GOJackermeier2024Dataset"),
         ("anatomy", "ANATOMY", "AnatomyJackermeier2024Dataset"))

BENCHMARKS = [
    Benchmark(
        key=f"regd-{name}-inference", name=f"RegD inference F1 ({label})",
        task=Task.DEDUCTIVE_INFERENCE,
        evaluation_regime="regd-sampled-f1",
        description=_INFERENCE_DESC,
        dataset="mowl.datasets.builtin.jackermeier2024_reference."
                "Jackermeier2024InferencesDataset",
        dataset_kwargs={"ontology": label},
        protocol=_P, protocol_kwargs={"gci_name": "gci0"},
        metrics=("f1",), higher_is_better={"f1": True},
        references=(YANG2025_REGD, JACKERMEIER2024), notes=_NOTES)
    for name, label, _ in _ONTS
] + [
    Benchmark(
        key=f"regd-{name}-prediction", name=f"RegD prediction F1 ({label})",
        task=Task.AXIOM_PREDICTION,
        evaluation_regime="regd-sampled-f1",
        description=_PREDICTION_DESC,
        dataset=f"mowl.datasets.builtin.jackermeier2024.{cls}",
        protocol=_P, protocol_kwargs={"gci_name": "gci2", "reference_ontology": label},
        metrics=("f1",), higher_is_better={"f1": True},
        references=(YANG2025_REGD, JACKERMEIER2024), notes=_NOTES)
    for name, label, cls in _ONTS
]
