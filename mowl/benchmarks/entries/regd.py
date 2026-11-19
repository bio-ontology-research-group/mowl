"""Benchmarks of RegD (Yang and Chen, arXiv:2501.17518v3, Section 5.2-5.3, Tables 3-4)."""

from mowl.benchmarks.spec import Benchmark, Task
from mowl.benchmarks.entries._refs import YANG2025_REGD, JACKERMEIER2024

_P = "mowl.benchmarks.protocols.bindings.RegDProtocol"
_NOTES = (
    "RegD reports F1 (%) at embedding dimensions d = 5 and d = 10; the dimension is a model "
    "setting (--param embed_dim=5). mOWL reports F1 as a fraction (published values are "
    "stored as fractions too). Negatives follow the reference code "
    "(HuiYang1997/RegD, EXP_ontology/evaluate.py): for C ⊑ D the superclass D is replaced "
    "(the paper's text says 'the child', i.e. C, since it reads C ⊑ D as D ≺ C), for "
    "A ⊑ ∃r.B the subclass A; they are not checked against the ontology. The paper says "
    "the threshold is the one that 'achieved the best performance on the evaluation set'; "
    "mOWL tunes it on validation pairs sampled like the test pairs (the reference code "
    "either uses a fixed threshold or tunes it on the evaluated pairs). Scores: the "
    "model's loss for the axiom type (RegD's own models use their energy).")

_ONTS = (("galen", "GALEN", "GALENJackermeier2024Dataset"),
         ("go", "GO", "GOJackermeier2024Dataset"),
         ("anatomy", "ANATOMY", "AnatomyJackermeier2024Dataset"))

BENCHMARKS = [
    Benchmark(
        key=f"regd-{name}-inference", name=f"RegD inference F1 ({label})",
        task=Task.DEDUCTIVE_INFERENCE,
        description=("Train on the full ontology (Box2EL inference data). Sample 1,000 "
                     "inferred subsumptions u ⊑ v from the test set, generate 10 negatives "
                     "per positive by replacing the superclass with a random class, classify "
                     "a pair as true iff its score is at most the threshold that maximises "
                     "F1 on validation pairs sampled the same way, and report F1 "
                     "(Section 5.2, Table 3)."),
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
        description=("Box2EL 80/10/10 split; queries A ⊑ ∃r.B (NF3, written ∃r.B ≺ ?A in "
                     "the paper, which reads C ⊑ D as D ≺ C). Sample 1,000 test axioms, "
                     "10 random negatives each (A replaced), threshold chosen on "
                     "validation, F1 (Section 5.3, Table 4)."),
        dataset=f"mowl.datasets.builtin.jackermeier2024.{cls}",
        protocol=_P, protocol_kwargs={"gci_name": "gci2"},
        metrics=("f1",), higher_is_better={"f1": True},
        references=(YANG2025_REGD, JACKERMEIER2024), notes=_NOTES)
    for name, label, cls in _ONTS
]
