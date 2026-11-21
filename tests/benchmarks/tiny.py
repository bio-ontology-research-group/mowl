"""A tiny EL++ dataset with training, validation and test ontologies covering the four
normal forms, written in OWL functional syntax (no downloads)."""

import os
import tempfile

NS = "http://mowl.borg/bench#"


def _c(name):
    return f"<{NS}{name}>"


TRAIN = [
    "SubClassOf({A} {B})", "SubClassOf({B} {C})", "SubClassOf({D} {C})",
    "SubClassOf({E} {F})", "SubClassOf({G} {F})", "SubClassOf({H} {F})",
    "SubClassOf(ObjectIntersectionOf({A} {D}) {E})",
    "SubClassOf(ObjectIntersectionOf({B} {G}) {H})",
    "SubClassOf({A} ObjectSomeValuesFrom({r} {E}))",
    "SubClassOf({B} ObjectSomeValuesFrom({r} {G}))",
    "SubClassOf({D} ObjectSomeValuesFrom({r} {H}))",
    "SubClassOf(ObjectSomeValuesFrom({r} {E}) {F})",
    "SubClassOf(ObjectSomeValuesFrom({r} {G}) {C})",
    "SubClassOf({P1} ObjectSomeValuesFrom({r} {P2}))",
    "SubClassOf({P2} ObjectSomeValuesFrom({r} {P1}))",
    "SubClassOf({P2} ObjectSomeValuesFrom({r} {P3}))",
]
VALID = ["SubClassOf({D} {F})", "SubClassOf({G} ObjectSomeValuesFrom({r} {E}))"]
TEST = [
    "SubClassOf({A} {C})", "SubClassOf({E} {C})",
    "SubClassOf(ObjectIntersectionOf({D} {G}) {F})",
    "SubClassOf({E} ObjectSomeValuesFrom({r} {G}))",
    "SubClassOf({P3} ObjectSomeValuesFrom({r} {P1}))",
    "SubClassOf(ObjectSomeValuesFrom({r} {H}) {F})",
]


def _fill(line):
    names = {n: _c(n) for n in ("A", "B", "C", "D", "E", "F", "G", "H", "P1", "P2", "P3")}
    names["r"] = _c("r")
    return line.format(**names)


def _write(path, iri, lines):
    with open(path, "w") as f:
        f.write(f"Prefix(owl:=<http://www.w3.org/2002/07/owl#>)\nOntology(<{iri}>\n")
        for line in lines:
            f.write(_fill(line) + "\n")
        f.write(")\n")


def scratch_dir():
    """A fresh directory under the system temporary directory (honours $TMPDIR)."""
    base = tempfile.gettempdir()
    os.makedirs(base, exist_ok=True)
    return tempfile.mkdtemp(prefix="mowl_bench_test_", dir=base)


def write_tiny(directory=None):
    """Write the three ontologies and return their paths."""
    directory = directory or scratch_dir()
    paths = [os.path.join(directory, n) for n in ("train.owl", "valid.owl", "test.owl")]
    for path, lines, tag in zip(paths, (TRAIN, VALID, TEST), ("train", "valid", "test")):
        _write(path, f"http://mowl.borg/bench/{tag}", lines)
    return paths


def tiny_dataset(directory=None):
    """A :class:`mowl.datasets.PathDataset` over the tiny ontologies (needs the JVM)."""
    from mowl.datasets import PathDataset
    from mowl.datasets.base import OWLClasses

    class TinyDataset(PathDataset):
        @property
        def evaluation_classes(self):
            prots = [c for n, c in self.classes.as_dict.items() if "#P" in n]
            return OWLClasses(prots), OWLClasses(prots)

    return TinyDataset(*write_tiny(directory))
