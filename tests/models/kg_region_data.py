"""A tiny ontology with a TBox and an ABox, for the region-based knowledge graph models."""
import os
import tempfile

NS = "http://mowl.borg/kgr#"

AXIOMS = [
    "SubClassOf(<{ns}A> <{ns}B>)", "SubClassOf(<{ns}B> <{ns}C>)",
    "SubClassOf(<{ns}D> <{ns}C>)", "SubClassOf(<{ns}E> <{ns}D>)",
    "SubClassOf(<{ns}A> ObjectSomeValuesFrom(<{ns}r> <{ns}D>))",
    "SubClassOf(<{ns}B> ObjectSomeValuesFrom(<{ns}r> <{ns}E>))",
    "ClassAssertion(<{ns}A> <{ns}a1>)", "ClassAssertion(<{ns}A> <{ns}a2>)",
    "ClassAssertion(<{ns}D> <{ns}d1>)", "ClassAssertion(<{ns}E> <{ns}e1>)",
    "ObjectPropertyAssertion(<{ns}r> <{ns}a1> <{ns}d1>)",
    "ObjectPropertyAssertion(<{ns}r> <{ns}a2> <{ns}e1>)",
    "ObjectPropertyAssertion(<{ns}s> <{ns}d1> <{ns}e1>)",
    "ObjectPropertyAssertion(<{ns}s> <{ns}e1> <{ns}a1>)",
]


def abox_dataset():
    """A :class:`mowl.datasets.PathDataset` over the ontology above (needs the JVM)."""
    from mowl.datasets import PathDataset
    base = tempfile.gettempdir()
    os.makedirs(base, exist_ok=True)
    path = os.path.join(tempfile.mkdtemp(prefix="mowl_kgr_", dir=base), "onto.owl")
    with open(path, "w") as f:
        f.write("Prefix(owl:=<http://www.w3.org/2002/07/owl#>)\n"
                "Ontology(<http://mowl.borg/kgr>\n")
        for axiom in AXIOMS:
            f.write(axiom.format(ns=NS) + "\n")
        f.write(")\n")
    return PathDataset(path)
