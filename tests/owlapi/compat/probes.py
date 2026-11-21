"""Behaviour probes for the Java side of mOWL.

Every probe takes the path of an ontology and returns a JSON value that describes one
piece of mOWL's behaviour on it in a way that does not depend on the OWL API version:
IRIs and counts, never ``toString()`` of OWL API objects. The values recorded with the
OWL API version that mOWL shipped before an upgrade are the *golden* files in
``golden/``; ``tests/owlapi/test_compat_golden.py`` compares the current behaviour with
them. See ``README.md`` in this directory.
"""

import os
import shutil
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, "fixtures")
GOLDEN = os.path.join(HERE, "golden")

#: name -> path. ``el`` stays within EL++; ``rich`` uses constructs up to SROIQ(D).
ONTOLOGIES = {
    "el": os.path.join(FIXTURES, "el.ofn"),
    "rich": os.path.join(FIXTURES, "rich.ofn"),
    "t3": os.path.join(HERE, "..", "..", "geometric", "fixtures", "t3.owl"),
    "roles": os.path.join(HERE, "..", "..", "geometric", "fixtures", "roles.owl"),
}

TOP = "http://www.w3.org/2002/07/owl#Thing"
BOT = "http://www.w3.org/2002/07/owl#Nothing"


def load(path):
    import java.io
    from mowl.owlapi import OWLAPIAdapter
    manager = OWLAPIAdapter().owl_manager
    for loaded in list(manager.getOntologies()):       # the manager is shared
        manager.removeOntology(loaded)
    return manager.loadOntologyFromOntologyDocument(java.io.File(os.path.abspath(path)))


def _iri(entity):
    return str(entity.toStringID())


def _iris(entities):
    return sorted(_iri(e) for e in entities)


# --------------------------------------------------------------------------- structure

def probe_signature(path):
    """Entities of the ontology by kind, and its IRI."""
    ont = load(path)
    oid = ont.getOntologyID()
    iri = oid.getOntologyIRI()
    return {
        "ontology_iri": str(iri.get().toString()) if iri.isPresent() else None,
        "classes": _iris(ont.getClassesInSignature()),
        "object_properties": _iris(ont.getObjectPropertiesInSignature()),
        "data_properties": _iris(ont.getDataPropertiesInSignature()),
        "individuals": _iris(ont.getIndividualsInSignature()),
        "annotation_properties": _iris(ont.getAnnotationPropertiesInSignature()),
        "datatypes": _iris(ont.getDatatypesInSignature()),
    }


def probe_axiom_counts(path):
    """Number of axioms per axiom type, and of logical axioms."""
    ont = load(path)
    counts = {}
    for axiom in ont.getAxioms():
        name = str(axiom.getAxiomType().getName())
        counts[name] = counts.get(name, 0) + 1
    return {"by_type": dict(sorted(counts.items())),
            "total": int(ont.getAxiomCount()),
            "logical": int(ont.getLogicalAxiomCount())}


def probe_annotations(path):
    """Annotation assertions with literal values: subject, property, text, language,
    datatype."""
    from org.semanticweb.owlapi.model import AxiomType, OWLLiteral
    ont = load(path)
    rows = []
    for axiom in ont.getAxioms(AxiomType.ANNOTATION_ASSERTION):
        value = axiom.getValue()
        if isinstance(value, OWLLiteral):
            rows.append([str(axiom.getSubject().toString()),
                         _iri(axiom.getProperty()), str(value.getLiteral()),
                         str(value.getLang()), _iri(value.getDatatype())])
        else:
            rows.append([str(axiom.getSubject().toString()),
                         _iri(axiom.getProperty()), "IRI:" + str(value.toString()), "", ""])
    return sorted(rows)


def probe_expressivity(path):
    """The description logic mOWL detects (mowl.logic)."""
    from mowl.logic import ontology_expressivity
    return str(ontology_expressivity(load(path)))


def probe_expressivity_reports(path):
    """What each fully described method would ignore of the ontology."""
    from mowl.catalog import list_methods
    from mowl.logic import check_ontology
    ont = load(path)
    result = {}
    for card in list_methods():
        if not getattr(card, "expressivity", ""):
            continue
        try:
            supported = card.supported_expressivity
        except Exception:
            continue
        if supported is None:
            continue
        report = check_ontology(ont, supported, warn=False)
        result[card.key] = {"ok": bool(report.ok), "summary": str(report.summary())}
    return result


# ----------------------------------------------------------------------- normalisation

def _gci_tuple(name, gci):
    if name in ("gci0", "gci0_bot"):
        return [gci.subclass, gci.superclass]
    if name in ("gci1", "gci1_bot"):
        return sorted([gci.left_subclass, gci.right_subclass]) + [gci.superclass]
    if name == "gci2":
        return [gci.subclass, gci.object_property, gci.filler]
    if name in ("gci3", "gci3_bot"):
        return [gci.object_property, gci.filler, gci.superclass]
    if name == "class_assertion":
        return [gci.class_, gci.individual]
    if name == "object_property_assertion":
        return [gci.subject, gci.object_property, gci.object_]
    raise KeyError(name)


def _normalised(path):
    from mowl.ontology.normalize import ELNormalizer
    return ELNormalizer().normalize(load(path), use_cache=False)


def probe_el_normalisation(path):
    """The EL normal forms. Axioms over named classes only are recorded exactly; the
    auxiliary classes that the normaliser introduces have arbitrary names, so for axioms
    that mention one only the number per normal form is recorded."""
    from mowl.ontology.normalize import is_auxiliary_iri
    gcis = _normalised(path)
    result = {}
    auxiliary = set()
    for name in sorted(gcis):
        named, with_aux = [], 0
        for gci in gcis[name]:
            row = [str(x) for x in _gci_tuple(name, gci)]
            aux = [x for x in row if is_auxiliary_iri(x)]
            if aux:
                with_aux += 1
                auxiliary.update(aux)
            else:
                named.append(row)
        result[name] = {"named": sorted(named), "with_auxiliary": with_aux}
    result["auxiliary_classes"] = len(auxiliary)
    return result


def probe_normalisation_preserves_classification(path):
    """ELK on the normalised axioms entails the same subsumptions between the named
    classes as ELK on the ontology itself (an invariant: recorded value is ``True``).
    The role axioms of the ontology are added unchanged."""
    from mowl.owlapi import OWLAPIAdapter
    from org.semanticweb.owlapi.model import IRI
    from org.semanticweb.owlapi.model import AxiomType
    source = load(path)
    original = _closure(source, "elk")["subsumptions"]
    gcis = _normalised(path)
    adapter = OWLAPIAdapter()
    normalised = adapter.owl_manager.createOntology(
        IRI.create("http://mowl.borg/compat/normalised/" + os.path.basename(path)))
    # The normaliser returns class axioms only; role axioms are kept as they are.
    for axiom in source.getAxioms():
        if axiom.isOfType(AxiomType.RBoxAxiomTypes):
            adapter.owl_manager.addAxiom(normalised, axiom)
    for name in gcis:
        for gci in gcis[name]:
            adapter.owl_manager.addAxiom(normalised, gci.owl_axiom)
    named = {c for pair in original for c in pair}
    after = [p for p in _closure(normalised, "elk")["subsumptions"]
             if p[0] in named and p[1] in named]
    adapter.owl_manager.removeOntology(normalised)
    missing = [p for p in original if p not in after]
    return {"preserved": not missing, "missing": missing}


def probe_role_axioms(path):
    """Role inclusions and role chains extracted for the EL++ models."""
    from mowl.ontology.normalize import extract_role_axioms
    inclusions, chains = extract_role_axioms(load(path))
    return {
        "role_inclusion": sorted([str(a.sub_property), str(a.super_property)]
                                 for a in inclusions),
        "role_chain": sorted([str(x) for x in a.sub_chain] + [str(a.super_property)]
                             for a in chains),
    }


def probe_el_dataset(path):
    """What an EL model gets from a dataset: index dictionaries and the number of rows
    per normal form (role axioms included)."""
    import mowl.datasets as D
    work = tempfile.mkdtemp(prefix="mowl_compat_")
    try:
        copy = os.path.join(work, os.path.basename(path))
        shutil.copy(path, copy)
        dataset = D.PathDataset(copy)
        from mowl.datasets.el import ELDataset
        el = ELDataset(dataset.ontology, load_role_axioms=True)
        gcis = el.get_gci_datasets()
        return {
            "classes": sorted(str(k) for k in dataset.classes.as_str),
            "object_properties": sorted(str(k) for k in dataset.object_properties.as_str),
            "individuals": sorted(str(k) for k in dataset.individuals.as_str),
            "rows": {name: len(gcis[name]) for name in sorted(gcis)},
        }
    finally:
        shutil.rmtree(work, ignore_errors=True)


# -------------------------------------------------------------------------- projection

def _edges(projector, ont):
    return sorted([str(e.src), str(e.rel), str(e.dst)] for e in projector.project(ont))


def probe_projections(path):
    """Edges of every graph projection."""
    from mowl.projection import (DL2VecProjector, OWL2VecStarProjector, TaxonomyProjector,
                                 TaxonomyWithRelationsProjector)
    result = {}
    result["taxonomy"] = _edges(TaxonomyProjector(), load(path))
    result["taxonomy_bidirectional"] = _edges(
        TaxonomyProjector(bidirectional_taxonomy=True), load(path))
    result["taxonomy_rels"] = _edges(
        TaxonomyWithRelationsProjector(taxonomy=True), load(path))
    result["dl2vec"] = _edges(DL2VecProjector(), load(path))
    result["dl2vec_bidirectional"] = _edges(
        DL2VecProjector(bidirectional_taxonomy=True), load(path))
    result["owl2vecstar"] = _edges(OWL2VecStarProjector(), load(path))
    result["owl2vecstar_literals"] = _edges(
        OWL2VecStarProjector(bidirectional_taxonomy=True, include_literals=True), load(path))
    return result


def probe_categorical_projection(path):
    """Nodes and edges of the categorical (CatE) projection."""
    from mowl.projection import CategoricalProjector
    edges = CategoricalProjector("str").project(load(path))
    rows = sorted([str(e.src), str(e.rel), str(e.dst)] for e in edges)
    return {"edges": len(rows), "relations": sorted({r[1] for r in rows}),
            "named_edges": [r for r in rows if r[0].startswith("http") and
                            r[2].startswith("http")]}


# ------------------------------------------------------------------------------ corpus

def probe_corpus(path):
    """The text corpora (Manchester syntax sentences of axioms, annotation sentences),
    as sorted lines."""
    from mowl.corpus import extract_annotation_corpus, extract_axiom_corpus
    return {"axioms": sorted(str(s) for s in extract_axiom_corpus(load(path))),
            "annotations": sorted(str(s) for s in extract_annotation_corpus(load(path)))}


# --------------------------------------------------------------------------- reasoning

def _closure(ont, backend):
    from mowl.reasoning import create_reasoner
    reasoner, _ = create_reasoner(ont, backend, warn=False)
    try:
        classes = sorted(ont.getClassesInSignature(), key=_iri)
        unsat = {_iri(c) for c in
                 reasoner.getUnsatisfiableClasses().getEntitiesMinusBottom()}
        pairs = []
        for c in classes:
            if _iri(c) in unsat or _iri(c) == BOT:
                continue
            supers = set(_iris(reasoner.getSuperClasses(c, False).getFlattened()))
            supers |= set(_iris(reasoner.getEquivalentClasses(c).getEntities()))
            for d in sorted(supers - {_iri(c), TOP}):
                pairs.append([_iri(c), d])
        types = []
        try:
            for individual in sorted(ont.getIndividualsInSignature(), key=_iri):
                for d in _iris(reasoner.getTypes(individual, False).getFlattened()):
                    if d != TOP:
                        types.append([_iri(individual), d])
        except Exception as error:                     # backend without instance queries
            types = "unsupported: " + type(error).__name__
        return {"consistent": bool(reasoner.isConsistent()),
                "unsatisfiable": sorted(unsat), "subsumptions": sorted(pairs),
                "types": types if isinstance(types, str) else sorted(types)}
    finally:
        reasoner.dispose()


def _reasoner_probe(backend):
    def probe(path):
        return _closure(load(path), backend)
    probe.__doc__ = f"Classification and realisation by the {backend!r} reasoner backend."
    return probe


probe_reasoner_structural = _reasoner_probe("structural")
probe_reasoner_elk = _reasoner_probe("elk")
probe_reasoner_hermit = _reasoner_probe("hermit")
probe_reasoner_km = _reasoner_probe("km")


def probe_reasoner_selection(path):
    """The backend chosen automatically for the ontology."""
    from mowl.logic import ontology_expressivity
    from mowl.reasoning import select_reasoner
    return str(select_reasoner(ontology_expressivity(load(path))).name)


def probe_subsumption_oracle(path):
    """Entailment of class expressions built at query time (existential restrictions and
    intersections of the first classes and roles), with the automatically chosen
    backend: the logical status of each query.

    On ``rich.ofn`` the backend is fixed to HermiT: the automatically chosen KM backend
    does not reliably answer on this ontology (see ``README.md``, known limits)."""
    from mowl.owlapi import OWLAPIAdapter
    from mowl.reasoning import SubsumptionOracle
    ont = load(path)
    adapter = OWLAPIAdapter()
    classes = sorted(ont.getClassesInSignature(), key=_iri)[:5]
    roles = sorted(ont.getObjectPropertiesInSignature(), key=_iri)[:2]
    backend = "hermit" if os.path.basename(path) == "rich.ofn" else "auto"
    oracle = SubsumptionOracle(ont, reasoner=backend)
    rows = []
    try:
        for c in classes:
            for d in classes:
                if c == d:
                    continue
                rows.append([_iri(c), _iri(d), str(oracle.status(c, d).name)])
                for r in roles:
                    some = adapter.create_object_some_values_from(r, d)
                    rows.append([_iri(c), "some " + _iri(r) + " " + _iri(d),
                                 str(oracle.status(c, some).name)])
                    both = adapter.create_object_intersection_of(c, some)
                    rows.append(["sat " + _iri(c) + " and some " + _iri(r) + " " + _iri(d),
                                 "", str(bool(oracle.is_satisfiable(both)))])
    finally:
        oracle.dispose()
    return rows


def probe_mowl_reasoner(path):
    """mowl.reasoning.MOWLReasoner over ELK: inferred subclass, equivalence and
    disjointness axioms (as IRIs)."""
    from mowl.reasoning import MOWLReasoner, create_reasoner
    ont = load(path)
    reasoner, _ = create_reasoner(ont, "elk", warn=False)
    try:
        wrapper = MOWLReasoner(reasoner)
        classes = sorted(ont.getClassesInSignature(), key=_iri)
        sub = sorted([_iri(a.getSubClass()), _iri(a.getSuperClass())]
                     for a in wrapper.infer_subclass_axioms(classes))
        equivalent = sorted(sorted(_iri(c) for c in a.getClassExpressions())
                            for a in wrapper.infer_equivalent_class_axioms(classes))
        return {"subclass": sub, "equivalent": [e for e in equivalent if len(e) > 1]}
    finally:
        reasoner.dispose()


# -------------------------------------------------------------------------- round trip

def _content(path):
    """Logical content that every syntax must keep: logical axioms per type and the
    entities. Declarations and the datatypes are left out: Manchester syntax declares
    built-in entities, and plain literals come back typed as ``xsd:string``."""
    counts = probe_axiom_counts(path)
    signature = probe_signature(path)
    signature.pop("datatypes")
    by_type = {k: v for k, v in counts["by_type"].items() if k != "Declaration"}
    return {"by_type": by_type, "logical": counts["logical"]}, signature


def probe_round_trip(path):
    """Saving and loading in every syntax keeps the content: for each syntax, whether
    the reloaded ontology has the same axioms per type (declarations aside) and the
    same entities."""
    import java.io
    from org.semanticweb.owlapi.formats import (FunctionalSyntaxDocumentFormat,
                                                ManchesterSyntaxDocumentFormat,
                                                OWLXMLDocumentFormat,
                                                RDFXMLDocumentFormat,
                                                TurtleDocumentFormat)
    from org.semanticweb.owlapi.model import IRI
    reference = _content(path)
    work = tempfile.mkdtemp(prefix="mowl_compat_")
    result = {}
    try:
        formats = {"rdfxml": RDFXMLDocumentFormat, "owlxml": OWLXMLDocumentFormat,
                   "functional": FunctionalSyntaxDocumentFormat,
                   "manchester": ManchesterSyntaxDocumentFormat,
                   "turtle": TurtleDocumentFormat}
        for name, fmt in formats.items():
            ont = load(path)
            out = os.path.join(work, f"saved.{name}")
            ont.getOWLOntologyManager().saveOntology(
                ont, fmt(), IRI.create(java.io.File(out).toURI()))
            again = _content(out)
            result[name] = {"axioms": again[0] == reference[0],
                            "entities": again[1] == reference[1]}
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return result


PROBES = {name[len("probe_"):]: fn for name, fn in sorted(globals().items())
          if name.startswith("probe_") and callable(fn)}

#: Probes that do not apply to an ontology (a reasoner outside its fragment).
SKIP = {
    ("rich", "reasoner_elk"), ("rich", "mowl_reasoner"),
    ("rich", "normalisation_preserves_classification"),
}


#: Cases whose reference behaviour is not stable enough to be a test: the golden file is
#: kept as a record but not compared (see ``README.md``, known limits).
UNSTABLE = {
    ("rich", "reasoner_km"),
}

#: Probes that start the KM reasoner. It runs a helper process and waits for it without a
#: time limit, so the tests run these under a watchdog.
USES_KM = {"reasoner_km", "subsumption_oracle"}


def cases():
    return [(o, p) for o in ONTOLOGIES for p in PROBES
            if (o, p) not in SKIP and (o, p) not in UNSTABLE]


def golden_path(ontology, probe):
    return os.path.join(GOLDEN, f"{ontology}.{probe}.json")
