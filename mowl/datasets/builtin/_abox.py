"""Helpers of the datasets that split the ABox of an ontology: entailed assertions by a
reasoner, and writing an ontology atomically."""

import os


def entailed_assertions(ontology, reasoner="hermit", individuals=None):
    """The entailed atomic class assertions and object property assertions of ``ontology``.

    The class assertions are the named classes of the (indirect) types of every individual
    (``owl:Thing`` excluded), the role assertions the entailed fillers of every object
    property of the signature. The ontology must be consistent.

    :param ontology: An OWL API ontology.
    :param reasoner: A backend of :mod:`mowl.reasoning` (default HermiT).
    :param individuals: Restrict to these individuals (OWL API objects); default: all in
        the signature.
    :return: ``(class_assertions, role_assertions)``: sorted lists of
        ``(individual IRI, class IRI)`` and ``(subject IRI, role IRI, object IRI)``.
    :raises ValueError: If the ontology is inconsistent.
    """
    from mowl.reasoning import create_reasoner
    engine, _ = create_reasoner(ontology, reasoner, warn=False)
    try:
        if not engine.isConsistent():
            raise ValueError("The ontology is inconsistent: its entailed assertions are "
                             "undefined.")
        props = list(ontology.getObjectPropertiesInSignature())
        class_assertions, role_assertions = set(), set()
        inds = list(ontology.getIndividualsInSignature()) if individuals is None \
            else list(individuals)
        for ind in inds:
            a = str(ind.toStringID())
            for node in engine.getTypes(ind, False).getNodes():
                for cls in node.getEntities():
                    if not cls.isOWLThing():
                        class_assertions.add((a, str(cls.toStringID())))
            for prop in props:
                if prop.isOWLTopObjectProperty():
                    continue
                for other in engine.getObjectPropertyValues(ind, prop).getFlattened():
                    role_assertions.add((a, str(prop.toStringID()), str(other.toStringID())))
    finally:
        engine.dispose()
    return sorted(class_assertions), sorted(role_assertions)


def tbox_axioms(ontology):
    """The axioms of ``ontology`` that are neither declarations nor assertions about
    individuals (TBox and RBox), as a Python list of OWL API axioms."""
    from org.semanticweb.owlapi.model import AxiomType
    skip = {AxiomType.DECLARATION, AxiomType.CLASS_ASSERTION,
            AxiomType.OBJECT_PROPERTY_ASSERTION, AxiomType.NEGATIVE_OBJECT_PROPERTY_ASSERTION,
            AxiomType.DATA_PROPERTY_ASSERTION, AxiomType.NEGATIVE_DATA_PROPERTY_ASSERTION,
            AxiomType.SAME_INDIVIDUAL, AxiomType.DIFFERENT_INDIVIDUALS,
            AxiomType.ANNOTATION_ASSERTION}
    return [ax for ax in ontology.getAxioms() if ax.getAxiomType() not in skip]


def write_ontology(adapter, iri, axioms, classes=(), roles=(), individuals=(),
                   class_assertions=(), role_assertions=(), path=None):
    """Write an ontology in functional syntax, atomically (``.part`` + ``os.replace``).

    :param adapter: An :class:`~mowl.owlapi.OWLAPIAdapter`.
    :param axioms: OWL API axioms to include as they are.
    :param classes, roles, individuals: IRIs to declare.
    :param class_assertions: ``(individual IRI, class IRI)`` pairs.
    :param role_assertions: ``(subject IRI, role IRI, object IRI)`` triples.
    """
    import java.io
    from java.util import HashSet
    from org.semanticweb.owlapi.formats import FunctionalSyntaxDocumentFormat
    from org.semanticweb.owlapi.model import IRI
    ont = adapter.create_ontology(iri)
    df = adapter.data_factory
    out = HashSet()
    for ax in axioms:
        out.add(ax)
    for c in classes:
        out.add(df.getOWLDeclarationAxiom(adapter.create_class(c)))
    for r in roles:
        out.add(df.getOWLDeclarationAxiom(adapter.create_object_property(r)))
    for i in individuals:
        out.add(df.getOWLDeclarationAxiom(adapter.create_individual(i)))
    for i, c in class_assertions:
        out.add(adapter.create_class_assertion(adapter.create_class(c),
                                               adapter.create_individual(i)))
    for a, r, b in role_assertions:
        out.add(adapter.create_object_property_assertion(
            adapter.create_object_property(r), adapter.create_individual(a),
            adapter.create_individual(b)))
    adapter.owl_manager.addAxioms(ont, out)
    partial = f"{path}.{os.getpid()}.part"
    adapter.owl_manager.saveOntology(ont, FunctionalSyntaxDocumentFormat(),
                                     IRI.create(java.io.File(partial).toURI()))
    os.replace(partial, path)
    adapter.owl_manager.removeOntology(ont)
    return path
