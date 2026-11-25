from typing import Type
from org.semanticweb.owlapi.manchestersyntax.renderer import \
    ManchesterOWLSyntaxOWLObjectRendererImpl
from org.semanticweb.owlapi.model import OWLLiteral, OWLOntology
from org.semanticweb.owlapi.search import EntitySearcher
from deprecated.sphinx import deprecated

from jpype.types import JString

from org.mowl import MOWLShortFormProvider
import logging


ENTITY_FORMS = ("iri", "bracketed iri")


def _check_entity_form(entity_form):
    if entity_form not in ENTITY_FORMS:
        raise ValueError(f"entity_form must be one of {ENTITY_FORMS}, got {entity_form!r}.")


def _render_axioms(renderer, ontology):
    """The rendered axioms, sorted: the ontology yields its axioms in the order of a Java hash
    set, which differs from one load to the next in one process."""
    rendered = []
    for axiom in ontology.getAxioms():
        rax = renderer.render(axiom)
        rax = rax.replaceAll(JString("[\\r\\n|\\r|\\n()|<|>]"), JString(""))
        rendered.append(str(rax))
    return sorted(rendered)


def _annotation_lines(ontology, entity_form):
    """One line ``<entity> <property> <text>`` per literal annotation of a class or an
    individual, classes first, each group sorted by IRI (the signature is a Java hash set, whose
    order differs from one load to the next). The entity is its IRI, or ``<IRI>`` for classes
    with ``entity_form="bracketed iri"``."""
    lines = []
    entities = sorted(ontology.getClassesInSignature(), key=lambda e: str(e.toStringID()))
    entities += sorted(ontology.getIndividualsInSignature(), key=lambda e: str(e.toStringID()))
    n_classes = len(ontology.getClassesInSignature())
    for i, entity in enumerate(entities):
        if i < n_classes and entity_form == "bracketed iri":
            name = str(entity)
        else:
            name = str(entity.toStringID())
        annotations = EntitySearcher.getAnnotations(entity, ontology)
        found = []
        for annotation in annotations:
            if isinstance(annotation.getValue(), OWLLiteral):
                obj_property = str(annotation.getProperty()).replace("\n", " ")
                # could filter on property
                value = str(annotation.getValue().getLiteral()).replace("\n", " ")
                found.append(f'{name} {obj_property} {value}\n')
        lines += sorted(found)
    return lines


def extract_and_save_axiom_corpus(ontology, out_file, mode="w"):
    """Method to extract axioms of a particular ontology and save it into a file.

    :param ontology: Input ontology.
    :type ontology: :class:`org.semanticweb.owlapi.model.OWLOntology`
    :param out_file: File path to save the extracted axioms.
    :type out_file: str
    :param mode: mode for opening the `out_file`, defaults to `"w"`
    :type mode: str, optional
    """

    if not isinstance(ontology, OWLOntology):
        raise TypeError(
            "Parameter ontology must be of type org.semanticweb.owlapi.model.OWLOntology")
    if not isinstance(out_file, str):
        raise TypeError("Parameter out_file must be of type str")
    if not isinstance(mode, str):
        raise TypeError("Optional parameter mode must be of type str")

    if mode not in ["w", "a"]:
        raise ValueError("Parameter mode must be a file reading mode. Options are 'a' or 'w'")

    logging.info("Generating axioms corpus")
    renderer = ManchesterOWLSyntaxOWLObjectRendererImpl()
    shortFormProvider = MOWLShortFormProvider()
    renderer.setShortFormProvider(shortFormProvider)
    with open(out_file, mode) as f:
        for rax in _render_axioms(renderer, ontology):
            f.write(f'{rax}\n')


def extract_axiom_corpus(ontology):
    """Method to extract axioms of a particular ontology. Similar to \
:func:`extract_and_save_axiom_corpus` but this method returns a list instead saving into a file.

    :param ontology: Input ontology.
    :type ontology: :class:`org.semanticweb.owlapi.model.OWLOntology`
    :rtype: list[str]
    """

    if not isinstance(ontology, OWLOntology):
        raise TypeError(
            "Parameter ontology must be of type org.semanticweb.owlapi.model.OWLOntology")

    logging.info("Generating axioms corpus")
    renderer = ManchesterOWLSyntaxOWLObjectRendererImpl()
    shortFormProvider = MOWLShortFormProvider()
    renderer.setShortFormProvider(shortFormProvider)

    corpus = []

    corpus += _render_axioms(renderer, ontology)
    return corpus


def extract_and_save_annotation_corpus(ontology, out_file, mode="w", entity_form="iri"):
    """This method generates a textual representation of the annotation axioms in an ontology \
following the Manchester Syntax.

    The lines are sorted (classes, then individuals, each by IRI), so the same ontology gives \
the same file on every load.

    :param ontology: OWL ontology from which the annotations will be extracted.
    :type ontology: :class:`org.semanticweb.owlapi.model.OWLOntology`
    :param out_file: File path to save the extracted annotations.
    :type out_file: str
    :param mode: mode for opening the `out_file`, defaults to `"w"`
    :type mode: str ,optional
    :param entity_form: How a class is written. ``"iri"`` (default): the IRI, the same token \
as in the axiom corpus and in :func:`extract_annotation_corpus`, so that the annotation is \
about the class vector. ``"bracketed iri"``: ``<IRI>``, a token no axiom sentence contains (the \
output of this function before the option existed, kept to reproduce it). Individuals are \
written as IRI in both forms.
    :type entity_form: str, optional
    """

    if not isinstance(ontology, OWLOntology):
        raise TypeError(
            "Parameter ontology must be of type org.semanticweb.owlapi.model.OWLOntology")
    if not isinstance(out_file, str):
        raise TypeError("Parameter out_file must be of type str")
    if not isinstance(mode, str):
        raise TypeError("Optional parameter mode must be of type str")

    if mode not in ["w", "a"]:
        raise ValueError("Parameter mode must be a file reading mode. Options are 'a' or 'w'")
    _check_entity_form(entity_form)

    logging.info("Generating annotation corpus")
    with open(out_file, mode) as f:
        for line in _annotation_lines(ontology, entity_form):
            f.write(line)


def extract_annotation_corpus(ontology, entity_form="iri"):
    """This method generates a textual representation of the annotation axioms in an ontology \
following the Manchester Syntax. Similar to :func:`extract_and_save_annotation_corpus` but \
this method returns a list instead saving into a file.

    :param ontology: Input ontology
    :type ontology: :class:`org.semanticweb.owlapi.model.OWLOntology`
    :param entity_form: ``"iri"`` (default) or ``"bracketed iri"``, see \
:func:`extract_and_save_annotation_corpus`.
    :type entity_form: str, optional
    :rtype: list[str]
    """

    if not isinstance(ontology, OWLOntology):
        raise TypeError(
            "Parameter ontology must be of type org.semanticweb.owlapi.model.OWLOntology")
    _check_entity_form(entity_form)

    logging.info("Generating annotation corpus")
    return _annotation_lines(ontology, entity_form)
