"""Labels of OWL entities and natural-language verbalisation of :math:`\\mathcal{EL}` concept
expressions.

Text-based embedding methods (HiT, OnT) encode concepts by the text of their labels.
:func:`entity_labels` collects ``rdfs:label`` annotations (or another annotation property)
of classes, object properties and individuals, falling back to the IRI fragment.
:class:`Verbaliser` turns class expressions into text with the compositional rules of OnT
(Yang et al., ISWC 2025, Section 4.1):

.. list-table::
   :header-rows: 1

   * - Expression
     - ``style="paper"`` (OnT, Section 4.1)
     - ``style="release"`` (OnT data release)
   * - :math:`A`
     - :math:`V(A)`, the label of :math:`A`
     - same
   * - :math:`C \\sqcap D`
     - "*V(C)* and *V(D)*"
     - same
   * - :math:`\\exists r.C`
     - "something that *V(r)* some *V(C)*"
     - "*V(r)* some *V(C)*"

The ``"release"`` style reproduces the texts of the published OnT training and test data
(``normalization/transferID2text.py`` of the reference code builds ``f"{role} some
{concept}"``). ``style="deeponto"`` follows the rules of DeepOnto's ``OntologyVerbaliser``
(He et al., 2023, with quantifier words), which the OnT normaliser uses to name the
concepts it introduces: existential restrictions inside a conjunction are attached with
"that" ("*V(A)* that *V(r)* some *V(C)*"), restrictions on the same role are merged, and a
conjunction of restrictions only starts with "something that". DeepOnto's optional
spaCy-based auto-correction of labels is not reproduced.

:math:`\\top` and :math:`\\bot` are verbalised as "thing" and "nothing", and a nominal
:math:`\\{a\\}` as the label of :math:`a`. Other constructors are outside
:math:`\\mathcal{EL}` and raise :class:`NotImplementedError`.

The operands of ``ObjectIntersectionOf`` are taken in the order the OWL API returns them
(its canonical sorted order), which need not be the order in which they were written.

Requires a running JVM (:func:`mowl.init_jvm`).
"""

import re

from org.semanticweb.owlapi.model import (OWLClass, OWLClassExpression, OWLEntity,
                                          OWLLiteral, OWLObjectIntersectionOf,
                                          OWLObjectOneOf, OWLObjectProperty,
                                          OWLObjectSomeValuesFrom)

from mowl.owlapi.defaults import TOP, BOT

#: IRI of ``rdfs:label``.
RDFS_LABEL = "http://www.w3.org/2000/01/rdf-schema#label"

STYLES = ("paper", "release", "deeponto")


def iri_fragment(iri):
    """The part of an IRI after the last ``#``, or else after the last ``/``.

    >>> iri_fragment("http://www.co-ode.org/ontologies/galen#RectalArtery")
    'RectalArtery'
    """
    iri = str(iri).strip("<>")
    for sep in ("#", "/"):
        if sep in iri:
            frag = iri.rsplit(sep, 1)[1]
            if frag:
                return frag
    return iri


_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")


def split_camel_case(name):
    """Split a CamelCase or snake_case identifier into lower-case words; acronyms
    (several capitals) keep their case.

    >>> split_camel_case("ABOBloodGroupTest")
    'ABO blood group test'
    >>> split_camel_case("hasStructuralComponent")
    'has structural component'
    """
    words = []
    for part in re.split(r"[_\-\s]+", name):
        for w in _CAMEL.findall(part):
            words.append(w if (len(w) > 1 and w.isupper()) else w.lower())
    return " ".join(words) if words else name


def _entity_iri(entity):
    if isinstance(entity, str):
        return entity.strip("<>")
    return str(entity.toStringID())


def _literal_labels(ontology, iri, label_property):
    from org.semanticweb.owlapi.model import IRI
    java_iri = IRI.create(iri)
    found = []
    for ont in ontology.getImportsClosure():
        for ax in ont.getAnnotationAssertionAxioms(java_iri):
            if str(ax.getProperty().toStringID()) != label_property:
                continue
            value = ax.getValue()
            if isinstance(value, OWLLiteral):
                found.append((str(value.getLang()), str(value.getLiteral())))
    return found


def _choose(found, lang):
    """Prefer the requested language, then untagged literals, then anything; ties are
    broken by the literal (deterministic)."""
    if not found:
        return None
    for wanted in ([lang] if lang else []) + [""]:
        cands = sorted(text for tag, text in found if tag.lower() == (wanted or "").lower())
        if cands:
            return cands[0]
    return sorted(text for _, text in found)[0]


def fallback_label(iri, fallback="fragment"):
    """Label used when an entity has no label annotation.

    :param fallback: ``"fragment"`` (the IRI fragment), ``"camel_case"`` (the fragment
        split by :func:`split_camel_case`) or ``None`` (raise :class:`KeyError`).
    """
    if fallback is None:
        raise KeyError(f"No label found for {iri}.")
    frag = iri_fragment(iri)
    if fallback == "fragment":
        return frag
    if fallback == "camel_case":
        return split_camel_case(frag)
    raise ValueError(f"Unknown fallback {fallback!r}.")


def entity_labels(ontology, entities=None, label_property=RDFS_LABEL, lang="en",
                  fallback="fragment"):
    """Labels of the classes, object properties and individuals of an ontology.

    For every entity the ``label_property`` annotations (in the ontology and its imports
    closure) are read; if there are several, an ``@lang`` literal is preferred, then an
    untagged one, and among those the alphabetically first (HiT, footnote 9, takes "the
    first name (in English)"). Entities without a label get :func:`fallback_label`.

    :param ontology: An ``OWLOntology``.
    :param entities: IRIs or OWL entities to label; default: all classes, object
        properties and individuals in the signature.
    :param label_property: IRI of the annotation property.
    :param lang: Preferred language tag.
    :param fallback: See :func:`fallback_label`.
    :return: ``{iri: label}``.
    :rtype: dict
    """
    if entities is None:
        entities = list(ontology.getClassesInSignature(True))
        entities.extend(ontology.getObjectPropertiesInSignature(True))
        entities.extend(ontology.getIndividualsInSignature(True))
    out = {}
    for entity in entities:
        iri = _entity_iri(entity)
        label = _choose(_literal_labels(ontology, iri, label_property), lang)
        out[iri] = label if label is not None else fallback_label(iri, fallback)
    return out


class Verbaliser:
    """Natural-language verbalisation of :math:`\\mathcal{EL}` class expressions.

    :param ontology: Ontology to read labels from (optional if ``labels`` is complete).
    :param labels: ``{iri: label}`` overriding or replacing the ontology's labels.
    :param style: ``"paper"``, ``"release"`` or ``"deeponto"`` (see the module docstring).
    :param label_property: Annotation property of labels (default ``rdfs:label``).
    :param lang: Preferred language of labels.
    :param fallback: Label of entities without one, see :func:`fallback_label`.
    """

    def __init__(self, ontology=None, labels=None, style="paper", label_property=RDFS_LABEL,
                 lang="en", fallback="fragment"):
        if style not in STYLES:
            raise ValueError(f"style must be one of {STYLES}, got {style!r}.")
        self.ontology = ontology
        self.labels = dict(labels or {})
        self.style = style
        self.label_property = label_property
        self.lang = lang
        self.fallback = fallback

    def label(self, entity):
        """Label of an entity (OWL entity or IRI string), cached."""
        iri = _entity_iri(entity)
        if iri not in self.labels:
            if iri == TOP:
                return "thing"
            if iri == BOT:
                return "nothing"
            found = (_literal_labels(self.ontology, iri, self.label_property)
                     if self.ontology is not None else [])
            label = _choose(found, self.lang)
            self.labels[iri] = label if label is not None else fallback_label(
                iri, self.fallback)
        return self.labels[iri]

    def role(self, prop):
        """Verbalisation :math:`V(r)` of an object property."""
        if not isinstance(prop, OWLObjectProperty):
            raise NotImplementedError(
                f"Only named object properties can be verbalised, got {prop}.")
        return self.label(prop)

    def __call__(self, expr):
        return self.verbalise(expr)

    def verbalise(self, expr):
        """Verbalise a class expression (or an entity, or an IRI string)."""
        if isinstance(expr, str):
            return self.label(expr)
        if isinstance(expr, OWLClass):
            if expr.isOWLThing():
                return "thing"
            if expr.isOWLNothing():
                return "nothing"
            return self.label(expr)
        if isinstance(expr, OWLObjectProperty):
            return self.role(expr)
        if isinstance(expr, OWLEntity):
            return self.label(expr)
        if isinstance(expr, OWLObjectOneOf):
            # a set in the OWL API: sorted, so that the text does not depend on its order
            return " or ".join(sorted(self.label(i) for i in expr.getIndividuals()))
        if self.style == "deeponto":
            return self._deeponto(expr, add_something=True)
        if isinstance(expr, OWLObjectIntersectionOf):
            return " and ".join(self.verbalise(op) for op in expr.getOperandsAsList())
        if isinstance(expr, OWLObjectSomeValuesFrom):
            body = f"{self.role(expr.getProperty())} some {self.verbalise(expr.getFiller())}"
            return "something that " + body if self.style == "paper" else body
        raise NotImplementedError(self._unsupported(expr))

    @staticmethod
    def _unsupported(expr):
        kind = str(expr.getClassExpressionType()) if isinstance(expr, OWLClassExpression) \
            else type(expr).__name__
        return f"Cannot verbalise {kind}: only EL constructors are supported ({expr})."

    def _deeponto(self, expr, add_something):
        if isinstance(expr, OWLObjectSomeValuesFrom):
            body = f"{self.role(expr.getProperty())} some {self.verbalise(expr.getFiller())}"
            return "something that " + body if add_something else body
        if isinstance(expr, OWLObjectIntersectionOf):
            others, merged = [], {}
            for op in expr.getOperandsAsList():
                if isinstance(op, OWLObjectSomeValuesFrom):
                    merged.setdefault(self.role(op.getProperty()), []).append(
                        self.verbalise(op.getFiller()))
                else:
                    others.append(self.verbalise(op))
            restrictions = [f"{r} some " + " and ".join(fillers)
                            for r, fillers in merged.items()]
            if not others:
                return "something that " + " and ".join(restrictions)
            text = " and ".join(others)
            if restrictions:
                text += " that " + " and ".join(restrictions)
            return text
        if isinstance(expr, (OWLClass, OWLObjectOneOf)):
            return self.verbalise(expr)
        raise NotImplementedError(self._unsupported(expr))

    def axiom_sides(self, axiom):
        """Verbalised ``(subclass, superclass)`` pairs of a ``SubClassOf`` or
        ``EquivalentClasses`` axiom (an equivalence of :math:`n` classes gives all ordered
        pairs). Other axioms give an empty list."""
        from org.semanticweb.owlapi.model import AxiomType
        if axiom.getAxiomType() == AxiomType.SUBCLASS_OF:
            return [(self.verbalise(axiom.getSubClass()),
                     self.verbalise(axiom.getSuperClass()))]
        if axiom.getAxiomType() == AxiomType.EQUIVALENT_CLASSES:
            ops = list(axiom.getClassExpressionsAsList())
            return [(self.verbalise(a), self.verbalise(b))
                    for i, a in enumerate(ops) for j, b in enumerate(ops) if i != j]
        return []
