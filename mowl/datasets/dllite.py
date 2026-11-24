"""A small :math:`\\text{DL-Lite}_{\\mathcal{H}}` front end.

:math:`\\text{DL-Lite}_{\\mathcal{H}}` (Artale et al., 2009) has *basic concepts*
:math:`B ::= A \\mid \\exists R \\mid \\exists R^-` (:math:`\\exists R` abbreviates
:math:`\\exists R.\\top`), concept inclusions :math:`B \\sqsubseteq B'` and
:math:`B \\sqsubseteq \\neg B'`, role inclusions :math:`S \\sqsubseteq T` between roles or
inverse roles, and assertions :math:`B(a)` and :math:`R(a, b)`. This is the language of
BoxLitE (Lourenço et al., KR 2026, Section 2).

:func:`extract_dllite` reads exactly these axioms from an OWL ontology with the OWL API and
returns a :class:`DLLiteKB` of integer index arrays. OWL axioms are translated only when
the translation is equivalent:

========================================================  ===============================
OWL axiom                                                 DL-Lite_H axioms
========================================================  ===============================
``SubClassOf(B C)``                                       :math:`B \\sqsubseteq C`
``SubClassOf(B ObjectComplementOf(C))``                   :math:`B \\sqsubseteq \\neg C`
``SubClassOf(B owl:Nothing)``                             :math:`B \\sqsubseteq \\neg B`
``SubClassOf(B ObjectIntersectionOf(C1 ... Cn))``         one inclusion per conjunct
``EquivalentClasses(B1 ... Bn)``                          pairwise inclusions
``DisjointClasses(B1 ... Bn)``                            :math:`B_i \\sqsubseteq \\neg B_j`
``ObjectPropertyDomain(R C)`` / ``ObjectPropertyRange``   :math:`\\exists R \\sqsubseteq C`
                                                          / :math:`\\exists R^- \\sqsubseteq C`
``SubObjectPropertyOf(S T)``                              :math:`S \\sqsubseteq T`
``EquivalentObjectProperties`` / ``InverseObjectProp.``   two role inclusions
``SymmetricObjectProperty(R)``                            :math:`R \\sqsubseteq R^-`
``ClassAssertion(B a)``                                   :math:`B(a)`
``ObjectPropertyAssertion(S a b)``                        :math:`R(a, b)` (or
                                                          :math:`R(b, a)` if
                                                          :math:`S = R^-`)
========================================================  ===============================

where :math:`B, B_i` are basic concepts (a named class or ``ObjectSomeValuesFrom(S
owl:Thing)``) and :math:`C, C_i` basic concepts or complements of basic concepts on the
right-hand side. Everything else (conjunction or qualified existentials on the left,
universal restrictions, role chains, negative assertions, ...) is outside
:math:`\\text{DL-Lite}_{\\mathcal{H}}`: it is **not** translated, and counted in
:attr:`DLLiteKB.skipped` by reason. Axioms that are trivially true (``owl:Thing`` on the
right, ``owl:Nothing`` on the left) are counted in :attr:`DLLiteKB.trivial`; declarations
and annotation axioms in :attr:`DLLiteKB.ignored`. Nothing is dropped silently.

Basic concepts are numbered as in the BoxLitE reference code: concept names
``0 .. |N_C|-1``, then :math:`\\exists R_j` at ``|N_C| + j`` and :math:`\\exists R_j^-` at
``|N_C| + |N_R| + j``.
"""

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

THING = "http://www.w3.org/2002/07/owl#Thing"
NOTHING = "http://www.w3.org/2002/07/owl#Nothing"

_NON_LOGICAL = {"Declaration", "AnnotationAssertion", "SubAnnotationPropertyOf",
                "AnnotationPropertyDomain", "AnnotationPropertyRangeOf"}


def _empty(cols):
    return np.zeros((0, cols), dtype=np.int64)


@dataclass
class DLLiteKB:
    """A :math:`\\text{DL-Lite}_{\\mathcal{H}}` knowledge base as index arrays.

    :param concepts: IRIs (or names) of the concept names :math:`N_C`.
    :param roles: IRIs of the role names :math:`N_R`.
    :param individuals: IRIs of the individual names :math:`N_I`.
    :param concept_inclusions: ``(n, 2)`` rows ``[B1, B2]`` for :math:`B_1 \\sqsubseteq B_2`
        (basic-concept indices).
    :param negative_inclusions: ``(n, 2)`` rows ``[B1, B2]`` for
        :math:`B_1 \\sqsubseteq \\neg B_2`.
    :param role_inclusions: ``(n, 4)`` rows ``[R, inv_R, S, inv_S]`` for
        :math:`R^{(-)} \\sqsubseteq S^{(-)}` (``inv`` is 0 or 1).
    :param concept_assertions: ``(n, 2)`` rows ``[B, a]`` for :math:`B(a)`.
    :param role_assertions: ``(n, 3)`` rows ``[R, a, b]`` for :math:`R(a, b)`.
    :param skipped: Logical axioms outside the language, ``{reason: count}``.
    :param trivial: Tautological axioms that were not translated, ``{type: count}``.
    :param ignored: Non-logical axioms (declarations, annotations), ``{type: count}``.
    """

    concepts: list
    roles: list
    individuals: list
    concept_inclusions: np.ndarray = field(default_factory=lambda: _empty(2))
    negative_inclusions: np.ndarray = field(default_factory=lambda: _empty(2))
    role_inclusions: np.ndarray = field(default_factory=lambda: _empty(4))
    concept_assertions: np.ndarray = field(default_factory=lambda: _empty(2))
    role_assertions: np.ndarray = field(default_factory=lambda: _empty(3))
    skipped: dict = field(default_factory=dict)
    trivial: dict = field(default_factory=dict)
    ignored: dict = field(default_factory=dict)

    def __post_init__(self):
        self.concepts = list(self.concepts)
        self.roles = list(self.roles)
        self.individuals = list(self.individuals)
        for name, cols in (("concept_inclusions", 2), ("negative_inclusions", 2),
                           ("role_inclusions", 4), ("concept_assertions", 2),
                           ("role_assertions", 3)):
            arr = np.asarray(getattr(self, name), dtype=np.int64)
            if arr.size == 0:
                arr = _empty(cols)
            if arr.ndim != 2 or arr.shape[1] != cols:
                raise ValueError(f"{name} must have shape (n, {cols}), got {arr.shape}.")
            setattr(self, name, arr)
        nb, nr, ni = self.n_basic, len(self.roles), len(self.individuals)
        for name, arr, bound in (("concept_inclusions", self.concept_inclusions, nb),
                                 ("negative_inclusions", self.negative_inclusions, nb)):
            if arr.size and (arr.min() < 0 or arr.max() >= bound):
                raise ValueError(f"{name}: basic-concept index out of range.")
        if self.role_inclusions.size:
            r = self.role_inclusions
            if r[:, [0, 2]].min() < 0 or r[:, [0, 2]].max() >= nr or \
                    not np.isin(r[:, [1, 3]], (0, 1)).all():
                raise ValueError("role_inclusions: bad role index or inverse flag.")
        if self.concept_assertions.size:
            c = self.concept_assertions
            if c[:, 0].min() < 0 or c[:, 0].max() >= nb or c[:, 1].min() < 0 or \
                    c[:, 1].max() >= ni:
                raise ValueError("concept_assertions: index out of range.")
        if self.role_assertions.size:
            r = self.role_assertions
            if r[:, 0].min() < 0 or r[:, 0].max() >= nr or r[:, 1:].min() < 0 or \
                    r[:, 1:].max() >= ni:
                raise ValueError("role_assertions: index out of range.")

    # ----------------------------------------------------------------------------------
    # Basic concepts
    # ----------------------------------------------------------------------------------

    @property
    def n_basic(self):
        """:math:`|N_C| + 2|N_R|`, the number of basic concepts (:math:`N^{\\exists}_C`)."""
        return len(self.concepts) + 2 * len(self.roles)

    def exists_index(self, role, inverse=False):
        """Basic-concept index of :math:`\\exists R` (or :math:`\\exists R^-`)."""
        return len(self.concepts) + int(inverse) * len(self.roles) + role

    def basic_concept(self, index):
        """``("concept", i)``, ``("exists", j, False)`` or ``("exists", j, True)``."""
        nc, nr = len(self.concepts), len(self.roles)
        if index < nc:
            return ("concept", index)
        j = index - nc
        return ("exists", j % nr, j >= nr)

    def basic_concept_name(self, index):
        """Readable name of a basic concept: ``A``, ``∃R`` or ``∃R⁻``."""
        kind = self.basic_concept(index)
        if kind[0] == "concept":
            return _short(self.concepts[kind[1]])
        return "∃" + _short(self.roles[kind[1]]) + ("⁻" if kind[2] else "")

    def summary(self):
        """Axiom counts, including skipped, trivial and ignored axioms."""
        return {
            "concepts": len(self.concepts), "roles": len(self.roles),
            "individuals": len(self.individuals),
            "concept_inclusions": len(self.concept_inclusions),
            "negative_inclusions": len(self.negative_inclusions),
            "role_inclusions": len(self.role_inclusions),
            "concept_assertions": len(self.concept_assertions),
            "role_assertions": len(self.role_assertions),
            "skipped": sum(self.skipped.values()), "trivial": sum(self.trivial.values()),
            "ignored": sum(self.ignored.values())}

    def as_tensors(self):
        """The index arrays as ``torch.LongTensor`` (``{name: tensor}``)."""
        import torch as th
        return {name: th.as_tensor(getattr(self, name), dtype=th.long)
                for name in ("concept_inclusions", "negative_inclusions", "role_inclusions",
                             "concept_assertions", "role_assertions")}

    def with_assertions(self, concept_assertions=None, role_assertions=None):
        """A copy with the same signature and TBox and the given ABox."""
        return DLLiteKB(
            self.concepts, self.roles, self.individuals, self.concept_inclusions,
            self.negative_inclusions, self.role_inclusions,
            _empty(2) if concept_assertions is None else concept_assertions,
            _empty(3) if role_assertions is None else role_assertions)

    # ----------------------------------------------------------------------------------
    # Construction without the OWL API (tests, small examples)
    # ----------------------------------------------------------------------------------

    @classmethod
    def from_lists(cls, concepts=(), roles=(), individuals=(), inclusions=(),
                   negative_inclusions=(), role_inclusions=(), concept_assertions=(),
                   role_assertions=()):
        """Build a KB from names.

        Basic concepts are written ``"A"``, ``"∃R"`` or ``"∃R-"`` (``"some R"`` and
        ``"some R-"`` are accepted too, and ``⁻`` for ``-``); roles ``"R"`` or ``"R-"``.
        Names not listed in ``concepts``, ``roles`` or ``individuals`` are added in order of
        appearance.

        :param inclusions: Pairs ``(B1, B2)`` for :math:`B_1 \\sqsubseteq B_2`.
        :param negative_inclusions: Pairs ``(B1, B2)`` for :math:`B_1 \\sqsubseteq \\neg B_2`.
        :param role_inclusions: Pairs ``(S, T)`` for :math:`S \\sqsubseteq T`.
        :param concept_assertions: Pairs ``(B, a)``.
        :param role_assertions: Triples ``(S, a, b)``; ``S = R-`` gives :math:`R(b, a)`.
        """
        concepts, roles, individuals = list(concepts), list(roles), list(individuals)

        def role(text):
            text = text.strip().replace("⁻", "-")
            inv = text.endswith("-")
            name = text[:-1] if inv else text
            if name not in roles:
                roles.append(name)
            return name, inv

        def basic(text):
            text = text.strip()
            for prefix in ("∃", "some "):
                if text.startswith(prefix):
                    return ("exists",) + role(text[len(prefix):])
            if text not in concepts:
                concepts.append(text)
            return ("concept", text)

        def ind(name):
            if name not in individuals:
                individuals.append(name)
            return name

        ci = [(basic(a), basic(b)) for a, b in inclusions]
        ni = [(basic(a), basic(b)) for a, b in negative_inclusions]
        ri = [(role(a), role(b)) for a, b in role_inclusions]
        ca = [(basic(b), ind(a)) for b, a in concept_assertions]
        ra = [(role(r), ind(a), ind(b)) for r, a, b in role_assertions]

        cidx = {c: i for i, c in enumerate(concepts)}
        ridx = {r: i for i, r in enumerate(roles)}
        iidx = {a: i for i, a in enumerate(individuals)}
        nc, nr = len(concepts), len(roles)

        def b_index(b):
            if b[0] == "concept":
                return cidx[b[1]]
            return nc + int(b[2]) * nr + ridx[b[1]]

        ra_rows = []
        for (name, inv), a, b in ra:
            s, o = (b, a) if inv else (a, b)
            ra_rows.append([ridx[name], iidx[s], iidx[o]])
        return cls(
            concepts, roles, individuals,
            [[b_index(a), b_index(b)] for a, b in ci],
            [[b_index(a), b_index(b)] for a, b in ni],
            [[ridx[a[0]], int(a[1]), ridx[b[0]], int(b[1])] for a, b in ri],
            [[b_index(b), iidx[a]] for b, a in ca],
            ra_rows)


def _short(iri):
    iri = str(iri)
    for sep in ("#", "/"):
        if sep in iri:
            iri = iri.rsplit(sep, 1)[1]
    return iri.strip("<>")


# --------------------------------------------------------------------------------------
# OWL API extraction
# --------------------------------------------------------------------------------------

class _Skip(Exception):
    """Raised for an axiom (part) outside DL-Lite_H; the message is the reason."""


class _Trivial(Exception):
    """Raised for an axiom that is a tautology (nothing to translate)."""


def _iri(entity):
    return str(entity.toStringID())


def _role(pe):
    """``(role IRI, inverse)`` of an object property expression."""
    inverse = False
    while pe.isAnonymous():  # ObjectInverseOf(...)
        pe = pe.getInverse()
        inverse = not inverse
    if pe.isOWLTopObjectProperty() or pe.isOWLBottomObjectProperty():
        raise _Skip("top or bottom object property")
    return _iri(pe.asOWLObjectProperty()), inverse


def _basic(ce):
    """A basic concept ``("concept", iri)`` / ``("exists", iri, inv)``, ``"top"`` or
    ``"bottom"``."""
    kind = str(ce.getClassExpressionType())
    if kind == "Class":
        if ce.isOWLThing():
            return "top"
        if ce.isOWLNothing():
            return "bottom"
        return ("concept", _iri(ce.asOWLClass()))
    if kind == "ObjectSomeValuesFrom":
        if not ce.getFiller().isOWLThing():
            raise _Skip("qualified existential restriction")
        return ("exists",) + _role(ce.getProperty())
    raise _Skip(f"{kind} is not a basic concept")


def _rhs(ce):
    """Right-hand side as a list of ``(positive: bool, basic)``; ``[]`` for owl:Thing."""
    kind = str(ce.getClassExpressionType())
    if kind == "ObjectIntersectionOf":
        out = []
        for op in ce.getOperandsAsList():
            out.extend(_rhs(op))
        return out
    if kind == "ObjectComplementOf":
        b = _basic(ce.getOperand())
        if b == "top":
            return [(False, "top")]  # B ⊑ ⊥
        if b == "bottom":
            return []
        return [(False, b)]
    b = _basic(ce)
    if b == "top":
        return []
    if b == "bottom":
        return [(False, "top")]
    return [(True, b)]


def _lhs(ce):
    b = _basic(ce) if str(ce.getClassExpressionType()) in ("Class", "ObjectSomeValuesFrom") \
        else None
    if b is None:
        raise _Skip(f"{ce.getClassExpressionType()} on the left-hand side")
    if b == "top":
        raise _Skip("owl:Thing on the left-hand side")
    if b == "bottom":
        raise _Trivial()
    return b


def _inclusions(sub, sup):
    """Translate ``sub ⊑ sup`` into ``[("pos"|"neg", lhs, rhs)]``."""
    lhs = _lhs(sub)
    out = []
    for positive, rhs in _rhs(sup):
        if rhs == "top":  # B ⊑ ⊥  is  B ⊑ ¬B
            out.append(("neg", lhs, lhs))
        else:
            out.append(("pos" if positive else "neg", lhs, rhs))
    if not out:
        raise _Trivial()
    return out


def _translate(axiom):
    """Translate one OWL axiom into a list of symbolic DL-Lite_H axioms."""
    kind = str(axiom.getAxiomType())
    if kind == "SubClassOf":
        return _inclusions(axiom.getSubClass(), axiom.getSuperClass())
    if kind == "EquivalentClasses":
        ops = list(axiom.getClassExpressionsAsList())
        out = []
        for i, a in enumerate(ops):
            for b in ops[i + 1:]:
                out.extend(_inclusions(a, b))
                out.extend(_inclusions(b, a))
        return out
    if kind == "DisjointClasses":
        ops = [_lhs(c) for c in axiom.getClassExpressionsAsList()]
        return [("neg", a, b) for i, a in enumerate(ops) for b in ops[i + 1:]]
    if kind in ("ObjectPropertyDomain", "ObjectPropertyRange"):
        name, inv = _role(axiom.getProperty())
        inv = inv if kind == "ObjectPropertyDomain" else not inv
        lhs = ("exists", name, inv)
        out = []
        for positive, rhs in _rhs(axiom.getDomain() if kind == "ObjectPropertyDomain"
                                  else axiom.getRange()):
            out.append(("neg", lhs, lhs) if rhs == "top"
                       else ("pos" if positive else "neg", lhs, rhs))
        if not out:
            raise _Trivial()
        return out
    if kind == "SubObjectPropertyOf":
        return [("role", _role(axiom.getSubProperty()), _role(axiom.getSuperProperty()))]
    if kind == "EquivalentObjectProperties":
        ops = [_role(p) for p in axiom.getPropertiesMinusTop()] \
            if hasattr(axiom, "getPropertiesMinusTop") else \
            [_role(p) for p in axiom.getProperties()]
        return [("role", a, b) for a in ops for b in ops if a != b]
    if kind == "InverseObjectProperties":
        (r, ri), (s, si) = _role(axiom.getFirstProperty()), _role(axiom.getSecondProperty())
        return [("role", (r, ri), (s, not si)), ("role", (s, si), (r, not ri))]
    if kind == "SymmetricObjectProperty":
        r, ri = _role(axiom.getProperty())
        return [("role", (r, ri), (r, not ri))]
    if kind == "ClassAssertion":
        ind = axiom.getIndividual()
        if ind.isAnonymous():
            raise _Skip("anonymous individual")
        out = []
        for positive, rhs in _rhs(axiom.getClassExpression()):
            if not positive:
                raise _Skip("negative concept assertion")
            out.append(("cassert", rhs, _iri(ind)))
        if not out:
            raise _Trivial()
        return out
    if kind == "ObjectPropertyAssertion":
        a, b = axiom.getSubject(), axiom.getObject()
        if a.isAnonymous() or b.isAnonymous():
            raise _Skip("anonymous individual")
        name, inv = _role(axiom.getProperty())
        a, b = _iri(a), _iri(b)
        return [("rassert", name, b, a) if inv else ("rassert", name, a, b)]
    raise _Skip(f"axiom type {kind}")


def extract_dllite(ontology, signature=None, include_imports=True):
    """Extract the :math:`\\text{DL-Lite}_{\\mathcal{H}}` part of an OWL ontology.

    :param ontology: An ``org.semanticweb.owlapi.model.OWLOntology`` (or a list of them,
        whose axioms are merged).
    :param signature: A :class:`DLLiteKB` whose concept, role and individual lists are
        reused (for validation and test ontologies that must share the training indices).
        Axioms mentioning other entities are counted as skipped (``"entity not in
        signature"``). If ``None``, the signature is that of the ontology (``owl:Thing``,
        ``owl:Nothing`` and the top/bottom properties excluded), sorted by IRI.
    :param include_imports: Also read the imports closure.
    :rtype: :class:`DLLiteKB`
    """
    from mowl.owlapi import Imports
    imports = Imports.INCLUDED if include_imports else Imports.EXCLUDED
    ontologies = ontology if isinstance(ontology, (list, tuple)) else [ontology]

    if signature is None:
        concepts, roles, individuals = set(), set(), set()
        for ont in ontologies:
            for c in ont.getClassesInSignature(imports):
                if not (c.isOWLThing() or c.isOWLNothing()):
                    concepts.add(_iri(c))
            for p in ont.getObjectPropertiesInSignature(imports):
                if not (p.isOWLTopObjectProperty() or p.isOWLBottomObjectProperty()):
                    roles.add(_iri(p))
            for i in ont.getIndividualsInSignature(imports):
                if not i.isAnonymous():
                    individuals.add(_iri(i))
        concepts, roles, individuals = sorted(concepts), sorted(roles), sorted(individuals)
    else:
        concepts, roles = list(signature.concepts), list(signature.roles)
        individuals = list(signature.individuals)

    cidx = {c: i for i, c in enumerate(concepts)}
    ridx = {r: i for i, r in enumerate(roles)}
    iidx = {a: i for i, a in enumerate(individuals)}
    nc, nr = len(concepts), len(roles)

    def b_index(b):
        if b[0] == "concept":
            if b[1] not in cidx:
                raise _Skip("entity not in signature")
            return cidx[b[1]]
        if b[1] not in ridx:
            raise _Skip("entity not in signature")
        return nc + int(b[2]) * nr + ridx[b[1]]

    def r_index(name):
        if name not in ridx:
            raise _Skip("entity not in signature")
        return ridx[name]

    def i_index(name):
        if name not in iidx:
            raise _Skip("entity not in signature")
        return iidx[name]

    rows = {"pos": [], "neg": [], "role": [], "cassert": [], "rassert": []}
    skipped, trivial, ignored = Counter(), Counter(), Counter()
    for ont in ontologies:
        for axiom in ont.getAxioms(imports):
            kind = str(axiom.getAxiomType())
            if kind in _NON_LOGICAL or not axiom.isLogicalAxiom():
                ignored[kind] += 1
                continue
            try:
                translated = []
                for item in _translate(axiom):
                    tag = item[0]
                    if tag in ("pos", "neg"):
                        translated.append((tag, (b_index(item[1]), b_index(item[2]))))
                    elif tag == "role":
                        (r, ri), (s, si) = item[1], item[2]
                        translated.append((tag, (r_index(r), int(ri), r_index(s), int(si))))
                    elif tag == "cassert":
                        translated.append((tag, (b_index(item[1]), i_index(item[2]))))
                    else:
                        translated.append(
                            (tag, (r_index(item[1]), i_index(item[2]), i_index(item[3]))))
            except _Trivial:
                trivial[kind] += 1
                continue
            except _Skip as reason:
                skipped[f"{kind}: {reason}"] += 1
                continue
            for tag, row in translated:
                rows[tag].append(row)

    def arr(items, cols):
        items = list(dict.fromkeys(items))  # de-duplicate, keep order
        return np.asarray(items, dtype=np.int64).reshape(-1, cols)

    return DLLiteKB(
        concepts, roles, individuals,
        concept_inclusions=arr(rows["pos"], 2), negative_inclusions=arr(rows["neg"], 2),
        role_inclusions=arr(rows["role"], 4), concept_assertions=arr(rows["cassert"], 2),
        role_assertions=arr(rows["rassert"], 3),
        skipped=dict(skipped), trivial=dict(trivial), ignored=dict(ignored))
