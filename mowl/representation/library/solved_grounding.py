"""Grounding: the part that turns axioms into constraints over a finite domain.

A solved or grounded method does not read an axiom as a loss on regions of the whole space;
it *grounds* the axiom: every axiom of the ontology becomes a constraint on the parameters
of finitely many things, the elements of a finite domain (the classes, the individuals),
and a constraint is either **hard** (a linear inequality a solver must satisfy: BoxLitE) or
**soft** (a hinge whose violation is minimised: GroundedEL). This module holds what the two
share and what any other grounded method needs:

:class:`Domain`
    The finite domain: one element per class followed by one per individual (the numbering of
    GroundedEL; a class id is also its element id).
:class:`AxiomTable`
    The axioms of an ontology as integer rows by form, checked against a domain: the ground
    instances. A form is the name of :data:`mowl.representation.FORMS` or
    :data:`~mowl.representation.ROLE_FORMS`.
:func:`inside_bounds`, :func:`inside_complement_bounds`, :func:`member_bounds`
    The *hard* reading of inclusion between boxes given by lower and upper bounds (as
    expressions of a :class:`~mowl.representation.library.solved_solver.Program`'s algebra):
    :math:`A \\subseteq B` iff :math:`L_B \\le L_A` and :math:`U_A \\le U_B`; every
    constraint is linear.

Interface summary (for unifying with other constructions)::

    Domain(classes, individuals) -> size, class_element(ids), individual_element(ids)
    AxiomTable(domain, {form: rows}).instances(form) -> LongTensor / ndarray of rows
    inside_bounds(alg, sub, sup, name)  with sub, sup = (lower, upper) expressions
"""

from dataclasses import dataclass, field

import numpy as np

from mowl.representation.objectives import FORMS, ROLE_FORMS

__all__ = ["Domain", "AxiomTable", "inside_bounds", "inside_complement_bounds",
           "member_bounds"]


@dataclass(frozen=True)
class Domain:
    """A finite domain: the elements ``0 .. nb_classes - 1`` are the classes, the next
    ``nb_individuals`` the individuals."""

    nb_classes: int
    nb_individuals: int = 0
    nb_relations: int = 0

    @property
    def size(self):
        return self.nb_classes + self.nb_individuals

    def class_element(self, ids):
        """The element of the classes ``ids`` (the identity)."""
        return ids

    def individual_element(self, ids):
        """The element of the individuals ``ids``."""
        return ids + self.nb_classes

    def is_individual(self, element):
        return element >= self.nb_classes


@dataclass
class AxiomTable:
    """The ground instances of an ontology over a domain: ``rows[form]`` is an integer
    array (or tensor) with one row per axiom, in the column layout of the form. Rows are
    checked against the domain (class, relation and individual ids in range)."""

    domain: Domain
    rows: dict = field(default_factory=dict)

    #: Columns of every form that hold ``"c"`` class ids, ``"r"`` relation ids and ``"i"``
    #: individual ids.
    COLUMNS = {
        "gci0": "cc", "gci1": "ccc", "gci2": "crc", "gci3": "rcc", "gci0_bot": "c",
        "gci1_bot": "cc", "gci3_bot": "rc", "class_assertion": "ic",
        "object_property_assertion": "iri", "role_inclusion": "rr", "role_chain": "rrr"}

    def __post_init__(self):
        for form, rows in self.rows.items():
            if form not in FORMS + ROLE_FORMS:
                raise ValueError(f"Unknown axiom form {form!r}.")
            layout = self.COLUMNS[form]
            array = np.asarray(rows)
            if not array.size:
                continue
            if array.ndim != 2 or array.shape[1] < len(layout):
                raise ValueError(f"{form}: rows need at least {len(layout)} columns.")
            bound = {"c": self.domain.nb_classes, "r": self.domain.nb_relations,
                     "i": self.domain.nb_individuals}
            for column, kind in enumerate(layout):
                ids = array[:, column]
                if ids.min() < 0 or ids.max() >= bound[kind]:
                    raise ValueError(f"{form}: column {column} holds an id outside the "
                                     f"{ {'c': 'classes', 'r': 'relations', 'i': 'individuals'}[kind]} "
                                     f"of the domain.")

    def instances(self, form):
        """The rows of ``form`` (an empty list if the ontology has none)."""
        return self.rows.get(form, [])

    def forms(self):
        """The forms with at least one ground instance."""
        return [f for f, r in self.rows.items() if len(r)]

    def count(self):
        return {f: len(r) for f, r in self.rows.items()}


def inside_bounds(alg, sub, sup, name="inclusion"):
    """Hard constraints of :math:`A \\subseteq B` for boxes ``sub = (L_A, U_A)`` and
    ``sup = (L_B, U_B)`` (arrays or expressions of equal shape)."""
    alg.le(sup[0], sub[0], name)
    alg.le(sub[1], sup[1], name)


def inside_complement_bounds(alg, sub, other, s_omega, name="negative inclusion"):
    """Hard constraints of :math:`A \\subseteq \\neg B` in the convex complement box
    :math:`[-s - L_B, s - U_B]` of BoxLitE."""
    alg.le(-s_omega - other[0], sub[0], name)
    alg.le(sub[1], s_omega - other[1], name)


def member_bounds(alg, point, box, margin=0.0, name="membership"):
    """Hard constraints of a point in the box :math:`\\{L + m \\le x \\le U - m\\}`."""
    alg.le(box[0] + margin, point, name)
    alg.le(point, box[1] - margin, name)
