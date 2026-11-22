"""The direct al-cone constructions (Özçep, Leemhuis & Wolter, JAIR 2023), kept for
comparison (see :mod:`mowl.baselines`).

:class:`ALConeInterpretation` here is the interpretation class of :mod:`mowl.baselines.geometric.core.cones`
with the two constructions as they were written for the paper: elements of a finite model
on half-axes (:meth:`~ALConeInterpretation.from_finite_model`) and the atoms of a Boolean
TBox on half-axes (:meth:`~ALConeInterpretation.from_boolean_ontology`, JAIR Props. 7, 8).
In :mod:`mowl.baselines.geometric.core.cones` both are combinations of a source (``finite model``,
``canonical model``, ``boolean atoms``) and the ``half-axes`` reading.
"""
import itertools
from typing import Iterable, Optional

import numpy as np

from mowl.baselines.geometric.core import cones
from mowl.baselines.geometric.core.cones import (ALCAxiom, ALCone, ALConeChecker, ConceptAssertion,
                                  FiniteInterpretation, GCI, RoleAssertion, _double,
                                  _doubled_point, _eval_boolean, halfaxis_vector, has_roles,
                                  signature)
from mowl.baselines.geometric.core.errors import (InconsistentOntology, InternalConsistencyError,
                                   UnsupportedAxiom)

__all__ = ["ALConeInterpretation"]


class ALConeInterpretation(cones.ALConeInterpretation):
    """An al-cone interpretation with the direct constructions."""

    @classmethod
    def from_finite_model(cls, model: FiniteInterpretation,
                          axioms: Optional[Iterable[ALCAxiom]] = None
                          ) -> "ALConeInterpretation":
        """Build an al-cone model from a finite classical interpretation (our
        construction; see the module docstring).

        Domain element ``k`` is placed on half-axis ``k``; if the domain size is odd, the
        remaining half-axis carries a copy of element ``0`` (same concept memberships,
        same successors, same predecessors), which preserves the truth of every
        :math:`\\mathcal{ALC}` concept. Concept names become the conic hulls of the
        half-axes of their extensions, roles the incidence matrices of their extensions,
        and individual ``a`` the unit vector on the half-axis of :math:`a^I`.

        :param model: A :class:`FiniteInterpretation`.
        :param axioms: If given, the classical model is first checked to satisfy them
            (:class:`ValueError` otherwise), and the result is verified with
            :class:`ALConeChecker` (:class:`InternalConsistencyError` otherwise).
        """
        axioms = None if axioms is None else list(axioms)
        if axioms is not None:
            bad = model.violations(axioms)
            if bad:
                listed = "; ".join(map(str, bad[:5]))
                raise ValueError(
                    f"The finite interpretation is not a model of the axioms: {listed}")
        size = model.domain_size
        if size < 1:
            raise ValueError("The domain must not be empty.")
        dim = (size + 1) // 2
        f = list(range(size)) + [0] * (2 * dim - size)  # half-axis -> element
        concepts = {}
        for name, ext in model.concepts.items():
            concepts[name] = ALCone(np.array([f[h] in ext for h in range(2 * dim)]))
        roles = {}
        for name, pairs in model.roles.items():
            mat = np.zeros((2 * dim, 2 * dim), dtype=bool)
            for i in range(2 * dim):
                for j in range(2 * dim):
                    if (f[i], f[j]) in pairs:
                        mat[j, i] = True
            roles[name] = mat
        inds = {a: halfaxis_vector(dim, k) for a, k in model.individuals.items()}
        interp = cls(dim, concepts, roles, inds, halfaxis_elements=f)
        if axioms is not None:
            bad = ALConeChecker(interp).violations(axioms)
            if bad:
                raise InternalConsistencyError(
                    "al-cone construction violates: " + "; ".join(map(str, bad[:5])))
        return interp

    @classmethod
    def from_boolean_ontology(cls, axioms: Iterable[ALCAxiom], faithful: bool = False,
                              names: Optional[Iterable[str]] = None,
                              max_names: int = 16) -> "ALConeInterpretation":
        """Al-cone model of a Boolean :math:`\\mathcal{ALC}` ontology (no roles) by the
        construction of JAIR Prop. 7 (``faithful=False``) or Prop. 8 (``faithful=True``).

        The algebraic atoms of the Lindenbaum-Tarski algebra of the TBox are the
        complete conjunctions of literals over the concept names that satisfy every GCI;
        they are enumerated exhaustively and placed on half-axes (one atom is doubled if
        their number is odd). A concept name is the conic hull of the half-axes of the
        atoms below it. The result is TBox-faithful: :math:`C \\sqsubseteq D` holds iff
        it is entailed by the TBox.

        Individuals: with ``faithful=False`` each individual is put on the half-axis of
        one atom consistent with its assertions (Prop. 7, Example 1). With
        ``faithful=True`` every coordinate is doubled (Prop. 8) and the individual is put
        on the point whose support is the set of half-axes of *all* atoms consistent
        with its assertions; then :math:`a^I \\in C^I` iff :math:`\\mathcal{O} \\models
        C(a)` for every Boolean concept :math:`C` (strong concept-faithfulness).

        :param axioms: GCIs and concept assertions without roles.
        :param faithful: Use the doubled, strongly concept-faithful model of Prop. 8.
        :param names: Extra concept names to include in the signature.
        :param max_names: Refuse signatures larger than this (the atoms are enumerated).
        :raises UnsupportedAxiom: If an axiom contains a role.
        :raises InconsistentOntology: If the TBox or an individual's assertions are
            unsatisfiable.
        """
        axioms = list(axioms)
        tbox, abox = [], {}
        for ax in axioms:
            if isinstance(ax, RoleAssertion) or \
                    (isinstance(ax, GCI) and (has_roles(ax.sub) or has_roles(ax.sup))) or \
                    (isinstance(ax, ConceptAssertion) and has_roles(ax.concept)):
                raise UnsupportedAxiom(f"Not a Boolean ALC axiom: {ax}")
            if isinstance(ax, GCI):
                tbox.append(ax)
            else:
                abox.setdefault(ax.individual, []).append(ax.concept)
        sig, _, _ = signature(axioms)
        sig = sorted(sig | set(names or ()))
        if len(sig) > max_names:
            raise ValueError(f"{len(sig)} concept names exceed max_names={max_names}.")

        def holds(c, val):
            return _eval_boolean(c, val)

        types = []
        for bits in itertools.product((False, True), repeat=len(sig)):
            val = dict(zip(sig, bits))
            if all((not holds(g.sub, val)) or holds(g.sup, val) for g in tbox):
                types.append(val)
        if not types:
            raise InconsistentOntology("The Boolean TBox is unsatisfiable (no atom).")
        m = len(types)
        dim = max(1, (m + 1) // 2)
        f = list(range(m)) + [0] * (2 * dim - m)  # half-axis -> atom
        concepts = {a: ALCone(np.array([types[f[h]][a] for h in range(2 * dim)]))
                    for a in sig}
        compat = {}
        for ind, cs in abox.items():
            ok = [t for t in range(m) if all(holds(c, types[t]) for c in cs)]
            if not ok:
                raise InconsistentOntology(f"Assertions on {ind!r} are unsatisfiable.")
            compat[ind] = ok
        if not faithful:
            inds = {a: halfaxis_vector(dim, ok[0]) for a, ok in compat.items()}
            interp = cls(dim, concepts, {}, inds, halfaxis_elements=f)
        else:
            concepts = {a: _double(c) for a, c in concepts.items()}
            inds = {}
            for a, ok in compat.items():
                sup = [h for h in range(2 * dim) if f[h] in ok]
                inds[a] = _doubled_point(dim, sup)
            interp = cls(2 * dim, concepts, {}, inds,
                         halfaxis_elements=[f[2 * (h // 4) + h % 2]
                                            for h in range(4 * dim)])
        bad = ALConeChecker(interp).violations(axioms)
        if bad:
            raise InternalConsistencyError(
                "Boolean al-cone construction violates: " + "; ".join(map(str, bad[:5])))
        return interp
