"""Helly-satisfiability check and non-box-representability certificates.

Implements the Helly-satisfiability diagnostic and minimal jointly-unsatisfiable
concept-name set enumeration of Spec §5.3 (G4).

What a certificate proves (Spec §5.3):
- By Helly's theorem for axis-parallel boxes in :math:`\\mathbb{R}^d` (Helly
  number 2; Eckhoff 1988; Leemhuis & Kutz, NeSy 2025, Def. 3, Prop. 4), any
  finite family of axis-parallel boxes whose members pairwise intersect has a
  non-empty common intersection.
- An ontology :math:`\\mathcal{O}` is *Helly-satisfiable* if it has a model in
  which the family of extensions of all definable concepts has Helly's property
  (Leemhuis & Kutz, NeSy 2025, Def. 6). Every box interpretation has Helly's
  property (Prop. 4), so non-Helly-satisfiability implies that no box model
  exists in any dimension (Prop. 5).
- A :class:`HellyCertificate` is a triple of concepts :math:`(C_1, C_2, C_3)`,
  each a conjunction of concept names, such that
  :math:`\\mathcal{O} \\models C_1 \\sqcap C_2 \\sqcap C_3 \\sqsubseteq \\bot` while
  each pairwise conjunction :math:`C_1 \\sqcap C_2`, :math:`C_1 \\sqcap C_3`, and
  :math:`C_2 \\sqcap C_3` is satisfiable wrt :math:`\\mathcal{O}`. In any box
  model of :math:`\\mathcal{O}` in which those three pairwise conjunctions are
  non-empty, the three boxes :math:`\\eta(C_i)` pairwise intersect and therefore
  share a common point, contradicting
  :math:`\\mathcal{O} \\models C_1 \\sqcap C_2 \\sqcap C_3 \\sqsubseteq \\bot`.
  Consequently, a certificate proves: *no box model of* :math:`\\mathcal{O}`
  *exists in which* :math:`C_1 \\sqcap C_2`, :math:`C_2 \\sqcap C_3`, *and*
  :math:`C_1 \\sqcap C_3` *are all non-empty*.
- Two certificate classes are reported in :class:`HellyReport`:
  - ``certificates_nondegenerate``: certificates whose three pairwise
    conjunctions :math:`C_1 \\sqcap C_2`, :math:`C_1 \\sqcap C_3`, and
    :math:`C_2 \\sqcap C_3` are each :math:`\\mathcal{O}`-equivalent to a
    satisfiable concept name (recorded in ``pairwise_names``). Because every
    non-degenerate model must assign a non-empty region to each satisfiable
    concept name, such a certificate excludes *every non-degenerate box model*
    of :math:`\\mathcal{O}` in any dimension (as in :math:`T_3`, where
    :math:`X_1 \\sqcap X_2 \\equiv A`, :math:`X_2 \\sqcap X_3 \\equiv B`,
    :math:`X_1 \\sqcap X_3 \\equiv C`). Note that :math:`T_3` without the names
    :math:`A, B, C` is Helly-satisfiable in the sense of Leemhuis & Kutz (2025)
    because a model need not realise the pairwise conjunctions unless they are
    named or forced by strong faithfulness.
  - ``certificates_strong``: all certificates found. They exclude *strongly
    faithful box models* (in which every satisfiable conjunction of concept
    names is non-empty).

Propositional exactness lemma and completeness (Spec §5.3):
- **Lemma.** In the propositional fragment (no :class:`~mowl.geometric.normal_forms.NF3`,
  :class:`~mowl.geometric.normal_forms.NF4`,
  :class:`~mowl.geometric.normal_forms.NF6`, or
  :class:`~mowl.geometric.normal_forms.NF7` role axioms), let :math:`\\mathcal{U}`
  be the family of inclusion-minimal sets of concept names that are jointly
  unsatisfiable (:math:`\\mathcal{O} \\models \\sqcap U \\sqsubseteq \\bot`). There
  is a ``certificates_strong`` certificate if and only if some
  :math:`U \\in \\mathcal{U}` has :math:`|U| \\ge 3`: for
  :math:`U = \\{u_1, \\dots, u_k\\}` with :math:`k \\ge 3`, the triple
  :math:`(\\{u_1\\}, \\{u_2\\}, U \\setminus \\{u_1, u_2\\})` has satisfiable
  pairwise conjunctions (proper subsets of :math:`U`) and unsatisfiable triple
  conjunction :math:`\\sqcap U`.
- ``HellyReport.complete`` is ``True`` **if and only if** the enumeration was
  not cut off by ``k_max`` or ``max_certificates`` **and** the ontology has no
  role axioms (``has_roles == False``). The guarantee that the check is exact
  for the propositional fragment holds only when ``complete=True``.
- **Roles caveat:** When the ontology contains role axioms
  (:class:`~mowl.geometric.normal_forms.NF3`,
  :class:`~mowl.geometric.normal_forms.NF4`,
  :class:`~mowl.geometric.normal_forms.NF6`, or
  :class:`~mowl.geometric.normal_forms.NF7`), ``has_roles`` is ``True`` and
  ``complete`` is always ``False``: as shown by Leemhuis & Kutz (NAI 2026,
  Proposition 38), in the presence of roles even their ABox-closure Helly test
  is necessary but not sufficient for box representability. Any certificate
  returned when ``has_roles=True`` remains a sound proof of
  non-box-representability.

References:
- Leemhuis & Kutz, *Wait, That's Not a Box! On the Box-Representability of
  Description Logic Ontologies*, NeSy 2025 (PMLR 284:303–321), Definitions 3, 6;
  Propositions 4, 5.
- Leemhuis & Kutz, NAI 2026, Proposition 38 (incompleteness of ABox-closure
  Helly tests in the presence of roles).
- Eckhoff, *Intersection properties of boxes. Part I: An upper-bound theorem*,
  Israel Journal of Mathematics 62(3):283–301, 1988 (Helly number 2 for
  axis-parallel boxes in :math:`\\mathbb{R}^d`).
"""
from __future__ import annotations

from dataclasses import dataclass
import itertools
from typing import TYPE_CHECKING, Iterator, List, Optional, Tuple

from mowl.geometric.normal_forms import NF3, NF4, NF6, NF7, TOP

if TYPE_CHECKING:  # pragma: no cover
    from mowl.geometric.completion import CanonicalModel


@dataclass(frozen=True)
class HellyCertificate:
    """Certificate proving non-existence of a box model realising three pairwise conjunctions.

    A certificate is a triple :math:`(C_1, C_2, C_3)` of non-empty concept-name
    tuples (representing the conjunctions :math:`\\sqcap C_i`) such that
    :math:`\\mathcal{O} \\models C_1 \\sqcap C_2 \\sqcap C_3 \\sqsubseteq \\bot` while
    each pairwise conjunction :math:`C_1 \\sqcap C_2`, :math:`C_1 \\sqcap C_3`, and
    :math:`C_2 \\sqcap C_3` is satisfiable wrt :math:`\\mathcal{O}`. By the Helly
    property of axis-parallel boxes (Helly number 2; Eckhoff 1988; Leemhuis &
    Kutz, NeSy 2025, Defs. 3, 6; Props. 4, 5), no box model of :math:`\\mathcal{O}`
    can assign non-empty regions to all three pairwise conjunctions
    :math:`C_1 \\sqcap C_2`, :math:`C_1 \\sqcap C_3`, and :math:`C_2 \\sqcap C_3`.

    Attributes
    ----------
    c1 : Tuple[str, ...]
        First concept-name conjunction :math:`C_1 = (u_1,)`.
    c2 : Tuple[str, ...]
        Second concept-name conjunction :math:`C_2 = (u_2,)`.
    c3 : Tuple[str, ...]
        Third concept-name conjunction :math:`C_3 = (u_3, \\dots, u_s)`.
    pairwise_names : Tuple[Optional[str], Optional[str], Optional[str]]
        Tuple ``(p12, p13, p23)`` where ``p_ij`` is a satisfiable concept name
        :math:`N` with :math:`\\mathcal{O} \\models N \\equiv \\sqcap(C_i \\cup C_j)`
        for :math:`(i, j) \\in ((1, 2), (1, 3), (2, 3))`, or ``None`` if no such
        concept name exists in :math:`\\mathcal{O}`. The certificate is
        *non-degenerate* (excluding all non-degenerate box models) iff all three
        entries of ``pairwise_names`` are not ``None``.
    """

    c1: Tuple[str, ...]
    c2: Tuple[str, ...]
    c3: Tuple[str, ...]
    pairwise_names: Tuple[Optional[str], Optional[str], Optional[str]]


@dataclass
class HellyReport:
    """Result of :func:`helly_check` on a :class:`~mowl.geometric.completion.CanonicalModel`.

    ``complete=True`` holds if and only if the minimal unsatisfiable set
    enumeration was not cut off by ``k_max`` or ``max_certificates`` and the
    ontology has no role axioms (:class:`~mowl.geometric.normal_forms.NF3`,
    :class:`~mowl.geometric.normal_forms.NF4`,
    :class:`~mowl.geometric.normal_forms.NF6`,
    :class:`~mowl.geometric.normal_forms.NF7`). With roles, ``complete`` is
    always ``False`` (Leemhuis & Kutz, NAI 2026, Prop. 38), though any returned
    certificate remains a sound proof of non-box-representability (Eckhoff 1988;
    Leemhuis & Kutz, NeSy 2025, Defs. 3, 6; Props. 4, 5).

    Attributes
    ----------
    certificates_strong : List[HellyCertificate]
        All certificates found (each excludes strongly faithful box models).
    certificates_nondegenerate : List[HellyCertificate]
        Subset of ``certificates_strong`` whose ``pairwise_names`` contain no
        ``None`` (each excludes every non-degenerate box model).
    complete : bool
        ``True`` iff enumeration was not cut off by ``k_max`` or
        ``max_certificates`` and ``not has_roles``.
    has_roles : bool
        ``True`` iff ``cm.axioms`` contains any ``NF3``, ``NF4``, ``NF6``, or
        ``NF7`` role axiom.
    k_max : int
        Maximum minimal unsatisfiable set size inspected.
    """

    certificates_strong: List[HellyCertificate]
    certificates_nondegenerate: List[HellyCertificate]
    complete: bool
    has_roles: bool
    k_max: int

    @property
    def box_representable_nondegenerate(self) -> Optional[bool]:
        """Return ``False`` if non-degenerate certificates exist, ``None`` if ``not complete``, else ``True``."""
        if self.certificates_nondegenerate:
            return False
        if not self.complete:
            return None
        return True

    @property
    def box_representable_strong(self) -> Optional[bool]:
        """Return ``False`` if strong certificates exist, ``None`` if ``not complete``, else ``True``."""
        if self.certificates_strong:
            return False
        if not self.complete:
            return None
        return True


def _find_equivalent_name(
    cm: CanonicalModel, candidates: List[str], conj: Tuple[str, ...]
) -> Optional[str]:
    """Return the first satisfiable name ``N`` in ``candidates`` with ``N ≡ ⨅conj``, or ``None``."""
    for name in candidates:
        if all(cm.entails(name, x) for x in conj) and cm.entails_conjunction(
            conj, name
        ):
            return name
    return None


def _make_certificate(
    cm: CanonicalModel, candidates: List[str], u_set: Tuple[str, ...]
) -> HellyCertificate:
    """Build a :class:`HellyCertificate` for a sorted minimal unsatisfiable set ``u_set`` of size >= 3."""
    c1 = (u_set[0],)
    c2 = (u_set[1],)
    c3 = tuple(u_set[2:])
    p12 = _find_equivalent_name(cm, candidates, c1 + c2)
    p13 = _find_equivalent_name(cm, candidates, c1 + c3)
    p23 = _find_equivalent_name(cm, candidates, c2 + c3)
    return HellyCertificate(
        c1=c1,
        c2=c2,
        c3=c3,
        pairwise_names=(p12, p13, p23),
    )


def _iter_next_candidates(
    frontier: List[Tuple[str, ...]], s: int
) -> Iterator[Tuple[str, ...]]:
    """Yield sorted ``s``-tuples whose all ``(s - 1)``-subsets belong to ``frontier``."""
    if len(frontier) < s:
        return
    frontier_set = set(frontier)
    prefix_len = s - 2
    for prefix, group in itertools.groupby(frontier, key=lambda t: t[:-1]):
        lasts = [t[-1] for t in group]
        m = len(lasts)
        for i in range(m):
            u = lasts[i]
            for j in range(i + 1, m):
                cand = prefix + (u, lasts[j])
                if all(
                    (cand[:idx] + cand[idx + 1:]) in frontier_set
                    for idx in range(prefix_len)
                ):
                    yield cand


def helly_check(
    cm: CanonicalModel, k_max: int = 4, max_certificates: int = 10
) -> HellyReport:
    """Run the Helly-satisfiability check on ``cm`` and return a :class:`HellyReport`.

    Enumerates inclusion-minimal jointly-unsatisfiable sets of satisfiable
    concept names (``cm.names`` minus ``TOP``) of size :math:`3 \\dots k_{\\max}`
    using levelwise frontier construction over satisfiable :math:`(s - 1)`-subsets.
    For each minimal unsatisfiable set :math:`U = (u_1, \\dots, u_s)` (:math:`s \\ge 3`),
    emits ``HellyCertificate(c1=(u1,), c2=(u2,), c3=tuple(U[2:]), pairwise_names=(p12, p13, p23))``,
    where ``p_ij`` is a satisfiable concept name :math:`N` satisfying
    :math:`\\mathcal{O} \\models N \\equiv \\sqcap(C_i \\cup C_j)` (checked via
    ``cm.entails(N, x)`` for all :math:`x \\in C_i \\cup C_j` and
    ``cm.entails_conjunction(C_i | C_j, N)``), or ``None`` if no such name exists.

    What a certificate proves:
    - Each certificate proves that no axis-parallel box model of :math:`\\mathcal{O}`
      exists in which the three pairwise conjunctions :math:`C_1 \\sqcap C_2`,
      :math:`C_1 \\sqcap C_3`, and :math:`C_2 \\sqcap C_3` are all non-empty
      (Eckhoff 1988; Leemhuis & Kutz, NeSy 2025, Defs. 3, 6; Props. 4, 5).
    - A certificate in ``certificates_nondegenerate`` (all three ``pairwise_names``
      not ``None``) proves that no *non-degenerate* box model of :math:`\\mathcal{O}`
      exists in any dimension.
    - A certificate in ``certificates_strong`` proves that no *strongly faithful*
      box model of :math:`\\mathcal{O}` exists in any dimension.
    - ``complete=True`` holds if and only if the enumeration was not cut off by
      ``k_max`` or ``max_certificates`` and the ontology has no role axioms
      (:class:`~mowl.geometric.normal_forms.NF3`,
      :class:`~mowl.geometric.normal_forms.NF4`,
      :class:`~mowl.geometric.normal_forms.NF6`,
      :class:`~mowl.geometric.normal_forms.NF7`). When role axioms are present,
      ``complete`` is ``False`` (Leemhuis & Kutz, NAI 2026, Prop. 38).
    """
    has_roles = any(isinstance(ax, (NF3, NF4, NF6, NF7)) for ax in cm.axioms)
    candidates = [name for name in cm.names if name != TOP]
    exts = {name: cm.extension(name) for name in candidates}

    frontier: List[Tuple[str, ...]] = []
    n_cand = len(candidates)
    for i in range(n_cand):
        a = candidates[i]
        ext_a = exts[a]
        for j in range(i + 1, n_cand):
            b = candidates[j]
            if not ext_a.isdisjoint(exts[b]) or cm.joint_satisfiable((a, b)):
                frontier.append((a, b))

    certificates_strong: List[HellyCertificate] = []
    certificates_nondegenerate: List[HellyCertificate] = []
    cut_off = False

    s = 3
    while True:
        next_cands_iter = _iter_next_candidates(frontier, s)
        first_cand = next(next_cands_iter, None)
        if first_cand is None:
            break
        if s > k_max or max_certificates <= 0:
            cut_off = True
            break

        next_frontier: List[Tuple[str, ...]] = []
        for cand in itertools.chain((first_cand,), next_cands_iter):
            common = exts[cand[0]]
            for x in cand[1:]:
                common = common & exts[x]
                if not common:
                    break
            if common or cm.joint_satisfiable(cand):
                next_frontier.append(cand)
            else:
                cert = _make_certificate(cm, candidates, cand)
                certificates_strong.append(cert)
                if all(p is not None for p in cert.pairwise_names):
                    certificates_nondegenerate.append(cert)
                if len(certificates_strong) >= max_certificates:
                    cut_off = True
                    break

        if cut_off:
            break
        frontier = next_frontier
        s += 1

    complete = (not cut_off) and (not has_roles)
    return HellyReport(
        certificates_strong=certificates_strong,
        certificates_nondegenerate=certificates_nondegenerate,
        complete=complete,
        has_roles=has_roles,
        k_max=k_max,
    )
