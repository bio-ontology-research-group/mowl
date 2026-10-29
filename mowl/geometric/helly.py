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
    named or required to be non-empty (conjunction faithfulness).
  - ``certificates_strong``: all certificates found. They exclude
    *conjunction-faithful* box models, i.e. box models in which every
    satisfiable conjunction of concept names is non-empty. This is stronger
    than strong faithfulness on names, so a strong certificate does **not**
    exclude box models that are strongly faithful on names: for
    :math:`\\{X_1 \\sqcap X_2 \\sqsubseteq Y, Y \\sqcap X_3 \\sqsubseteq \\bot\\}`,
    :math:`(X_1, X_2, X_3)` is a strong certificate, yet the 1-D boxes
    :math:`X_1 = [0, 1], X_2 = [2, 3], X_3 = [4, 5], Y = [6, 7]` form a
    non-degenerate box model that is strongly faithful on names.
- **What** ``box_representable_* = True`` **means.**
  :attr:`HellyReport.box_representable_nondegenerate` and
  :attr:`HellyReport.box_representable_strong` being ``True`` means "no Helly
  obstruction of this kind was found" (obstruction-free), **not** that a box
  model exists; no sufficiency theorem is known. Use
  :class:`~mowl.geometric.boxes.BoxModelBuilder` to search for a model.

Propositional exactness lemma, relevance pruning, and completeness (Spec §5.3):
- **Lemma (strong reading only).** In the propositional fragment (no
  :class:`~mowl.geometric.normal_forms.NF3`,
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
- **Soundness vs. completeness for non-degenerate certificates (Spec §5.3 v3).**
  The Lemma is complete for *strong* certificates only. Non-degenerate
  certificates are a sound sufficient condition for non-box-representability,
  not a complete one: whether a certificate is non-degenerate depends on how a
  minimal set is split into :math:`(C_1, C_2, C_3)`, and the union of names in a
  non-degenerate certificate need not itself be an inclusion-minimal
  unsatisfiable set. Consequently,
  :attr:`HellyReport.box_representable_nondegenerate` returns ``False`` when
  ``certificates_nondegenerate`` is non-empty, ``True`` only when ``complete``
  is ``True`` and ``certificates_strong`` is empty (since every non-degenerate
  certificate is also a strong certificate), and ``None`` otherwise.
- **Relevance pruning (Spec §5.3 v3).** Only :math:`\\bot`-relevant concept names
  are enumerated as candidates, where relevance is the backward closure from the
  :math:`\\bot`-axioms through :class:`~mowl.geometric.normal_forms.NF1` and
  :class:`~mowl.geometric.normal_forms.NF2` (a name is relevant if it is a
  left-hand name ``sub1``/``sub2`` of an ``NF2`` with ``sup == BOTTOM`` or the
  ``sub`` of an ``NF1`` with ``sup == BOTTOM``, or the ``sub`` of an ``NF1`` /
  ``sub1`` or ``sub2`` of an ``NF2`` whose ``sup`` is relevant). In the
  propositional fragment every derivation of :math:`Z \\sqsubseteq \\bot` uses
  only :math:`\\bot`-relevant names, so every inclusion-minimal jointly
  unsatisfiable set consists entirely of :math:`\\bot`-relevant names. If the
  ontology contains no :math:`\\bot`-axiom (no ``NF1``, ``NF2``, or ``NF4`` with
  ``sup == BOTTOM``), :math:`\\mathcal{U} = \\emptyset` and :func:`helly_check`
  returns immediately with no certificates and ``complete = (not has_roles)``.
- ``HellyReport.complete`` is ``True`` **if and only if** the enumeration was
  not cut off by ``k_max`` or ``max_certificates`` **and** the ontology has no
  role axioms (``has_roles == False``). The guarantee that the check is exact
  for the propositional fragment holds only when ``complete=True``, and **only
  for the strong reading** (``certificates_strong`` /
  :attr:`HellyReport.box_representable_strong`): it then decides exactly
  whether a Helly obstruction to conjunction-faithful box models exists.
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
- **Scalability note:** The check re-saturates the ontology per candidate set
  (in :meth:`~mowl.geometric.completion.CanonicalModel.joint_satisfiable`, for
  candidate sets whose canonical extensions are disjoint) and is intended for
  small to medium ontologies. Incremental joint satisfiability is future work.

References:
- M. Leemhuis & O. Kutz, *Understanding the Expressive Capabilities of
  Knowledge Base Embeddings under Box Semantics*, NeSy 2025 (PMLR
  284:303–321), Definitions 3, 6; Propositions 4, 5.
- M. Leemhuis & O. Kutz, *Trustworthy Knowledge Base Embeddings: A
  Foundational Study of Box Semantics*, Neurosymbolic Artificial Intelligence
  (NAI), 2026, Proposition 38 (incompleteness of ABox-closure Helly tests in
  the presence of roles).
- Eckhoff, *Intersection properties of boxes. Part I: An upper-bound theorem*,
  Israel Journal of Mathematics 62(3):283–301, 1988 (Helly number 2 for
  axis-parallel boxes in :math:`\\mathbb{R}^d`).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import itertools
from typing import (
    TYPE_CHECKING,
    Dict,
    FrozenSet,
    Iterable,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
)

from mowl.geometric.normal_forms import (
    Axiom,
    BOTTOM,
    NF1,
    NF2,
    NF3,
    NF4,
    NF6,
    NF7,
    TOP,
)

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
    :class:`~mowl.geometric.normal_forms.NF7`). When ``complete=True``, the
    check is exact for the **strong reading** only
    (:attr:`box_representable_strong` / ``certificates_strong``); non-degenerate
    certificates are a sound sufficient condition for non-box-representability,
    not a complete one (Spec §5.3 v3). With roles, ``complete`` is always
    ``False`` (Leemhuis & Kutz, NAI 2026, Prop. 38), though any returned
    certificate remains a sound proof of non-box-representability (Eckhoff 1988;
    Leemhuis & Kutz, NeSy 2025, Defs. 3, 6; Props. 4, 5) of its kind:
    non-degenerate certificates exclude every non-degenerate box model, strong
    certificates exclude conjunction-faithful box models. ``True`` from
    :attr:`box_representable_nondegenerate` or :attr:`box_representable_strong`
    means "no Helly obstruction of this kind was found", not that a box model
    exists.

    Attributes
    ----------
    certificates_strong : List[HellyCertificate]
        All certificates found. Each excludes *conjunction-faithful* box models
        (every satisfiable conjunction of concept names non-empty); it does not
        exclude box models that are merely strongly faithful on names.
    certificates_nondegenerate : List[HellyCertificate]
        Subset of ``certificates_strong`` whose ``pairwise_names`` contain no
        ``None`` (each is a sound sufficient certificate excluding every
        non-degenerate box model).
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
        """Return ``False`` if non-degenerate certificates exist, ``True`` only if ``complete`` and ``certificates_strong`` is empty, else ``None``.

        ``False`` proves that no non-degenerate box model exists in any
        dimension. ``True`` means only that no Helly obstruction to
        non-degenerate box models was found (obstruction-free), not that a box
        model exists.
        """
        if self.certificates_nondegenerate:
            return False
        if self.complete and not self.certificates_strong:
            return True
        return None

    @property
    def box_representable_strong(self) -> Optional[bool]:
        """Return ``False`` if strong certificates exist, ``None`` if ``not complete``, else ``True``.

        ``False`` proves that no *conjunction-faithful* box model (every
        satisfiable conjunction of concept names non-empty) exists; it does not
        exclude box models that are strongly faithful on names. ``True`` means
        only that no Helly obstruction of this kind was found
        (obstruction-free), not that a box model exists.
        """
        if self.certificates_strong:
            return False
        if not self.complete:
            return None
        return True


def _bottom_relevant_names(axioms: Iterable[Axiom]) -> Set[str]:
    """Compute the backward closure of :math:`\\bot`-relevant concept names through ``NF1`` and ``NF2``.

    A concept name is :math:`\\bot`-relevant if it is ``sub1`` or ``sub2`` of an
    :class:`~mowl.geometric.normal_forms.NF2` with ``sup == BOTTOM`` (or ``sub``
    of an :class:`~mowl.geometric.normal_forms.NF1` with ``sup == BOTTOM``), or
    if it is the ``sub`` of an ``NF1`` or ``sub1``/``sub2`` of an ``NF2`` whose
    ``sup`` is relevant, iterated to a fixpoint.
    """
    relevant: Set[str] = set()
    for ax in axioms:
        if isinstance(ax, NF2) and ax.sup == BOTTOM:
            relevant.add(ax.sub1)
            relevant.add(ax.sub2)
        elif isinstance(ax, NF1) and ax.sup == BOTTOM:
            relevant.add(ax.sub)
    if not relevant:
        return relevant

    changed = True
    while changed:
        changed = False
        for ax in axioms:
            if isinstance(ax, NF1) and ax.sup in relevant:
                if ax.sub not in relevant:
                    relevant.add(ax.sub)
                    changed = True
            elif isinstance(ax, NF2) and ax.sup in relevant:
                if ax.sub1 not in relevant:
                    relevant.add(ax.sub1)
                    changed = True
                if ax.sub2 not in relevant:
                    relevant.add(ax.sub2)
                    changed = True
    return relevant


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

    Enumerates inclusion-minimal jointly-unsatisfiable sets of :math:`\\bot`-relevant
    satisfiable concept names (``relevant & set(cm.names)`` minus ``TOP``) of size
    :math:`3 \\dots k_{\\max}` using levelwise frontier construction over
    satisfiable :math:`(s - 1)`-subsets. In the propositional fragment every
    derivation of :math:`Z \\sqsubseteq \\bot` uses only :math:`\\bot`-relevant
    names (backward closure from :math:`\\bot`-axioms through ``NF1`` and ``NF2``),
    so restricting candidates to :math:`\\bot`-relevant names preserves soundness
    and completeness of minimal unsatisfiable set enumeration (Spec §5.3 v3).
    If the ontology contains no :math:`\\bot`-axiom (no ``NF1``, ``NF2``, or
    ``NF4`` with ``sup == BOTTOM``), returns immediately with no certificates
    and ``complete = (not has_roles)``.

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
      not ``None``) is a sound sufficient condition proving that no
      *non-degenerate* box model of :math:`\\mathcal{O}` exists in any dimension,
      though not a complete one (whether a certificate is non-degenerate depends
      on the split of :math:`U` into :math:`(C_1, C_2, C_3)` and the union of a
      non-degenerate certificate need not be minimal; Spec §5.3 v3).
    - A certificate in ``certificates_strong`` proves that no
      *conjunction-faithful* box model of :math:`\\mathcal{O}` (one in which
      every satisfiable conjunction of concept names is non-empty) exists in
      any dimension; it does not exclude box models that are strongly faithful
      on names. When ``complete=True``, this check is exact for the strong
      reading in the propositional fragment.
    - ``box_representable_* = True`` on the returned report means "no Helly
      obstruction of this kind was found", not that a box model exists.
    - The check re-saturates per candidate set and is intended for small to
      medium ontologies; incremental joint satisfiability is future work.
    - ``complete=True`` holds if and only if the enumeration was not cut off by
      ``k_max`` or ``max_certificates`` and the ontology has no role axioms
      (:class:`~mowl.geometric.normal_forms.NF3`,
      :class:`~mowl.geometric.normal_forms.NF4`,
      :class:`~mowl.geometric.normal_forms.NF6`,
      :class:`~mowl.geometric.normal_forms.NF7`). When role axioms are present,
      ``complete`` is ``False`` (Leemhuis & Kutz, NAI 2026, Prop. 38).
    """
    has_roles = any(isinstance(ax, (NF3, NF4, NF6, NF7)) for ax in cm.axioms)
    has_bottom_axiom = any(
        isinstance(ax, (NF1, NF2, NF4)) and ax.sup == BOTTOM for ax in cm.axioms
    )
    if not has_bottom_axiom:
        return HellyReport(
            certificates_strong=[],
            certificates_nondegenerate=[],
            complete=not has_roles,
            has_roles=has_roles,
            k_max=k_max,
        )

    relevant = _bottom_relevant_names(cm.axioms)
    candidates = [name for name in cm.names if name != TOP and name in relevant]
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


def certificate_triangles(
    cm: CanonicalModel, *, max_triangles: Optional[int] = None
) -> List[HellyCertificate]:
    """Find non-degenerate singleton Helly certificates via triangle enumeration on direct ``NF2`` definitions.

    Constructs the undirected graph of concept names in which an edge
    ``{A, B}`` labelled ``N`` exists whenever ``cm.axioms`` contains an
    :class:`~mowl.geometric.normal_forms.NF2` axiom ``A ⊓ B ⊑ N`` with
    ``A != B``, ``N != BOTTOM``, ``A, B, N != TOP``, all three satisfiable in
    ``cm``, and ``cm.entails(N, A) and cm.entails(N, B)`` (so
    :math:`\\mathcal{O} \\models N \\equiv A \\sqcap B`). Enumerates triangles
    ``(X1, X2, X3)`` with ``X1 < X2 < X3`` lexicographically and retains those
    whose pairwise conjunctions ``{X1, X2}``, ``{X1, X3}``, ``{X2, X3}`` are
    jointly satisfiable and whose triple ``{X1, X2, X3}`` is jointly
    unsatisfiable.

    Guarantees
    ----------
    - Every returned :class:`HellyCertificate` is a sound non-degenerate
      certificate with ``c1=(X1,)``, ``c2=(X2,)``, ``c3=(X3,)``,
      ``X1 < X2 < X3`` lexicographically, and ``pairwise_names=(N12, N13, N23)``
      all non-``None`` and satisfiable, excluding every non-degenerate
      axis-parallel box model of :math:`\\mathcal{O}` in any dimension.
    - Returns every singleton non-degenerate certificate whose three pairwise
      conjunctions are direct :class:`~mowl.geometric.normal_forms.NF2`
      definitions in ``cm.axioms`` (up to ``max_triangles`` if specified).
    - The returned list is sorted lexicographically by ``(c1, c2, c3)``.
    - Triangle enumeration over the named-conjunction graph takes
      :math:`O(\\sum_v \\deg(v)^2)` time in the vertex degrees of that graph
      (plus the joint-satisfiability checks on candidate triangles).

    Non-guarantees
    --------------
    - Not complete for all non-degenerate or strong Helly certificates:
      pairwise conjunctions defined through chains of equivalences or
      multi-step normalisations without a direct ``NF2(A, B, N)`` axiom, as
      well as higher-arity certificates where :math:`|C_i| > 1`, are not found.

    Parameters
    ----------
    cm : CanonicalModel
        Canonical model of the normalised ontology.
    max_triangles : int, optional
        If given, stop after collecting ``max_triangles`` certificates
        (returns ``[]`` if ``max_triangles <= 0``).

    Returns
    ------
    List[HellyCertificate]
        Sorted list of non-degenerate singleton certificates.
    """
    if max_triangles is not None and max_triangles <= 0:
        return []

    edge_map: Dict[FrozenSet[str], str] = {}
    for ax in cm.axioms:
        if not isinstance(ax, NF2):
            continue
        a, b, n = ax.sub1, ax.sub2, ax.sup
        if a == b or n == BOTTOM or TOP in (a, b, n):
            continue
        if (
            cm.is_satisfiable(a)
            and cm.is_satisfiable(b)
            and cm.is_satisfiable(n)
            and cm.entails(n, a)
            and cm.entails(n, b)
        ):
            key: FrozenSet[str] = frozenset((a, b))
            if key not in edge_map or n < edge_map[key]:
                edge_map[key] = n

    adj: Dict[str, Set[str]] = defaultdict(set)
    for pair in edge_map:
        u, v = tuple(pair)
        adj[u].add(v)
        adj[v].add(u)

    triangles: List[HellyCertificate] = []
    for a in sorted(adj.keys()):
        higher = sorted(v for v in adj[a] if v > a)
        m = len(higher)
        for i in range(m):
            b = higher[i]
            adj_b = adj[b]
            for j in range(i + 1, m):
                c = higher[j]
                if c not in adj_b:
                    continue
                if (
                    cm.joint_satisfiable((a, b))
                    and cm.joint_satisfiable((a, c))
                    and cm.joint_satisfiable((b, c))
                    and not cm.joint_satisfiable((a, b, c))
                ):
                    n12 = edge_map[frozenset((a, b))]
                    n13 = edge_map[frozenset((a, c))]
                    n23 = edge_map[frozenset((b, c))]
                    triangles.append(
                        HellyCertificate(
                            c1=(a,),
                            c2=(b,),
                            c3=(c,),
                            pairwise_names=(n12, n13, n23),
                        )
                    )
                    if max_triangles is not None and len(triangles) >= max_triangles:
                        return triangles

    return triangles

