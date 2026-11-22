"""Independent model checker for BoxLitE embeddings.

The checker reads a solved :class:`~mowl.baselines.models.boxlite.embedding.BoxLitEEmbedding` and a
:class:`~mowl.datasets.dllite.DLLiteKB` and decides, with set semantics only (no loss
function, no solver), whether the box interpretation

* is well formed (Definition 3: widths in :math:`[0, 2 s_\\Omega]`, individuals in
  :math:`\\Omega`),
* is box consistent (:math:`\\eta(C) \\cap \\overline{\\eta(C)} = \\emptyset` for every
  basic concept, Section 3), and
* satisfies every axiom of the KB (Definition 4): role assertions (head, tail and bump
  memberships), concept assertions, concept inclusions :math:`\\eta(B) \\subseteq
  \\eta(C)`, negative inclusions :math:`\\eta(B) \\subseteq \\overline{\\eta(C)}` and role
  inclusions (head, tail and bump boxes).

A box with parameters :math:`L, U` is the set :math:`\\{x \\mid L + m \\le x \\le U - m\\}`
with ``m`` the embedding's margin (:math:`\\epsilon` in the paper, 0 in the reference
code). It is empty if :math:`L + m > U - m` in some dimension; an empty box is included in
every box. By Proposition 2 of the paper, a well-formed, box-consistent model of
:math:`\\mathcal{K}` is weakly KB faithful.
"""

from dataclasses import dataclass, field

import numpy as np


class _Box:
    """The set ``{lo <= x <= hi}`` (componentwise)."""

    def __init__(self, lo, hi):
        self.lo, self.hi = np.asarray(lo, float), np.asarray(hi, float)

    def empty(self, tol):
        """Certainly empty: some dimension with lo > hi + tol."""
        return bool((self.lo - self.hi > tol).any())

    def contains_point(self, x, tol):
        return bool((self.lo - tol <= x).all() and (x <= self.hi + tol).all())

    def subset_of(self, other, tol):
        if self.empty(tol):
            return True
        return bool((other.lo <= self.lo + tol).all() and (self.hi <= other.hi + tol).all())

    def disjoint_from(self, other, tol):
        """Certainly disjoint: empty, or separated by more than ``tol`` in a dimension."""
        if self.empty(tol) or other.empty(tol):
            return True
        return bool((np.maximum(self.lo, other.lo) - np.minimum(self.hi, other.hi) > tol).any())


@dataclass
class ModelCheck:
    """Result of :func:`check_model`.

    :param violations: ``{category: [description, ...]}`` for every failed check.
    :param checked: ``{category: number of checks}``.
    """

    violations: dict = field(default_factory=dict)
    checked: dict = field(default_factory=dict)

    @property
    def well_formed(self):
        return not any(self.violations.get(k) for k in ("width", "universe"))

    @property
    def box_consistent(self):
        return not self.violations.get("box_consistency")

    @property
    def is_model(self):
        """All TBox and ABox axioms are satisfied (Definition 4)."""
        return not any(self.violations.get(k) for k in (
            "concept_inclusion", "negative_inclusion", "role_inclusion",
            "concept_assertion", "role_assertion"))

    @property
    def weakly_kb_faithful(self):
        """Well formed, box consistent and a model: weakly KB faithful by Proposition 2."""
        return self.well_formed and self.box_consistent and self.is_model

    def summary(self):
        return {k: (self.checked.get(k, 0), len(self.violations.get(k, ())))
                for k in sorted(set(self.checked) | set(self.violations))}


def check_model(embedding, kb, tol=1e-6, max_reports=20):
    """Check that ``embedding`` is a box-consistent model of ``kb``.

    :param embedding: A :class:`~mowl.baselines.models.boxlite.embedding.BoxLitEEmbedding`.
    :param kb: A :class:`~mowl.datasets.dllite.DLLiteKB` with the same signature.
    :param tol: Absolute tolerance for solver round-off.
    :param max_reports: Maximum number of violations recorded per category (all are
        counted in ``checked``; ``is_model`` is exact either way).
    :rtype: :class:`ModelCheck`
    """
    e, m, s = embedding, embedding.margin, embedding.s_omega
    result = ModelCheck()

    def report(category, text):
        bucket = result.violations.setdefault(category, [])
        if len(bucket) < max_reports:
            bucket.append(text)
        elif len(bucket) == max_reports:
            bucket.append("...")

    def count(category, n=1):
        result.checked[category] = result.checked.get(category, 0) + n

    def box(lower, upper):
        return _Box(lower + m, upper - m)

    nc = len(kb.concepts)

    # Definition 3: widths and universe box.
    for name, lo, up in (("concept", e.concept_lower, e.concept_upper),
                         ("head", e.head_lower, e.head_upper),
                         ("tail", e.tail_lower, e.tail_upper),
                         ("bump", e.bump_lower, e.bump_upper)):
        width = up - lo
        count("width", width.shape[0])
        for i in np.where(((width < -tol) | (width > 2 * s + tol)).any(axis=1))[0]:
            report("width", f"{name} box {i}: width outside [0, 2 s_omega]")
    for name, x in (("pos", e.pos), ("bump", e.bump)):
        count("universe", x.shape[0])
        for i in np.where((np.abs(x) > s + tol).any(axis=1))[0]:
            report("universe", f"{name}({kb.individuals[i]}) outside the universe box")

    # Basic concepts as raw bounds; boxes and complement boxes.
    lower = np.concatenate([e.concept_lower, e.head_lower - e.bump_upper,
                            e.tail_lower - e.bump_upper], axis=0)
    upper = np.concatenate([e.concept_upper, e.head_upper - e.bump_lower,
                            e.tail_upper - e.bump_lower], axis=0)

    def basic(b):
        return box(lower[b], upper[b])

    def complement(b):
        return box(-s - lower[b], s - upper[b])

    def name(b):
        return kb.basic_concept_name(b)

    # Box consistency.
    for b in range(kb.n_basic):
        count("box_consistency")
        if not basic(b).disjoint_from(complement(b), tol):
            report("box_consistency", f"η({name(b)}) meets its complement box")

    # TBox.
    for sub, sup in kb.concept_inclusions:
        count("concept_inclusion")
        if not basic(sub).subset_of(basic(sup), tol):
            report("concept_inclusion", f"{name(sub)} ⊑ {name(sup)}")
    for sub, other in kb.negative_inclusions:
        count("negative_inclusion")
        if not basic(sub).subset_of(complement(other), tol):
            report("negative_inclusion", f"{name(sub)} ⊑ ¬{name(other)}")

    def role_boxes(r, inv):
        head = box(e.head_lower[r], e.head_upper[r])
        tail = box(e.tail_lower[r], e.tail_upper[r])
        if inv:
            head, tail = tail, head
        return head, tail, box(e.bump_lower[r], e.bump_upper[r])

    def role_name(r, inv):
        return kb.basic_concept_name(nc + r)[1:] + ("⁻" if inv else "")

    for r, ri, t, ti in kb.role_inclusions:
        count("role_inclusion")
        sub, sup = role_boxes(r, ri), role_boxes(t, ti)
        if not all(x.subset_of(y, tol) for x, y in zip(sub, sup)):
            report("role_inclusion", f"{role_name(r, ri)} ⊑ {role_name(t, ti)}")

    # ABox.
    for b, a in kb.concept_assertions:
        count("concept_assertion")
        if not basic(b).contains_point(e.pos[a], tol):
            report("concept_assertion", f"{name(b)}({kb.individuals[a]})")
    for r, a, b in kb.role_assertions:
        count("role_assertion")
        head, tail, bump = role_boxes(r, False)
        ok = all((head.contains_point(e.pos[a] + e.bump[b], tol),
                  tail.contains_point(e.pos[b] + e.bump[a], tol),
                  bump.contains_point(e.bump[a], tol), bump.contains_point(e.bump[b], tol)))
        if not ok:
            report("role_assertion",
                   f"{role_name(r, False)}({kb.individuals[a]}, {kb.individuals[b]})")
    return result
