"""Numerical BoxLitE embeddings (box interpretations, Definition 3 of Lourenço et al., KR
2026) and their scores (Section 5.2). Plain NumPy; no solver needed."""

from dataclasses import dataclass

import numpy as np


def signed_distance_to_nonpositive_orthant(y):
    """Signed distance of each row of ``y`` to :math:`\\mathbb{R}^n_-` (Proposition 3):
    :math:`\\|y^+\\|_2` if some component is positive, otherwise :math:`\\max_i y_i`.

    :param y: Array of shape ``(..., n)``.
    :return: Array of shape ``(...)``.
    """
    y = np.asarray(y, dtype=float)
    pos = np.maximum(y, 0.0)
    outside = (y > 0).any(axis=-1)
    return np.where(outside, np.sqrt((pos ** 2).sum(axis=-1)), y.max(axis=-1))


def box_distance(lower, upper, x, margin):
    """``dist(B, x)`` of Section 5.2 for a box with bounds ``lower``, ``upper`` (the set
    :math:`\\{x \\mid L + m \\le x \\le U - m\\}`, ``m = margin``): the signed distance of
    :math:`(L + m - x) \\oplus (x - U + m)` to :math:`\\mathbb{R}^{2d}_-`. Broadcasts over
    leading axes."""
    lower, upper, x = np.broadcast_arrays(np.asarray(lower, float), np.asarray(upper, float),
                                          np.asarray(x, float))
    y = np.concatenate([lower + margin - x, x - upper + margin], axis=-1)
    return signed_distance_to_nonpositive_orthant(y)


@dataclass
class BoxLitEEmbedding:
    """A BoxLitE box interpretation (Definition 3).

    Boxes are stored by their parameters :math:`L, U`. In the paper's semantics
    (``margin = epsilon``) the box is the set :math:`\\{x \\mid L + \\epsilon \\le x \\le
    U - \\epsilon\\}` (Definition 2); in the reference code's semantics (``margin = 0``) it
    is the closed box :math:`[L, U]`.

    :param pos: ``(|N_I|, d)`` positions of individuals.
    :param bump: ``(|N_I|, d)`` bumps of individuals.
    :param concept_lower: ``(|N_C|, d)``.
    :param concept_upper: ``(|N_C|, d)``.
    :param head_lower: ``(|N_R|, d)``; likewise ``head_upper``, ``tail_lower``,
        ``tail_upper``, ``bump_lower``, ``bump_upper``.
    :param s_omega: Half-width :math:`s_\\Omega` of the universe box.
    :param margin: ``epsilon`` (paper) or ``0`` (reference implementation).
    """

    pos: np.ndarray
    bump: np.ndarray
    concept_lower: np.ndarray
    concept_upper: np.ndarray
    head_lower: np.ndarray
    head_upper: np.ndarray
    tail_lower: np.ndarray
    tail_upper: np.ndarray
    bump_lower: np.ndarray
    bump_upper: np.ndarray
    s_omega: float = 1.0
    margin: float = 0.01

    @property
    def dim(self):
        return self.pos.shape[1] if self.pos.ndim == 2 else self.head_lower.shape[1]

    @property
    def n_concepts(self):
        return self.concept_lower.shape[0]

    @property
    def n_roles(self):
        return self.head_lower.shape[0]

    # ----------------------------------------------------------------------------------
    # Boxes of basic concepts, complements and roles (Definition 3, Section 5.1)
    # ----------------------------------------------------------------------------------

    def basic_bounds(self):
        """``(L, U)`` of all basic concepts, shape ``(|N_C| + 2|N_R|, d)``:
        concept names, then :math:`\\exists R` (:math:`L^H - U^B, U^H - L^B`), then
        :math:`\\exists R^-` (:math:`L^T - U^B, U^T - L^B`)."""
        lower = np.concatenate([self.concept_lower, self.head_lower - self.bump_upper,
                                self.tail_lower - self.bump_upper], axis=0)
        upper = np.concatenate([self.concept_upper, self.head_upper - self.bump_lower,
                                self.tail_upper - self.bump_lower], axis=0)
        return lower, upper

    def complement_bounds(self, lower, upper):
        """Bounds of the complement box :math:`\\overline{\\eta(C)}`:
        :math:`(-s_\\Omega - L, s_\\Omega - U)`."""
        return -self.s_omega - lower, self.s_omega - upper

    def role_boxes(self, role, inverse=False):
        """``((L^H, U^H), (L^T, U^T), (L^B, U^B))`` of :math:`R` or :math:`R^-` (head and
        tail swapped)."""
        head = (self.head_lower[role], self.head_upper[role])
        tail = (self.tail_lower[role], self.tail_upper[role])
        if inverse:
            head, tail = tail, head
        return head, tail, (self.bump_lower[role], self.bump_upper[role])

    # ----------------------------------------------------------------------------------
    # Scores (Section 5.2): s = -L_concept, s = -L_role
    # ----------------------------------------------------------------------------------

    def concept_loss(self, basic, individuals=None):
        """:math:`\\mathcal{L}_{concept}(B, a)` for one basic concept and the given
        individuals (all if ``None``)."""
        lower, upper = self.basic_bounds()
        pos = self.pos if individuals is None else self.pos[np.asarray(individuals)]
        return box_distance(lower[basic], upper[basic], pos, self.margin)

    def concept_scores(self, basic, individuals=None):
        """:math:`s(B, a) = -\\mathcal{L}_{concept}(B, a)`."""
        return -self.concept_loss(basic, individuals)

    def role_loss(self, role, subjects, objects):
        """:math:`\\mathcal{L}_{role}(R, a, b)` for paired (or broadcast) index arrays."""
        a, b = np.asarray(subjects), np.asarray(objects)
        m = self.margin
        head = box_distance(self.head_lower[role], self.head_upper[role],
                            self.pos[a] + self.bump[b], m)
        tail = box_distance(self.tail_lower[role], self.tail_upper[role],
                            self.pos[b] + self.bump[a], m)
        bump_a = box_distance(self.bump_lower[role], self.bump_upper[role], self.bump[a], m)
        bump_b = box_distance(self.bump_lower[role], self.bump_upper[role], self.bump[b], m)
        return np.maximum(np.maximum(head, tail), np.maximum(bump_a, bump_b))

    def role_scores(self, role, subject=None, obj=None):
        """Scores :math:`s(R, a, b) = -\\mathcal{L}_{role}(R, a, b)`.

        Exactly one of ``subject`` and ``obj`` is given: the scores of all individuals as
        objects (``R(subject, ?)``) or as subjects (``R(?, obj)``).
        """
        n = self.pos.shape[0]
        everyone = np.arange(n)
        if (subject is None) == (obj is None):
            raise ValueError("Give exactly one of subject and obj.")
        if obj is None:
            return -self.role_loss(role, np.full(n, subject), everyone)
        return -self.role_loss(role, everyone, np.full(n, obj))

    # ----------------------------------------------------------------------------------
    # I/O
    # ----------------------------------------------------------------------------------

    _ARRAYS = ("pos", "bump", "concept_lower", "concept_upper", "head_lower", "head_upper",
               "tail_lower", "tail_upper", "bump_lower", "bump_upper")

    def save(self, path):
        """Write the embedding to a ``.npz`` file."""
        np.savez(path, s_omega=self.s_omega, margin=self.margin,
                 **{k: getattr(self, k) for k in self._ARRAYS})

    @classmethod
    def load(cls, path):
        """Read an embedding written by :meth:`save`."""
        with np.load(path) as data:
            return cls(**{k: data[k] for k in cls._ARRAYS},
                       s_omega=float(data["s_omega"]), margin=float(data["margin"]))
