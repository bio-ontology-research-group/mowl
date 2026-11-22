"""BoxLitE (Lourenço et al., KR 2026) as a combination of parts that is fitted by a solver.

The method is the combination

========================  ======================================================================
class model               :class:`BoundsBoxes`: a box per concept name, given by its bounds
relation model            :class:`HeadTailBumpBoxes`: a head, a tail and a bump box per role
individual model          :class:`~mowl.representation.library.shapes_classes.Points` with a
                          bump: a position and a bump vector per individual
measure                   :class:`SignedDistance`: the signed distance of a point to a box
objective                 :class:`BoxLitEObjective`: assertions are signed distances, the
                          inclusions are inclusions of bounds
regularisers              :class:`NegativeConceptPenalty`, :class:`WidthPenalty` (``lambda1``,
                          ``lambda2``, ``lambda3``)
fit                       the program of :class:`BoxLitEProgram` through a solver
                          (:class:`~mowl.representation.library.solved_solver.SolverFit`)
========================  ======================================================================

and nothing else: the geometry (:class:`BoxLitEGeometry`) is a
:class:`~mowl.representation.ComposedGeometry` and answers inclusion, negation (the convex
complement box), existential restriction of DL-Lite and membership as any other geometry.
The formulas of the bounds of :math:`\\exists R`, of the signed distances and of the
regularisers are written once against an algebra
(:mod:`~mowl.representation.library.solved_solver`) and run on tensors (the geometry), on
arrays (the embedding, the checks) and as CVXPY expressions (the program).

Two variants of the programme are available (the option ``variant`` of the builder):

``"paper"``
    The programme as printed (Sections 3 and 5): boxes are :math:`\\{L + \\epsilon \\le x \\le
    U - \\epsilon\\}`, individuals satisfy :math:`-s_\\Omega + \\epsilon \\le x \\le s_\\Omega -
    \\epsilon`, box consistency is :math:`L_C[i_C] + U_C[i_C] \\le -s_\\Omega`, the objective takes the
    maximum of per-assertion losses, the width regulariser is :math:`\\sum \\|U - L\\|_2` per box,
    and the negatives are all pairs :math:`(D, a)` not witnessed by the ABox
    (:func:`negative_concept_assertions`).
``"reference"``
    What the authors' code (AleksVap/BoxLitE, commit 7d53063) solves, which produced Table 2:
    closed boxes, one signed distance of the stacked residuals of *all* assertions, an absent
    assertion term counted as 0, the width regulariser of CVXPY's ``mixed_norm`` of the width
    vector reshaped row-major, and the negatives of the code.
"""

from dataclasses import dataclass

import numpy as np
import torch as th
import torch.nn as nn

from mowl.representation.class_models import Classes
from mowl.representation.geometry import ComposedGeometry
from mowl.representation.library.other_graph import Related
from mowl.representation.library.shapes_classes import Points
from mowl.representation.library.shapes_regions import Point
from mowl.representation.library.solved_grounding import (inside_bounds,
                                                           inside_complement_bounds)
from mowl.representation.library.solved_solver import (NumpyAlgebra, Program, TorchAlgebra,
                                                       CompiledProblem, CvxpySolver,
                                                       SolveResult, SolverFit)
from mowl.representation.measures import Bounds
from mowl.representation.methods import register_method
from mowl.representation.objectives import Objective
from mowl.representation.regions import Box, ConstructorUnavailable, CornerBox
from mowl.representation.relation_models import Relations

__all__ = ["VARIANTS", "BoundsBoxes", "HeadTailBumpBoxes", "SignedDistance",
           "BoxLitEObjective", "NegativeConceptPenalty", "WidthPenalty", "BoxLitEProgram",
           "BoxLitEGeometry", "BoxLitEEmbedding", "Universe", "boxlite",
           "negative_concept_assertions", "kb_faithful_dimension", "tbox_faithful_dimension",
           "box_distance", "role_distance", "signed_distance_to_nonpositive_orthant",
           "BoxLitEProgramResult"]

#: Variants of the programme (see the module docstring).
VARIANTS = ("paper", "reference")
_NUMPY = NumpyAlgebra()
_TORCH = TorchAlgebra()

#: Names of the variables of the programme (the tables of the parts) and where they live.
CONCEPT_TABLES = ("concept_lower", "concept_upper")
ROLE_TABLES = ("head_lower", "head_upper", "tail_lower", "tail_upper", "bump_lower",
               "bump_upper")
INDIVIDUAL_TABLES = ("pos", "bump")


# -- formulas shared by the tensors, the arrays and the program --------------------------------
def signed_distance_to_nonpositive_orthant(y):
    """Signed distance of each row of ``y`` to :math:`\\mathbb{R}^n_-` (Proposition 3):
    :math:`\\|y^+\\|_2` if some component is positive, otherwise :math:`\\max_i y_i`."""
    return _NUMPY.sdist_rows(np.asarray(y, dtype=float))


def box_distance(lower, upper, x, margin, alg=_NUMPY):
    """``dist(B, x)`` of Section 5.2 for a box with bounds ``lower``, ``upper`` (the set
    :math:`\\{x \\mid L + m \\le x \\le U - m\\}`, ``m = margin``): the signed distance of
    :math:`(L + m - x) \\oplus (x - U + m)` to :math:`\\mathbb{R}^{2d}_-`. Broadcasts over
    leading axes (arrays and tensors; ``alg`` says which)."""
    if alg.name == "numpy":
        lower, upper, x = np.broadcast_arrays(np.asarray(lower, float), np.asarray(upper, float),
                                              np.asarray(x, float))
    return alg.sdist_rows(alg.columns([lower + margin - x, x - upper + margin]))


def role_distance(boxes, pos_a, bump_a, pos_b, bump_b, margin, alg=_NUMPY):
    """The role loss :math:`\\mathcal{L}_{role}(R, a, b)`: the largest of four distances
    (head, tail, and the two bumps in the bump box); ``boxes`` is ``((L^H, U^H), (L^T, U^T),
    (L^B, U^B))`` of the role."""
    (hl, hu), (tl, tu), (bl, bu) = boxes
    head = box_distance(hl, hu, pos_a + bump_b, margin, alg)
    tail = box_distance(tl, tu, pos_b + bump_a, margin, alg)
    da = box_distance(bl, bu, bump_a, margin, alg)
    db = box_distance(bl, bu, bump_b, margin, alg)
    return alg.maximum(alg.maximum(head, tail), alg.maximum(da, db))


def negative_concept_assertions(kb, mode="paper"):
    """The potential concept assertions pushed into complement boxes by the negative
    concept regularisation (Eq. 6), as rows ``[basic concept, individual]``.

    ``"paper"``: all :math:`D(a)` with :math:`D \\in N^\\exists_C` (concept names,
    :math:`\\exists R` and :math:`\\exists R^-`) that are not in the ABox, where a role
    assertion :math:`R(a, b)` counts as asserting :math:`\\exists R(a)` and
    :math:`\\exists R^-(b)`. Consequences of the TBox are not excluded.

    ``"reference"``: as in the authors' code: for a concept name, every individual without
    an assertion of it; for a role name occurring in the ABox, :math:`\\exists R(a)` for
    every individual that is not the subject of an :math:`R` assertion (no
    :math:`\\exists R^-` terms); for a role name not in the ABox, :math:`\\exists R(a)` and
    :math:`\\exists R^-(a)` for every individual.
    """
    n_ind, nb, nc, nr = len(kb.individuals), kb.n_basic, len(kb.concepts), len(kb.roles)
    asserted = np.zeros((nb, n_ind), dtype=bool)
    if len(kb.concept_assertions):
        asserted[kb.concept_assertions[:, 0], kb.concept_assertions[:, 1]] = True
    ra = kb.role_assertions
    if mode == "paper":
        if len(ra):
            asserted[nc + ra[:, 0], ra[:, 1]] = True
            asserted[nc + nr + ra[:, 0], ra[:, 2]] = True
        rows = np.argwhere(~asserted)
    elif mode == "reference":
        if len(ra):
            asserted[nc + ra[:, 0], ra[:, 1]] = True
        in_abox = set(ra[:, 0].tolist()) if len(ra) else set()
        keep = np.ones(nb, dtype=bool)
        for r in in_abox:
            keep[nc + nr + r] = False
        rows = np.argwhere(~asserted & keep[:, None])
    else:
        raise ValueError(f"Unknown negatives mode {mode!r}.")
    return rows.astype(np.int64).reshape(-1, 2)


def kb_faithful_dimension(kb):
    """:math:`d_{min} = |N_C| + |N_R|(2 + |N_I| + 2|N_R|)` of Corollary 2: in this
    dimension every satisfiable KB has a weakly KB-faithful BoxLitE model."""
    nc, nr, ni = len(kb.concepts), len(kb.roles), len(kb.individuals)
    return max(nc + nr * (2 + ni + 2 * nr), kb.n_basic, 1)


def tbox_faithful_dimension(kb):
    """:math:`d_{min} = |N_C| + 3|N_R|` of Corollary 1 (empty ABox)."""
    return max(len(kb.concepts) + 3 * len(kb.roles), kb.n_basic, 1)


# -- the numerical embedding ---------------------------------------------------------------------
@dataclass
class BoxLitEEmbedding:
    """A BoxLitE box interpretation (Definition 3) as arrays: the tables of the parts of a
    solved geometry (:meth:`BoxLitEGeometry.snapshot`), with the scores of Section 5.2.

    Boxes are stored by their parameters :math:`L, U`. In the paper's semantics
    (``margin = epsilon``) the box is the set :math:`\\{x \\mid L + \\epsilon \\le x \\le
    U - \\epsilon\\}`; in the reference code's semantics (``margin = 0``) it is the closed
    box :math:`[L, U]`.

    :param pos: ``(|N_I|, d)`` positions of individuals.
    :param bump: ``(|N_I|, d)`` bumps of individuals.
    :param concept_lower: ``(|N_C|, d)``; likewise ``concept_upper``.
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

    _ARRAYS = ("pos", "bump", "concept_lower", "concept_upper", "head_lower", "head_upper",
               "tail_lower", "tail_upper", "bump_lower", "bump_upper")

    @property
    def dim(self):
        return self.pos.shape[1] if self.pos.ndim == 2 else self.head_lower.shape[1]

    @property
    def n_concepts(self):
        return self.concept_lower.shape[0]

    @property
    def n_roles(self):
        return self.head_lower.shape[0]

    def tables(self):
        """The arrays by the names of the variables of the programme."""
        return {name: getattr(self, name) for name in self._ARRAYS}

    def basic_bounds(self):
        """``(L, U)`` of all basic concepts, shape ``(|N_C| + 2|N_R|, d)``: concept names,
        then :math:`\\exists R`, then :math:`\\exists R^-` (the bounds of the parts:
        :func:`basic_bounds`)."""
        return basic_bounds(_NUMPY, self.tables())

    def complement_bounds(self, lower, upper):
        """Bounds of the complement box: :math:`(-s_\\Omega - L, s_\\Omega - U)`."""
        return -self.s_omega - lower, self.s_omega - upper

    def role_boxes(self, role, inverse=False):
        """``((L^H, U^H), (L^T, U^T), (L^B, U^B))`` of :math:`R` or :math:`R^-` (head and
        tail swapped)."""
        head = (self.head_lower[role], self.head_upper[role])
        tail = (self.tail_lower[role], self.tail_upper[role])
        if inverse:
            head, tail = tail, head
        return head, tail, (self.bump_lower[role], self.bump_upper[role])

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
        return role_distance(self.role_boxes(role), self.pos[a], self.bump[a], self.pos[b],
                             self.bump[b], self.margin)

    def role_scores(self, role, subject=None, obj=None):
        """Scores :math:`s(R, a, b) = -\\mathcal{L}_{role}(R, a, b)`: those of all
        individuals as objects (``R(subject, ?)``) or as subjects (``R(?, obj)``)."""
        n = self.pos.shape[0]
        everyone = np.arange(n)
        if (subject is None) == (obj is None):
            raise ValueError("Give exactly one of subject and obj.")
        if obj is None:
            return -self.role_loss(role, np.full(n, subject), everyone)
        return -self.role_loss(role, everyone, np.full(n, obj))

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


# -- class model -------------------------------------------------------------------------------
class BoundsBoxes(Classes):
    """A class is an axis-parallel box given by its lower and upper bounds :math:`L, U`
    (not by a centre and a half-width: the widths are constrained, and the box may be
    empty). The region is a :class:`~mowl.representation.CornerBox` with exactly these
    corners; the measure applies the margin.

    :param lower, upper: Existing tables to use instead of new ones.

    Part (vocabulary value): ``box``. Used by: ``boxlite``.
    """

    region = CornerBox
    part = "box"
    exact_conjunction = True

    def __init__(self, size=None, embed_dim=None, lower=None, upper=None, scale=0.5):
        super().__init__()
        if lower is None:
            lower = nn.Embedding(size, embed_dim)
            upper = nn.Embedding(size, embed_dim)
            with th.no_grad():
                lower.weight.uniform_(-scale, 0.0)
                upper.weight.uniform_(0.0, scale)
        self.lower, self.upper = lower, upper

    def __len__(self):
        return self.lower.num_embeddings

    @property
    def embed_dim(self):
        return self.lower.embedding_dim

    @property
    def centres(self):
        return None

    def forward(self, ids):
        return CornerBox.from_corners(self.lower(ids), self.upper(ids))

    def all(self):
        return self(th.arange(len(self), device=self.lower.weight.device))

    @staticmethod
    def values(v):
        """``(lower, upper)`` of the concept names from the variables (``None`` if there
        are none)."""
        return v["concept_lower"], v["concept_upper"]


# -- relation model ----------------------------------------------------------------------------
class HeadTailBumpBoxes(Relations):
    """A relation is three boxes (BoxLitE, Definition 3): a *head* box, a *tail* box and a
    *bump* box. :math:`R(a, b)` holds iff :math:`pos(a) + bump(b)` lies in the head box,
    :math:`pos(b) + bump(a)` in the tail box and both bumps in the bump box.
    :math:`\\exists R` is the head box enlarged by the bump box,
    :math:`[L^H - U^B, U^H - L^B]`; :math:`\\exists R^-` is the same from the tail box
    (:meth:`exists_bounds`). The existential restriction is unqualified (DL-Lite): the
    preimage of the universe. Role inclusions are inclusions of the three boxes (with head
    and tail swapped for an inverse role).

    Part (vocabulary value): ``head and tail boxes``. Used by: ``boxlite``.
    """

    part = "head and tail boxes"

    def __init__(self, size=None, embed_dim=None, scale=0.5, **tables):
        super().__init__()
        for name in ROLE_TABLES:
            table = tables.get(name)
            if table is None:
                table = nn.Embedding(size, embed_dim)
                with th.no_grad():
                    if name.endswith("lower"):
                        table.weight.uniform_(-scale, 0.0)
                    else:
                        table.weight.uniform_(0.0, scale)
            setattr(self, name, table)

    def __len__(self):
        return self.head_lower.num_embeddings

    def tables(self):
        return {name: getattr(self, name).weight for name in ROLE_TABLES}

    def boxes(self, ids, inverse=False):
        """``((L^H, U^H), (L^T, U^T), (L^B, U^B))`` of the roles ``ids`` (head and tail
        swapped for the inverse)."""
        head = (self.head_lower(ids), self.head_upper(ids))
        tail = (self.tail_lower(ids), self.tail_upper(ids))
        if inverse:
            head, tail = tail, head
        return head, tail, (self.bump_lower(ids), self.bump_upper(ids))

    @staticmethod
    def exists_bounds(v):
        """``((L, U) of exists R, (L, U) of exists R^-)`` for every role, from the
        variables (``None`` if there are no roles): the head (tail) box enlarged by the bump
        box."""
        if v["head_lower"] is None:
            return None
        return ((v["head_lower"] - v["bump_upper"], v["head_upper"] - v["bump_lower"]),
                (v["tail_lower"] - v["bump_upper"], v["tail_upper"] - v["bump_lower"]))

    def preimage(self, ids, region, inverse=False):
        """The region of :math:`\\exists R` (or :math:`\\exists R^-`): only for the
        universe as the filler."""
        if not isinstance(region, Universe):
            raise ConstructorUnavailable(
                "DL-Lite has unqualified existentials: the filler must be the universe "
                "(geometry.universe()).")
        (hl, hu), (tl, tu), (bl, bu) = self.boxes(ids, inverse)
        return CornerBox.from_corners(hl - bu, hu - bl)

    def pair_distance(self, ids, a, b, margin):
        """The role loss of :math:`R(a, b)` for the relations ``ids`` and the individuals
        ``a`` and ``b`` (points with bumps): ``(B, 1)``."""
        return role_distance(self.boxes(ids), a.centre, a.bump, b.centre, b.bump, margin,
                             _TORCH).reshape(-1, 1)

    def role_inclusion(self, sub, sup, measure=None, margin=0.0):
        """Cost of :math:`R \\sqsubseteq S` for *signed* role ids ``r + n * inverse``
        (the rows of the programme): the violation of the inclusion of the head, tail and
        bump boxes."""
        n = len(self)
        head_l = th.cat([self.head_lower.weight, self.tail_lower.weight])
        head_u = th.cat([self.head_upper.weight, self.tail_upper.weight])
        tail_l = th.cat([self.tail_lower.weight, self.head_lower.weight])
        tail_u = th.cat([self.tail_upper.weight, self.head_upper.weight])
        r, s = sub % n, sup % n
        parts = [th.relu(head_l[sup] - head_l[sub]), th.relu(head_u[sub] - head_u[sup]),
                 th.relu(tail_l[sup] - tail_l[sub]), th.relu(tail_u[sub] - tail_u[sup]),
                 th.relu(self.bump_lower.weight[s] - self.bump_lower.weight[r]),
                 th.relu(self.bump_upper.weight[r] - self.bump_upper.weight[s])]
        return th.linalg.norm(th.cat(parts, dim=1), dim=1, keepdim=True)

    def role_chain(self, first, second, sup, measure=None, margin=0.0):
        raise ConstructorUnavailable("DL-Lite_H has no role chains.")

    @property
    def reads_role_axioms(self):
        return False


@dataclass(frozen=True)
class Universe(Box):
    """The universe box :math:`[-s_\\Omega, s_\\Omega]^d`: the filler of an unqualified
    :math:`\\exists R` of DL-Lite."""


# -- measure -----------------------------------------------------------------------------------
class SignedDistance(Bounds):
    """Boxes compared by their bounds (inclusion, overlap), and a point compared with a
    box by the *signed distance* of the paper: the distance of :math:`(L + m - x) \\oplus
    (x - U + m)` to the non-positive orthant, negative inside the box. Not a cost in the
    sense of the other measures (it is negative where the point lies in the box with room to
    spare): a ranking score (``reading = "score"``).

    Part (vocabulary value): ``exact``. Used by: ``boxlite``.
    """

    accepts = (Box, Point)
    part = "exact"
    reading = "score"

    def member(self, a, b, margin=0.0):
        return box_distance(b.lower, b.upper, a.centre, margin, _TORCH).reshape(-1, 1)


# -- objective ---------------------------------------------------------------------------------
class BoxLitEObjective(Objective):
    """The reading of the axiom forms of BoxLitE.

    :math:`B(a)` is the signed distance of the position to the box (``neg=True``: to the
    complement box, the negative concept loss), :math:`R(a, b)` the largest of the four
    distances of the role, :math:`B_1 \\sqsubseteq B_2` an inclusion of bounds (no margin: the
    constraints of the programme have none) and :math:`R \\sqsubseteq S` an inclusion of the
    three boxes of the roles (rows of signed role ids).

    :param epsilon: The margin :math:`\\epsilon` of the boxes (``0`` for the reference code).
    """

    def __init__(self, epsilon=0.01, **options):
        options.setdefault("negatives", ("class_assertion",))
        super().__init__(margin=epsilon, **options)

    def gci0(self, g, rows, neg=False):
        return g.measure.inside(g.classes(rows[:, 0]), g.classes(rows[:, 1]), 0.0)

    def class_assertion(self, g, rows, neg=False):
        a, c = g.individuals(rows[:, 0]), g.classes(rows[:, 1])
        if self._neg(neg, "class_assertion"):
            c = g.negation(c)
        return g.measure.member(a, c, self._m_of(g))

    def object_property_assertion(self, g, rows, neg=False):
        a, b = g.individuals(rows[:, 0]), g.individuals(rows[:, 2])
        return g.relation_model.pair_distance(rows[:, 1], a, b, self._m_of(g))

    def role_inclusion(self, g, rows, neg=False):
        return g.relation_model.role_inclusion(rows[:, 0], rows[:, 1], g.measure, 0.0)


# -- the program ---------------------------------------------------------------------------------
def basic_bounds(alg, v):
    """``(L, U)`` of all basic concepts, stacked: concept names, then :math:`\\exists R`,
    then :math:`\\exists R^-` (the bounds of the class model and of the relation model)."""
    lower, upper = [], []
    if v["concept_lower"] is not None:
        lo, up = BoundsBoxes.values(v)
        lower.append(lo)
        upper.append(up)
    exists = HeadTailBumpBoxes.exists_bounds(v)
    if exists is not None:
        for lo, up in exists:
            lower.append(lo)
            upper.append(up)
    if not lower:
        return None, None
    return alg.vstack(lower), alg.vstack(upper)


class _Loss:
    """The signed-distance loss of a set of residual blocks, in the reading of the variant
    (``"paper"``: the largest per-assertion distance; ``"reference"``: one distance of all
    stacked residuals)."""

    def __init__(self, alg, variant, margin):
        self.alg, self.variant, self.m = alg, variant, margin

    def residuals(self, lower, upper, x):
        return lower + self.m - x, x - upper + self.m

    def __call__(self, blocks):
        alg = self.alg
        if self.variant == "paper":
            y = alg.columns([alg.vstack([lo for lo, _ in blocks]),
                             alg.vstack([up for _, up in blocks])])
            return alg.amax(alg.sdist_rows(y))
        return alg.maximum(*[alg.sdist_vector(part) for block in blocks for part in block])


class _Context:
    """What a regulariser term reads: the variables, the bounds of the basic concepts, the
    individuals, the sizes and the loss."""

    def __init__(self, alg, v, s_omega, margin, variant, dim):
        self.alg, self.v, self.s, self.m, self.variant, self.d = alg, v, s_omega, margin, \
            variant, dim
        self.BL, self.BU = basic_bounds(alg, v)
        self.loss = _Loss(alg, variant, margin)


class NegativeConceptPenalty:
    """The negative concept regularisation (Eq. 6), weight :math:`\\lambda_1`: the largest
    signed distance of a potential concept assertion :math:`D(a)` that the ABox does not
    state to its *complement* box (the term is negative once :math:`a` is outside
    :math:`D`).

    :param negatives: Rows ``[basic concept, individual]`` (:func:`negative_concept_assertions`).
    :param weight: The weight of the penalty on a geometry (a solved program takes its
        weight as a parameter, ``weight_name``).
    """

    kind = "penalty"
    weight_name = "lambda1"

    def __init__(self, negatives, weight=1.0):
        self.negatives = np.asarray(negatives, dtype=np.int64).reshape(-1, 2)
        self.weight = weight

    def term(self, c):
        """The unweighted term, or ``None`` if there are no negatives or individuals."""
        if not len(self.negatives) or c.v["pos"] is None:
            return None
        q, a = self.negatives[:, 0], self.negatives[:, 1]
        return c.loss([c.loss.residuals(-c.s - c.BL[q], c.s - c.BU[q], c.v["pos"][a])])

    def penalty(self, geometry):
        term = self.term(geometry.context(_TORCH))
        return None if term is None else self.weight * term


class WidthPenalty:
    """The width regularisation: the sum of the norms of the box widths
    :math:`\\|U - L\\|_2` (Eq. 7), of the head, tail and concept boxes (``which="boxes"``,
    weight :math:`\\lambda_2`) or of the bump boxes (``which="bump"``, :math:`\\lambda_3`).
    In the reference reading the widths are the row-major vector reshaped to ``d`` rows
    (CVXPY's ``mixed_norm``)."""

    kind = "penalty"

    def __init__(self, which="boxes", weight=1.0):
        if which not in ("boxes", "bump"):
            raise ValueError("which must be 'boxes' or 'bump'.")
        self.which, self.weight = which, weight
        self.weight_name = "lambda2" if which == "boxes" else "lambda3"

    def term(self, c):
        alg, v, d = c.alg, c.v, c.d
        widths = [v[up] - v[lo] for lo, up in (("concept_lower", "concept_upper"),
                                               ("head_lower", "head_upper"),
                                               ("tail_lower", "tail_upper"))
                  if v[lo] is not None]
        bump = None if v["bump_lower"] is None else v["bump_upper"] - v["bump_lower"]
        if c.variant == "paper":
            if self.which == "bump":
                return None if bump is None else alg.sum(alg.row_norms(bump))
            return sum(alg.sum(alg.row_norms(w)) for w in widths) if widths else None

        def mixed(blocks):
            flat = alg.join([alg.flatten(w) for w in blocks])
            n = flat.shape[0]
            return alg.sum(alg.row_norms(alg.reshape(flat, (d, n // d))))

        if self.which == "bump":
            return mixed([bump]) if bump is not None else None
        if not widths:
            return None
        if v["concept_lower"] is None:
            widths = [alg.constant(np.zeros((1, d)))] + widths
        return mixed(widths)

    def penalty(self, geometry):
        term = self.term(geometry.context(_TORCH))
        return None if term is None else self.weight * term


def default_regularizers(negatives):
    """The three regularisers of the paper, ``lambda1``, ``lambda2``, ``lambda3``."""
    return (NegativeConceptPenalty(negatives), WidthPenalty("boxes"), WidthPenalty("bump"))


class BoxLitEProgram(Program):
    """The BoxLitE second-order cone programme for one :math:`\\text{DL-Lite}_\\mathcal{H}`
    KB (Section 5 and Appendix E): the variables are the tables of the parts (a position and
    a bump per individual, a box per concept name, a head, a tail and a bump box per role),
    the constraints are the grounding of the TBox and of the box widths (all linear), the
    objective is the maximum of the assertion losses plus the weighted regularisers.

    The regularisation weights are parameters, so the problem is compiled once and solved for
    every :math:`(\\lambda_1, \\lambda_2, \\lambda_3)` (as in Section 6, Q2).

    :param kb: The knowledge base (:class:`mowl.datasets.dllite.DLLiteKB`).
    :param dim: Dimension :math:`d`; at least :math:`|N_C| + 2|N_R|` (one reserved
        dimension per basic concept for box consistency, Eq. 1).
    :param s_omega: :math:`s_\\Omega` (the experiments use 1).
    :param epsilon: :math:`\\epsilon`; the experiments use 0.01. The proofs need
        :math:`0 < \\epsilon \\le s_\\Omega / 8`.
    :param variant: ``"paper"`` or ``"reference"`` (module docstring).
    :param negatives: Rows ``[basic, individual]`` of the negative concept assertions, or
        ``None`` for :func:`negative_concept_assertions` with the variant's mode.
    :param regularizers: The parts that add terms to the objective (default: the three of
        the paper). A regulariser has ``weight_name``, and ``term(context)``.
    """

    def __init__(self, kb, dim, s_omega=1.0, epsilon=0.01, variant="paper", negatives=None,
                 regularizers=None):
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}.")
        if not (s_omega > 0 and epsilon > 0 and epsilon <= s_omega):
            raise ValueError("Need 0 < epsilon <= s_omega (Section 3).")
        nb = kb.n_basic
        if dim < max(nb, 1):
            raise ValueError(
                f"dim={dim} is too small: box consistency reserves one dimension per basic "
                f"concept (Eq. 1), so dim >= |N_C| + 2|N_R| = {nb}.")
        self.kb, self.dim, self.variant = kb, int(dim), variant
        self.s_omega, self.epsilon = float(s_omega), float(epsilon)
        self.margin = self.epsilon if variant == "paper" else 0.0
        if negatives is None:
            negatives = negative_concept_assertions(kb, variant)
        self.negatives = np.asarray(negatives, dtype=np.int64).reshape(-1, 2)
        self.regularizers = tuple(default_regularizers(self.negatives)
                                  if regularizers is None else regularizers)
        self.weights = tuple(dict.fromkeys(r.weight_name for r in self.regularizers))
        self._compiled = None

    # -- Program -------------------------------------------------------------------------
    def variables(self):
        kb, d = self.kb, self.dim
        shapes = {}
        for names, n in ((INDIVIDUAL_TABLES, len(kb.individuals)),
                         (CONCEPT_TABLES, len(kb.concepts)), (ROLE_TABLES, len(kb.roles))):
            for name in names:
                shapes[name] = (n, d)
        return shapes

    def build(self, alg, v, w):
        kb, s, m, d, variant = self.kb, self.s_omega, self.margin, self.dim, self.variant
        ni, nr, nb = len(kb.individuals), len(kb.roles), kb.n_basic
        ctx = _Context(alg, v, s, m, variant, d)
        BL, BU = ctx.BL, ctx.BU

        # widths (Eq. 2) and universe (Eq. 3)
        for lo, up in (("concept_lower", "concept_upper"), ("head_lower", "head_upper"),
                       ("tail_lower", "tail_upper"), ("bump_lower", "bump_upper")):
            if v[lo] is not None:
                alg.le(0, v[up] - v[lo], "width")
                alg.le(v[up] - v[lo], 2 * s, "width")
        if ni:
            bound = s - m
            for name in INDIVIDUAL_TABLES:
                alg.le(v[name], bound, "universe")
                alg.le(-bound, v[name], "universe")

        # box consistency (Eq. 1): dimension i_C = index of C
        if nb:
            idx = np.arange(nb)
            rhs = -s if variant == "paper" else -s - 2 * self.epsilon
            alg.le(BL[idx, idx] + BU[idx, idx], rhs, "box consistency")

        # the TBox, grounded (Section 5.1)
        ci = kb.concept_inclusions
        if len(ci):
            inside_bounds(alg, (BL[ci[:, 0]], BU[ci[:, 0]]), (BL[ci[:, 1]], BU[ci[:, 1]]),
                          "concept inclusion")
        neg = kb.negative_inclusions
        if len(neg):
            inside_complement_bounds(alg, (BL[neg[:, 0]], BU[neg[:, 0]]),
                                     (BL[neg[:, 1]], BU[neg[:, 1]]), s,
                                     "negative inclusion")
        ri = kb.role_inclusions
        if len(ri):
            # rows 0..nr-1: R, rows nr..2nr-1: R^- (head and tail swapped)
            head_l = alg.vstack([v["head_lower"], v["tail_lower"]])
            head_u = alg.vstack([v["head_upper"], v["tail_upper"]])
            tail_l = alg.vstack([v["tail_lower"], v["head_lower"]])
            tail_u = alg.vstack([v["tail_upper"], v["head_upper"]])
            sub, sup = ri[:, 0] + nr * ri[:, 1], ri[:, 2] + nr * ri[:, 3]
            inside_bounds(alg, (head_l[sub], head_u[sub]), (head_l[sup], head_u[sup]),
                          "role inclusion")
            inside_bounds(alg, (tail_l[sub], tail_u[sub]), (tail_l[sup], tail_u[sup]),
                          "role inclusion")
            inside_bounds(alg, (v["bump_lower"][ri[:, 0]], v["bump_upper"][ri[:, 0]]),
                          (v["bump_lower"][ri[:, 2]], v["bump_upper"][ri[:, 2]]),
                          "role inclusion")

        # the assertions: the objective of the geometry
        terms = {}
        ca = kb.concept_assertions
        if len(ca):
            terms["concept"] = ctx.loss([ctx.loss.residuals(BL[ca[:, 0]], BU[ca[:, 0]],
                                                            v["pos"][ca[:, 1]])])
        ra = kb.role_assertions
        if len(ra):
            r, a, b = ra[:, 0], ra[:, 1], ra[:, 2]
            P, B = v["pos"], v["bump"]
            res = ctx.loss.residuals
            terms["role"] = ctx.loss([
                res(v["head_lower"][r], v["head_upper"][r], P[a] + B[b]),
                res(v["tail_lower"][r], v["tail_upper"][r], P[b] + B[a]),
                res(v["bump_lower"][r], v["bump_upper"][r], B[a]),
                res(v["bump_lower"][r], v["bump_upper"][r], B[b])])
        if variant == "paper":
            groups = [terms[k] for k in ("concept", "role") if k in terms]
            assertion = (alg.maximum(*groups) if len(groups) > 1 else groups[0]) if groups \
                else alg.constant(0.0)
        else:
            assertion = alg.maximum(terms.get("concept", alg.constant(0.0)),
                                    terms.get("role", alg.constant(0.0)))
        objective = assertion
        alg.terms.update(terms)
        alg.terms["assertion"] = assertion
        for reg in self.regularizers:
            term = reg.term(ctx)
            if term is not None:
                objective = objective + w[reg.weight_name] * term
                alg.terms[type(reg).__name__ + ":" + reg.weight_name] = term
        return objective

    # -- the compiled problem (needs cvxpy) ---------------------------------------------------
    @property
    def compiled(self):
        if self._compiled is None:
            self._compiled = CompiledProblem(self)
        return self._compiled

    @property
    def problem(self):
        """The CVXPY problem (compiled on first use)."""
        return self.compiled.problem

    @property
    def terms(self):
        """The CVXPY expressions of the terms of the objective, by name."""
        return self.compiled.terms

    @property
    def constraints(self):
        return self.compiled.constraints

    def size(self):
        """Numbers of scalar variables and constraints of the CVXPY problem."""
        sm = self.problem.size_metrics
        return {"variables": int(sm.num_scalar_variables),
                "equality_constraints": int(sm.num_scalar_eq_constr),
                "inequality_constraints": int(sm.num_scalar_leq_constr)}

    def solve(self, lambdas=(0.0, 0.0, 0.0), solver=None, verbose=False, **solver_options):
        """Solve for the given :math:`(\\lambda_1, \\lambda_2, \\lambda_3)` with CVXPY.

        :param solver: A CVXPY solver name (``"CLARABEL"``, ``"SCS"``, ``"MOSEK"``) or a
            :class:`~mowl.representation.library.solved_solver.Solver`; default: the
            open-source conic solvers in order.
        :rtype: :class:`BoxLitEProgramResult`
        """
        weights = dict(zip(("lambda1", "lambda2", "lambda3"), (float(x) for x in lambdas)))
        if solver is not None and not isinstance(solver, str):
            result = solver.solve(self, weights, **solver_options)
        else:
            result = CvxpySolver().solve(self.compiled, weights, solver=solver,
                                         verbose=verbose, terms=self.compiled.terms,
                                         **solver_options)
        return BoxLitEProgramResult.of(result, lambdas, self)

    def embedding(self, values):
        """The values of the variables as a :class:`BoxLitEEmbedding`."""
        d = self.dim
        shapes = self.variables()

        def val(name):
            x = values.get(name)
            return np.zeros((0, d)) if x is None or np.size(x) == 0 \
                else np.array(x, dtype=float).reshape(shapes[name])

        return BoxLitEEmbedding(**{k: val(k) for k in BoxLitEEmbedding._ARRAYS},
                                s_omega=self.s_omega, margin=self.margin)


@dataclass
class BoxLitEProgramResult(SolveResult):
    """A :class:`~mowl.representation.library.solved_solver.SolveResult` with the
    :class:`BoxLitEEmbedding` of the solution and the weights as ``lambdas``."""

    embedding: object = None
    lambdas: tuple = ()

    @classmethod
    def of(cls, result, lambdas, program):
        out = cls(**{k: getattr(result, k) for k in
                     ("status", "value", "solver", "weights", "values", "solve_time",
                      "total_time", "info")})
        out.lambdas = tuple(float(x) for x in lambdas)
        if out.solved:
            out.embedding = program.embedding(out.values)
        return out


# -- the geometry --------------------------------------------------------------------------------
class BoxLitEGeometry(ComposedGeometry):
    """BoxLitE as a combination of parts (module docstring).

    Class ids ``0 .. nc - 1`` are concept names, then :math:`\\exists R` for every role,
    then :math:`\\exists R^-` (the basic concepts of DL-Lite, in the order of
    :meth:`BoxLitEEmbedding.basic_bounds`). ``exists`` is defined for the universe
    (DL-Lite has unqualified existentials) and for a nominal; ``negation`` is the complement
    box :math:`[-s_\\Omega - L, s_\\Omega - U]`; ``member`` is the signed distance of the paper
    (negative inside), and the largest of four distances for :math:`\\exists R.\\{b\\}`.
    The rows of ``class_assertion`` are ``[individual, basic concept]``; of
    ``gci0``, ``[B1, B2]`` (basic concepts); of ``role_inclusion``, ``[R, S]`` with signed
    role ids ``r + nr * inverse``; of ``object_property_assertion`` ``[a, R, b]``.

    :param variant: ``"paper"`` or ``"reference"``.
    """

    def __init__(self, classes, relations, individuals, measure=None, objective=None,
                 s_omega=1.0, epsilon=0.01, variant="paper", regularizers=()):
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}.")
        self.s_omega, self.epsilon, self.variant = float(s_omega), float(epsilon), variant
        margin = self.epsilon if variant == "paper" else 0.0
        super().__init__(
            classes, relations, measure if measure is not None else SignedDistance(),
            objective if objective is not None else BoxLitEObjective(margin),
            individuals=individuals, regularizers=regularizers,
            supported_forms=("gci0", "class_assertion", "object_property_assertion",
                             "role_inclusion"),
            neg_capable_gcis=("class_assertion",))

    # -- tables --------------------------------------------------------------------------
    def table_values(self):
        """The tables of the parts by the names of the variables of the programme (``None``
        for a part that has no rows)."""
        out = {}
        for name in CONCEPT_TABLES:
            table = getattr(self.class_model, name.split("_")[1])
            out[name] = table.weight if table.num_embeddings else None
        for name, table in (self.relation_model.tables().items()
                            if self.relation_model is not None else ()):
            out[name] = table if len(table) else None
        for name in ROLE_TABLES:
            out.setdefault(name, None)
        ind = self.individual_model
        out["pos"] = ind.centres.weight if ind is not None and len(ind) else None
        out["bump"] = ind.bumps.weight if ind is not None and len(ind) else None
        return out

    def load_values(self, values):
        """Put a solution (arrays by variable name) into the tables of the parts."""
        for name, table in self.table_values().items():
            if table is not None and values.get(name) is not None:
                with th.no_grad():
                    table.copy_(th.as_tensor(np.asarray(values[name]), dtype=table.dtype))

    def context(self, alg=None):
        return _Context(alg or _TORCH, self.table_values(), self.s_omega, self.objective.margin,
                        self.variant, self.class_model.embed_dim)

    def snapshot(self):
        """The current tables as a :class:`BoxLitEEmbedding`."""
        d = self.class_model.embed_dim

        def array(t):
            return np.zeros((0, d)) if t is None else t.detach().cpu().numpy().astype(float)

        values = {name: array(t) for name, t in self.table_values().items()}
        return BoxLitEEmbedding(**values, s_omega=self.s_omega, margin=self.objective.margin)

    # -- regions ---------------------------------------------------------------------------
    def basic_bounds(self):
        """``(L, U)`` tensors of all basic concepts."""
        return basic_bounds(_TORCH, self.table_values())

    def classes(self, ids, frozen=False):
        lower, upper = self.basic_bounds()
        return CornerBox.from_corners(lower[ids], upper[ids])

    @property
    def num_classes(self):
        """The number of basic concepts: names, then exists R, then exists R^-."""
        return len(self.class_model) + 2 * len(self.relation_model)

    def universe(self, n=1):
        d = self.class_model.embed_dim
        return Universe(th.zeros(n, d), th.full((n, d), float(self.s_omega)))

    def exists(self, relation_ids, region, inverse=False):
        if isinstance(region, Point) and region.bump is not None:
            return Related(self.relation_model, relation_ids, region)
        if not isinstance(region, Universe):
            raise ConstructorUnavailable("DL-Lite has unqualified existentials: the filler "
                                         "must be geometry.universe().")
        return self.relation_model.preimage(relation_ids, region, inverse)

    def negation(self, a):
        """The complement box :math:`[-s_\\Omega - L, s_\\Omega - U]` (not the set
        complement: it is convex)."""
        s = self.s_omega
        return CornerBox.from_corners(-s - a.lower, s - a.upper)

    def member(self, individual_ids, region):
        if isinstance(region, Related):
            return region.relations.pair_distance(
                region.ids, self.individuals(individual_ids), region.filler, self.objective.m)
        return super().member(individual_ids, region)

    def negative_inclusion_cost(self, rows):
        """Violation of :math:`B_1 \\sqsubseteq \\neg B_2` (rows of basic concepts)."""
        return self.measure.inside(self.classes(rows[:, 0]),
                                   self.negation(self.classes(rows[:, 1])), 0.0)

    @property
    def parts(self):
        return {"classes": self.class_model.part, "relations": self.relation_model.part,
                "individuals": "point with bump", "measure": self.measure.part,
                "conjunction": self.class_model.conjunction_kind, "negation": "exact"}

    @property
    def device(self):
        return self.class_model.lower.weight.device


# -- the method ----------------------------------------------------------------------------------
@register_method("boxlite")
def boxlite(nb_classes, nb_relations, nb_individuals=0, embed_dim=32, s_omega=1.0, epsilon=0.01,
            variant="paper", classes=None, relations=None, individuals=None, measure=None,
            objective=None, regularizers=(), tables=None):
    """BoxLitE (Lourenço et al., KR 2026) from its parts: a box per concept name
    (:class:`BoundsBoxes`), a head, tail and bump box per role (:class:`HeadTailBumpBoxes`),
    a position and a bump per individual, the signed distance (:class:`SignedDistance`) and the
    objective of the paper (:class:`BoxLitEObjective`). Fitted by a solver
    (:class:`~mowl.representation.library.solved_solver.SolverFit`), see
    :class:`mowl.models.BoxLitE`.

    :param variant: ``"paper"`` (the programme as printed; :math:`\\epsilon` is the margin) or
        ``"reference"`` (the authors' code: closed boxes, margin 0).
    :param classes, relations, individuals, measure, objective: Replace a part (an instance).
    :param regularizers: Penalties added to the module's regularisation loss (the weights of
        the three of the paper are those of the solver; see
        :func:`default_regularizers`).
    :param tables: Not used (BoxLitE has no module of its own whose tables could be shared).
    """
    if tables is not None:
        raise TypeError("BoxLitE has no module whose tables could be shared.")
    classes = classes if classes is not None else BoundsBoxes(nb_classes, embed_dim)
    relations = relations if relations is not None else HeadTailBumpBoxes(nb_relations,
                                                                          embed_dim)
    if individuals is None:
        individuals = Points(nb_individuals, embed_dim, family="point", init="uniform",
                             scale=0.5, bump=True)
    return BoxLitEGeometry(classes, relations, individuals, measure, objective,
                           s_omega=s_omega, epsilon=epsilon, variant=variant,
                           regularizers=regularizers)
