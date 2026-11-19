r"""Losses of the learnable al-cone model (:class:`~mowl.nn.alc.alcone.module.ALConeModule`).

The published al-cone papers (Özçep, Leemhuis & Wolter, IJCAI 2020; JAIR 78, 2023) give
the geometry and model constructions but no learning objective for ontology embedding
(JAIR Sec. 7: "Development of a practical learning algorithm and its experimental
evaluation are outside the scope of this paper"). The relaxation below is **ours**
(an adaptation).

Parametrisation
---------------
An al-cone of :math:`\mathbb{R}^n` is a subset of the :math:`2n` half-axes (JAIR
Prop. 1; see :mod:`mowl.geometric.cones`). Every concept name gets a real *logit* per
half-axis; the discretised cone contains half-axis :math:`h` iff its logit is
:math:`> 0`. Per coordinate the two logits select one of the four factors
:math:`\{0\}, \mathbb{R}^+, \mathbb{R}^-, \mathbb{R}`. A role gets a logit per entry of its
:math:`2n \times 2n` incidence matrix (JAIR Def. 7), an individual a logit per half-axis
of its *support* (the discretised point has coordinate :math:`+1`, :math:`-1` or
:math:`0` on each axis).

Complex concepts are evaluated with Gödel operators on logits: :math:`\sqcap` is
:math:`\min`, :math:`\sqcup` is :math:`\max`, :math:`\neg` is :math:`-x` (the polar cone is
the complement of the half-axis set), :math:`(\exists R.C)_i = \max_j \min(R_{ji},
C_j)` and :math:`\forall R.C = \neg\exists R.\neg C`.

Exactness (our lemma, proved by induction on concepts and tested in
``tests/nn/test_alcone_module.py``): if the value of a concept at a half-axis is
:math:`> 0` the discretised cone contains the half-axis, and if it is :math:`< 0` it does
not. Consequently, with margin :math:`\gamma > 0`, each loss below is zero only if the
discretised interpretation satisfies the axiom; conversely, if the discretised
interpretation satisfies the axiom and every logit has absolute value at least
:math:`\gamma`, the loss is zero.
"""
import torch as th
import torch.nn.functional as F


def _smax(x, y, tau):
    if tau <= 0:
        return th.maximum(x, y)
    return tau * th.logaddexp(x / tau, y / tau)


def godel_and(x, y, tau=0.0):
    """Conjunction (intersection of cones): elementwise minimum. With ``tau > 0`` the
    smooth lower bound :math:`-\\tau \\log(e^{-x/\\tau} + e^{-y/\\tau})` is used
    (training only)."""
    return -_smax(-x, -y, tau)


def godel_or(x, y, tau=0.0):
    """Disjunction (convex union of cones): elementwise maximum (smooth upper bound for
    ``tau > 0``)."""
    return _smax(x, y, tau)


def godel_not(x):
    """Negation (polar cone, complement of the half-axis set): ``-x``."""
    return -x


def _reduce_max(x, dim, tau):
    if tau <= 0:
        return x.max(dim=dim)[0]
    return tau * th.logsumexp(x / tau, dim=dim)


def exists(r, c, tau=0.0):
    r"""Existential restriction on half-axis logits.

    :param r: Incidence logits of shape ``(B, 2n, 2n)``; ``r[b, j, i]`` is the logit of
        "half-axis ``i`` has half-axis ``j`` as successor" (JAIR's :math:`R_{ji}`).
    :param c: Filler logits of shape ``(B, 2n)``.
    :param tau: Temperature of the smooth relaxation (``0``: exact Gödel operators).
    :returns: Logits of :math:`\exists R.C`, shape ``(B, 2n)``:
        :math:`\max_j \min(r_{ji}, c_j)`.
    """
    return _reduce_max(godel_and(r, c.unsqueeze(-1).expand_as(r), tau), -2, tau)


def forall(r, c, tau=0.0):
    r""":math:`\forall R.C = \neg \exists R. \neg C`."""
    return -exists(r, -c, tau)


def _either(x, y, margin):
    r"""Hinge penalty of the disjunction ":math:`x \ge \gamma` or :math:`y \ge \gamma`":
    :math:`\mathrm{relu}(\gamma - x)\,\mathrm{relu}(\gamma - y)`. It is zero iff
    :math:`\max(x, y) \ge \gamma`, i.e. it has the same zero set as
    :math:`\mathrm{relu}(\gamma - \max(x, y))`, but its gradient reaches both arguments."""
    return F.relu(margin - x) * F.relu(margin - y)


def inclusion_loss(c, d, margin, tau=0.0):
    r"""Loss of :math:`C \sqsubseteq D` per row: mean over half-axes of
    :math:`\mathrm{relu}(\gamma + \min(c_h, -d_h))`. With ``tau = 0`` it is zero iff at
    every half-axis :math:`c_h \le -\gamma` or :math:`d_h \ge \gamma`."""
    return _either(-c, d, margin).mean(dim=-1)


def disjointness_loss(c, d, margin, tau=0.0):
    r"""Loss of :math:`C \sqcap D \sqsubseteq \bot` (intersection :math:`= \{0\}`)."""
    return _either(-c, -d, margin).mean(dim=-1)


def membership_loss(u, c, margin, tau=0.0):
    r"""Loss of :math:`C(a)` given the support logits ``u`` of :math:`a`: mean over
    half-axes of :math:`\mathrm{relu}(\gamma + \min(u_h, -c_h))`. With ``tau = 0`` it is
    zero iff at every half-axis :math:`u_h \le -\gamma` or :math:`c_h \ge \gamma`
    (the support of the point lies in the cone)."""
    return _either(-u, c, margin).mean(dim=-1)


def point_validity_loss(u, margin, tau=0.0):
    r"""Well-formedness of an individual's support logits ``u`` (shape ``(B, 2n)``):
    the point is non-zero (JAIR Def. 2: individuals are mapped to
    :math:`\Delta \setminus \{\vec 0\}`), i.e. :math:`\max_h u_h \ge \gamma`, and no
    coordinate is both positive and negative, i.e. :math:`\min(u_{2i}, u_{2i+1}) \le
    -\gamma`."""
    nonzero = F.relu(margin - _reduce_max(u, -1, tau))
    pairs = u.view(u.shape[0], -1, 2)
    exclusive = _either(-pairs[..., 0], -pairs[..., 1], margin).mean(dim=-1)
    return nonzero + exclusive


def successor_logits(r, u, tau=0.0):
    r"""Logits of the successor half-axes of a point: :math:`\max_i \min(u_i, r_{ji})`
    for each :math:`j` (the support of :math:`R\, a'`)."""
    return _reduce_max(godel_and(r, u.unsqueeze(-2).expand_as(r), tau), -1, tau)


def role_assertion_loss(u_subject, r, u_object, margin, tau=0.0):
    r"""Loss of :math:`R(a, b)`: every half-axis in the support of :math:`b` is an
    :math:`R`-successor of a half-axis in the support of :math:`a` (role-satisfiability,
    JAIR Sec. 6). Mean over half-axes :math:`j` of
    :math:`\mathrm{relu}(\gamma + \min(u^b_j, -\mathrm{succ}_j))`."""
    succ = successor_logits(r, u_subject, tau)
    return _either(-u_object, succ, margin).mean(dim=-1)


def nondegeneracy_loss(c, margin, tau=0.0):
    r"""Regulariser (not part of the semantics): concept ``c`` is not :math:`\{0\}`,
    i.e. some half-axis has logit :math:`\ge \gamma`."""
    return F.relu(margin - _reduce_max(c, -1, tau))
