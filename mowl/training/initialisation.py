"""Initialisation of the embedding parameters, as a training regime.

How the parameters are initialised is part of how a model is trained. mOWL's ball and
box modules normalise the centre and the radius (or offset) vector of a class
*separately* to unit norm, so every ELEm radius starts at :math:`|r| = 1`. The reference
code of ELEmbeddings and of the baselines of the Box\\ :sup:`2`\\ EL code stores centre and
radius (or offset) in one vector and normalises that vector *jointly*: with dimension
:math:`n` the radius then starts at about :math:`0.5/\\sqrt{(n+1)/3}` (about 0.06 for
:math:`n = 200`) and the centre slightly inside the unit sphere.
"""

import contextlib

import numpy as np
import torch as th

from mowl.training.regime import TrainingRegime


def _ball_groups(module):
    groups = [(module.class_embed, module.class_rad)]
    if getattr(module, "ind_embed", None) is not None and \
            getattr(module, "ind_rad", None) is not None:
        groups.append((module.ind_embed, module.ind_rad))
    return groups


def _box_groups(module):
    groups = [(module.class_embed, module.class_offset)]
    if getattr(module, "ind_embed", None) is not None and \
            getattr(module, "ind_offset", None) is not None:
        groups.append((module.ind_embed, module.ind_offset))
    return groups


#: The ``(centre, size)`` embedding pairs of a module that are normalised together, by the
#: kind the module declares in ``joint_init`` (``"ball"``: ``class_embed`` and ``class_rad``,
#: ``"box"``: ``class_embed`` and ``class_offset``, and the same for individuals). A module of
#: another kind declares its own kind here, or the regime takes ``strict=False``. The role
#: embeddings that are drawn and normalised on their own are the attribute the module names in
#: ``joint_init_roles``.
JOINT_GROUPS = {"ball": _ball_groups, "box": _box_groups}


def _groups(module):
    """The pairs the module's declared ``joint_init`` names, or ``None``."""
    return JOINT_GROUPS.get(getattr(module, "joint_init", None))


class JointNormalisedInitialisation(TrainingRegime):
    """Re-initialise centre and radius (or offset) of every class as *one* vector:
    uniform in :math:`[-1, 1]`, then the concatenation is scaled to unit norm. Role
    vectors are drawn and normalised on their own. This is the initialisation of
    ``model/Elem.py``, ``model/EmELpp.py`` and ``model/Elbe.py`` of the
    Box\\ :sup:`2`\\ EL code (``nn.Embedding(classes, dim + 1)`` or ``dim * 2``, then
    divided by the row norm) and of the original ELEmbeddings code.

    :param size_low: Lower end of the uniform draw of the radius (offset) coordinates
        before normalisation: ``-1`` (default; ELEm and the Box2EL code) or ``0`` (the
        EmEL++ code draws radii from :math:`[0, 1]`).
    :param strict: Raise :class:`TypeError` for a module without registered
        ``(centre, size)`` embeddings. With ``False`` such a module keeps its own
        initialisation (BoxEL in the Box2EL code is initialised as in mOWL).
    :param seed: Seed of the draw (``None``: PyTorch's global generator).
    """

    key = "joint-normalised-initialisation"

    def __init__(self, size_low=-1.0, strict=True, seed=None):
        super().__init__()
        self.strict = bool(strict)
        self.size_low = float(size_low)
        if not -1.0 <= self.size_low < 1.0:
            raise ValueError("size_low must be in [-1, 1).")
        self.seed = seed

    def attach(self, model):
        super().attach(model)
        module = getattr(model, "module", None)
        groups = _groups(module)
        if groups is None:
            if not self.strict:
                return
            raise TypeError(
                f"{type(module).__name__} does not declare (centre, size) embeddings to "
                f"normalise together (its attribute joint_init: one of "
                f"{sorted(JOINT_GROUPS)}).")
        gen = None
        if self.seed is not None:
            gen = th.Generator(device="cpu").manual_seed(int(self.seed))
        with th.no_grad():
            for centre, size in groups(module):
                w_c, w_s = centre.weight, size.weight
                n = w_c.shape[0]
                c = th.rand(n, w_c.shape[1], generator=gen) * 2 - 1
                sz = th.rand(n, w_s.shape[1], generator=gen) * (1 - self.size_low) \
                    + self.size_low
                joint = th.cat([c, sz], dim=1)
                joint = joint / th.linalg.norm(joint, dim=1, keepdim=True)
                w_c.copy_(joint[:, :w_c.shape[1]].to(w_c))
                w_s.copy_(joint[:, w_c.shape[1]:].to(w_s))
            rel = getattr(module, getattr(module, "joint_init_roles", None) or "", None)
            if rel is not None:
                w = th.rand(*rel.weight.shape, generator=gen) * 2 - 1
                rel.weight.copy_((w / th.linalg.norm(w, dim=1, keepdim=True)).to(rel.weight))


def extend_embeddings(previous, keys, dim):
    """The ``(len(keys), dim)`` table whose row for a key in ``previous`` (a dict key ->
    vector) is the old vector and whose row for a new key is drawn from the standard normal
    distribution (NumPy's global generator, in the order of ``keys``). Used when axioms are
    added to a trained model: old entities keep their embedding, new ones start at random."""
    return np.asarray([previous[key] if key in previous else np.random.normal(size=dim)
                       for key in keys])


def margin_embedding_range(gamma, dim, slack=2.0):
    """The half-width ``(gamma + slack) / dim`` of the uniform initialisation of the embedding
    tables of the query embeddings (Query2Box, BetaE, ConE: the reference code initialises
    every table in ``[-r, r]`` with ``r`` tied to the margin ``gamma`` of the score so that a
    random point starts inside the margin)."""
    return (float(gamma) + slack) / dim


def redraw_after(build, redraw):
    """Make something with ``build()``, then run ``redraw()`` from the state the random
    generator had before ``build``: the initial values ``redraw`` draws are those a module that
    drew them in a different order (or in a different place) would have drawn under the same
    seed. Returns what ``build`` returns. The generator is in the state ``redraw`` leaves it in
    afterwards."""
    state = th.get_rng_state()
    result = build()
    th.set_rng_state(state)
    redraw()
    return result


@contextlib.contextmanager
def seeded_scope(seed):
    """Run a block with the torch generator seeded with ``seed`` and put the generator back as
    it was afterwards: a deterministic draw (a fixed sample, a probe head) that does not touch
    the stream of random numbers of the run."""
    state = th.random.get_rng_state()
    th.manual_seed(seed)
    try:
        yield
    finally:
        th.random.set_rng_state(state)
