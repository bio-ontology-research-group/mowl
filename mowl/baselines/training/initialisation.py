"""Initialisation of the embedding parameters, as a training regime.

How the parameters are initialised is part of how a model is trained. mOWL's ball and
box modules normalise the centre and the radius (or offset) vector of a class
*separately* to unit norm, so every ELEm radius starts at :math:`|r| = 1`. The reference
code of ELEmbeddings and of the baselines of the Box\\ :sup:`2`\\ EL code stores centre and
radius (or offset) in one vector and normalises that vector *jointly*: with dimension
:math:`n` the radius then starts at about :math:`0.5/\\sqrt{(n+1)/3}` (about 0.06 for
:math:`n = 200`) and the centre slightly inside the unit sphere.
"""

import torch as th

from mowl.baselines.training.regime import TrainingRegime


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


# Reference code by definition: the initialisation of the ELEm / ELBE reference code, which
# is about the two specific modules; other methods register their groups here.
#: Module class name -> function returning the ``(centre, size)`` embedding pairs that are
#: normalised together. Base classes count (EmEL++ is an ELEm module).
JOINT_GROUPS = {
    "ELEmModule": _ball_groups,
    "ELBEModule": _box_groups,
}

#: Module class name -> role embeddings, normalised on their own.
ROLE_EMBEDDINGS = {
    "ELEmModule": "rel_embed",
    "ELBEModule": "rel_embed",
}


def _lookup(table, module):
    for cls in type(module).__mro__:
        if cls.__name__ in table:
            return table[cls.__name__]
    return None


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
        groups = _lookup(JOINT_GROUPS, module)
        if groups is None:
            if not self.strict:
                return
            raise TypeError(
                f"{type(module).__name__} has no registered (centre, size) embeddings; "
                f"register them in mowl.training.initialisation.JOINT_GROUPS.")
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
            rel = getattr(module, _lookup(ROLE_EMBEDDINGS, module) or "", None)
            if rel is not None:
                w = th.rand(*rel.weight.shape, generator=gen) * 2 - 1
                rel.weight.copy_((w / th.linalg.norm(w, dim=1, keepdim=True)).to(rel.weight))
