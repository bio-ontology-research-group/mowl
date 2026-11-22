"""Relation models of the methods that embed a hierarchy (group ``hierarchy``): OnT's map per
relation, which is a relation model on points."""

import torch as th

from mowl.representation.library.hierarchy_regions import Point
from mowl.representation.regions import ConstructorUnavailable
from mowl.representation.relation_models import Relations

__all__ = ["TextRotation"]


class TextRotation(Relations):
    r"""A relation is a map :math:`f_r(v) = k_r \otimes (R(\Theta_r) v)` of the points of a
    Poincare ball: a block rotation and a Moebius scaling whose parameters are a linear
    function of the encoding of the relation's name (OnT, Yang et al. 2025, Eq. 4). The
    region of :math:`\exists r.D` is :math:`f_r(x_D)`: OnT trains the point of the compound
    class (which it embeds from its verbalisation) to coincide with it (Eq. 6).

    The map goes one way, so there is no :meth:`image`; the rotation is an isometry of the
    ball, so the order of the regions is preserved.

    :param role_model: A :class:`mowl.nn.lm.ont.RoleTransformation` (its parameters are
        shared).
    :param encoder: The encoder of the role names (that of the class model, normally).
    :param curvature: The curvature parameter :math:`c` of the ball.
    :param role_texts: The name of every relation id, to look relations up by id; without it
        the relations are lists of strings.
    """

    part = "rotation"

    def __init__(self, role_model, encoder, curvature, role_texts=None):
        super().__init__()
        self.role_model = role_model
        self.encoder = encoder
        self.c = float(curvature)
        self.role_texts = None if role_texts is None else list(role_texts)

    def __len__(self):
        return 0 if self.role_texts is None else len(self.role_texts)

    def _texts(self, ids):
        if isinstance(ids, (list, tuple)) and all(isinstance(t, str) for t in ids):
            return list(ids)
        if self.role_texts is None:
            raise TypeError("This relation model has no list of names: pass a list of "
                            "strings.")
        return [self.role_texts[int(i)] for i in th.as_tensor(ids).reshape(-1)]

    def preimage(self, ids, region):
        embedding = self.encoder(self._texts(ids))
        return Point(self.role_model(region.centre, embedding, self.c), region.manifold)

    def image(self, ids, region):
        raise ConstructorUnavailable("The map of a relation has no inverse here.")
