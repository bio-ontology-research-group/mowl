"""The module of OnT (Yang, Chen, He, Gao, Horrocks, "Language Models as Ontology Encoders",
ISWC 2025, Section 4): HiT (:mod:`mowl.nn.lm.hit`) with a relation model, the map per role of
Eq. 4 (:class:`~mowl.representation.library.hierarchy_relations.TextRotation`), and the
logical losses of Eq. 6 and 7 (:meth:`~mowl.representation.library.hierarchy_geometry.TextGeometry.existential_loss`,
:meth:`~mowl.representation.library.hierarchy_geometry.TextGeometry.conjunction_loss`).

It is a combination of parts; the method is the function ``ont`` of
:mod:`mowl.representation.library.hierarchy_methods`. The hand-written functions it was once
are the reference in :mod:`mowl.baselines.nn.lm.ont`.

Scaling :math:`k_r`: Eq. 4 includes it, but the reference implementation computes
:math:`k_r = \\exp(\\cdot)` and does not apply it ("not used in final output"), and the
published models treat it as 1. :class:`RoleTransformation` follows the code by default
(``use_scaling=False``); with ``use_scaling=True`` it applies Eq. 4 literally.
"""

from mowl.nn.lm.hit import HiTModule
from mowl.representation.library import hierarchy_methods as HM
from mowl.representation.library.hierarchy_relations import RoleTransformation
from mowl.representation.regions import Region

__all__ = ["OnTModule", "RoleTransformation"]


class OnTModule(HiTModule):
    """HiT module with OnT's role transformations and logical losses.

    :param encoder: An encoder module (see :mod:`mowl.nn.lm.encoders`).
    :param use_roles: ``False`` gives the ablation OnT(w/o r) (Section 5.1): only the
        hierarchy loss of Eq. 5 is used (the geometry has no relation model).
    :param conjunction_weight: Weight of :math:`L_\\sqcap` (1 in the paper).
    :param existential_weight: Weight of :math:`L_r` (1 in the paper).
    :param role_pairing: See :class:`RoleTransformation`.
    :param role_scaling: See :class:`RoleTransformation`.

    The remaining parameters are those of :class:`~mowl.nn.lm.hit.HiTModule`; OnT's
    defaults (Section 5.1) are :math:`\\alpha = 3.0`, :math:`\\beta = 0.5`.
    """

    def __init__(self, encoder, curvature=None, clustering_margin=3.0, centripetal_margin=0.5,
                 clustering_weight=1.0, centripetal_weight=1.0, use_roles=True,
                 conjunction_weight=1.0, existential_weight=1.0, role_pairing="half",
                 role_scaling=False):
        self.use_roles = use_roles
        self.conjunction_weight = conjunction_weight
        self.existential_weight = existential_weight
        role_model = RoleTransformation(int(encoder.dim), role_pairing, role_scaling)
        self._adopt(HM.ont(encoder=encoder, curvature=curvature,
                           clustering_margin=clustering_margin,
                           centripetal_margin=centripetal_margin,
                           clustering_weight=clustering_weight,
                           centripetal_weight=centripetal_weight,
                           role_model=role_model if use_roles else None))
        # the transformation exists (and can be read) without roles too, as the ablation
        # OnT(w/o r) ignores it
        self.role_model = role_model

    def role_image(self, filler, role_texts):
        """:math:`f_r(x_D)` for filler embeddings ``(n, d)`` and role verbalisations."""
        return self.role_model(filler, self.embed(role_texts), self.c)

    def existential_loss(self, exists, roles, filler, negative_1, negative_2):
        """:math:`L_r` (Eq. 6). With regions (``exists``, the relation ids or names and the
        fillers' regions) it is :meth:`TextGeometry.existential_loss`, ``(B, 1)``; with
        verbalised :math:`\\exists r.D`, :math:`V(r)` and :math:`V(D)` (lists of texts) and
        embeddings as negatives it gives one value per row ``(n,)``."""
        if isinstance(exists, Region):
            return super().existential_loss(exists, roles, filler, negative_1, negative_2)
        x_exists, x_filler = self.embed(exists), self.embed(filler)
        image = self.role_image(x_filler, roles)
        a, b, n1, n2 = self._regions(x_exists, image, negative_1, negative_2)
        forward = self.hierarchy_loss(a, b, n1)
        backward = self.hierarchy_loss(b, a, n2)
        return (0.5 * (forward + backward)).squeeze(-1)

    def conjunction_loss(self, conjunction, conjuncts, negatives):
        """:math:`L_\\sqcap` (Eq. 7). With regions it is
        :meth:`TextGeometry.conjunction_loss`, ``(B, 1)``; with the verbalised conjunctions,
        a list (one entry per conjunct position) of lists of texts and embeddings as
        negatives it gives one value per row ``(n,)``."""
        if isinstance(conjunction, Region):
            return super().conjunction_loss(conjunction, conjuncts, negatives)
        x = self._regions(self.embed(conjunction))[0]
        parts = [self._regions(self.embed(t))[0] for t in conjuncts]
        negs = self._regions(*negatives)
        return super().conjunction_loss(x, parts, negs).squeeze(-1)
