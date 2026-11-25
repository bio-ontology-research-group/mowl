"""The module of HiT (He, Yuan, Chen, Horrocks, "Language Models as Hierarchy Encoders",
NeurIPS 2024, Section 3): a text encoder whose outputs are points of the Poincare ball of
curvature :math:`c = 1/d`, read by the measure ``distance and depth`` and trained with the
triplet loss :math:`L_{HiT}` (Eq. 5).

It is a combination of parts: the class model :class:`~mowl.representation.library.hierarchy_classes.TextPoints`
(encoder and ball), the measure :class:`~mowl.representation.library.hierarchy_measures.DistanceDepth`
and the objective of a hierarchy method; the method is the function ``hit`` of
:mod:`mowl.representation.library.hierarchy_methods`. The losses are those of the measure
(clustering Eq. 3, centripetal Eq. 4, their sum Eq. 5, the score Eq. 6). The hand-written
functions they were once are the reference in :mod:`mowl.baselines.nn.lm.hit`.
"""

from mowl.representation.library import hierarchy_methods as HM
from mowl.representation.library.hierarchy_geometry import TextGeometry
from mowl.representation.library.hierarchy_regions import ManifoldPoint
from mowl.representation.regions import Region

__all__ = ["HiTModule"]


class HiTModule(TextGeometry):
    """A text encoder whose outputs are points of a Poincare ball, with the HiT losses.

    :param encoder: An encoder module (see :mod:`mowl.representation.library.text_encoders`).
    :param curvature: Curvature parameter :math:`c`; default :math:`1/d` (HiT, Section 3).
    :param clustering_margin: :math:`\\alpha` (Eq. 3).
    :param centripetal_margin: :math:`\\beta` (Eq. 4).
    :param clustering_weight: Weight of the clustering loss (1 in the paper).
    :param centripetal_weight: Weight of the centripetal loss (1 in the paper).
    """

    def __init__(self, encoder, curvature=None, clustering_margin=5.0, centripetal_margin=0.1,
                 clustering_weight=1.0, centripetal_weight=1.0):
        self._adopt(HM.hit(encoder=encoder, curvature=curvature,
                           clustering_margin=clustering_margin,
                           centripetal_margin=centripetal_margin,
                           clustering_weight=clustering_weight,
                           centripetal_weight=centripetal_weight))

    def _adopt(self, geometry):
        """Become the combination of parts of ``geometry`` (the parts are shared)."""
        TextGeometry.__init__(
            self, geometry.class_model, geometry.relation_model, geometry.measure,
            geometry.objective, geometry.clustering_margin, geometry.centripetal_margin,
            geometry.clustering_weight, geometry.centripetal_weight)

    @property
    def encoder(self):
        return self.class_model.encoder

    @property
    def dim(self):
        return self.class_model.embed_dim

    @property
    def c(self):
        """Curvature parameter of the ball."""
        return self.class_model.manifold.c

    def embed(self, texts):
        """Encode a list of texts into points of the ball, shape ``(len(texts), d)``."""
        return self.encoder(list(texts))

    def _regions(self, *xs):
        return [x if isinstance(x, Region) else ManifoldPoint(x, self.class_model.manifold)
                for x in xs]

    def hierarchy_loss(self, child, parent, negative):
        """:math:`L_\\prec` (Eq. 5): for regions, ``(B, 1)`` (:meth:`TextGeometry.hierarchy_loss`);
        for embeddings, one value per row ``(n,)``."""
        if isinstance(child, Region):
            return super().hierarchy_loss(child, parent, negative)
        return super().hierarchy_loss(*self._regions(child, parent, negative)).squeeze(-1)

    def forward(self, child_texts, parent_texts, negative_texts):
        """Per-row HiT loss of triplets of texts."""
        return self.hierarchy_loss(self.embed(child_texts), self.embed(parent_texts),
                                   self.embed(negative_texts))

    def score(self, child, parent, centri_weight):
        """Subsumption score (Eq. 6) of embeddings; higher is better."""
        measure = type(self.measure)(centri_weight)
        a, b = self._regions(child, parent)
        return -measure.inside(a, b).squeeze(-1)
