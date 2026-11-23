r"""Concept2Box on an ontology viewed as a two-view knowledge graph."""
import logging

from mowl.base_models.elmodel import EmbeddingELModel
from mowl.models.composed.model import ComposedEL

logger = logging.getLogger(__name__)


class Concept2Box(ComposedEL):
    r"""Concept2Box [huang2023]_: concepts are Gumbel boxes, entities are points.

    The paper learns from two knowledge graphs, a concept graph (ontology view) and an
    entity graph (instance view), joined by links from entities to concepts. An ontology
    gives these three parts directly:

    * ontology view: ``C ⊑ D`` and ``C ⊑ ∃r.D`` (normal forms ``gci0`` and ``gci2``);
    * instance view: role assertions ``r(a, b)``;
    * cross-view links: class assertions ``C(a)``.

    The three losses are those of the paper, summed (Equation 11). The model is the
    combination of parts called ``"concept2box"``
    (:func:`~mowl.representation.library.shapes_methods.concept2box`): Gumbel boxes in the
    context of a relation, translations of points, the volume ratio, the point-to-box
    distance and the point distance as measures. It is a :class:`~mowl.models.ComposedEL`, so
    any part can be swapped; the hand-written loss module of the earlier implementation is
    :class:`mowl.baselines.models.concept2box.model.Concept2Box`. The generic training loop
    of :class:`~mowl.base_models.EmbeddingELModel` is used, with PyKEEN-free negative
    sampling by corrupting the right-hand class or individual. Only these four forms are
    trained: ``gci1``, ``gci3``, the bottom forms and role axioms are not used (a warning
    names the ones the ontology contains). See :class:`~mowl.nn.Concept2BoxModule` for the
    differences from the paper (no BERT, pointwise instead of pairwise hinge).

    :param dataset: mOWL dataset.
    :param embed_dim: Dimension of boxes and vectors.
    :param beta: Gumbel scale (default ``0.25 / embed_dim``; ``1.0`` in the earlier
        module, with which the ontology-view loss has no gradient beyond a few tens of
        dimensions).
    :param init_scale: Standard deviation of the initial vectors (default
        ``1 / embed_dim``; ``0.5`` in the earlier module, with which the sigmoid of the
        cross-view loss is saturated beyond about 30 dimensions). See
        :func:`~mowl.representation.library.shapes_methods.concept2box`.
    :param alpha: Weight of the inside distance.
    :param kge_margin: Margin of the instance view.
    :param cross_margin: Margin of the cross view.
    :param instance_weight: :math:`\lambda_1`.
    :param cross_weight: :math:`\lambda_2`.
    :param gci2_reading: ``"axiom"`` (default): ``C ⊑ ∃r.D`` puts the box of ``C`` inside
        the box of ``D`` in the context of ``r``; ``"triple"``: the paper's triple
        ``(C, r, D)``, the box of ``D`` inside that of ``C``
        (see :class:`~mowl.nn.Concept2BoxModule`).
    """

    axiom_scoring = "module"
    scored_forms = ("gci0", "gci2", "class_assertion", "object_property_assertion")
    SUPPORTED_GCIS = scored_forms

    def __init__(self, dataset, embed_dim=32, beta=None, alpha=0.5, kge_margin=0.3,
                 cross_margin=0.15, instance_weight=1.0, cross_weight=1.0,
                 learning_rate=0.005, batch_size=4096, model_filepath=None, device="cpu",
                 neg_sampling_gcis=None, gci2_reading="axiom", init_scale=None):
        self._warned = False
        super().__init__(dataset, method="concept2box", embed_dim=embed_dim,
                         batch_size=batch_size, model_filepath=model_filepath, device=device,
                         learning_rate=learning_rate, neg_sampling_gcis=neg_sampling_gcis,
                         beta=beta, alpha=alpha, kge_margin=kge_margin,
                         cross_margin=cross_margin, instance_weight=instance_weight,
                         cross_weight=cross_weight, gci2_reading=gci2_reading,
                         init_scale=init_scale)

    @property
    def beta(self):
        """The Gumbel scale in use (``0.25 / embed_dim`` unless one was given)."""
        return self.module.class_model.beta
    alpha = property(lambda self: self.method_options["alpha"])
    kge_margin = property(lambda self: self.method_options["kge_margin"])
    cross_margin = property(lambda self: self.method_options["cross_margin"])
    instance_weight = property(lambda self: self.method_options["instance_weight"])
    cross_weight = property(lambda self: self.method_options["cross_weight"])
    gci2_reading = property(lambda self: self.method_options["gci2_reading"])

    @property
    def training_datasets(self):
        usable = super().training_datasets
        datasets = EmbeddingELModel.training_datasets.fget(self)
        skipped = sorted(k for k, v in datasets.items()
                         if k not in usable and len(v) > 0)
        if skipped and not self._warned:
            logger.warning("Concept2Box does not use the axiom forms %s.", skipped)
            self._warned = True
        return usable
