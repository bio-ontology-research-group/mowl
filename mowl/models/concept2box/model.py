r"""Concept2Box on an ontology viewed as a two-view knowledge graph."""
import logging

from mowl.base_models.elmodel import EmbeddingELModel
from mowl.nn.el.concept2box.module import Concept2BoxModule

logger = logging.getLogger(__name__)


class Concept2Box(EmbeddingELModel):
    r"""Concept2Box [huang2023]_: concepts are Gumbel boxes, entities are points.

    The paper learns from two knowledge graphs, a concept graph (ontology view) and an
    entity graph (instance view), joined by links from entities to concepts. An ontology
    gives these three parts directly:

    * ontology view: ``C ⊑ D`` and ``C ⊑ ∃r.D`` (normal forms ``gci0`` and ``gci2``);
    * instance view: role assertions ``r(a, b)``;
    * cross-view links: class assertions ``C(a)``.

    The three losses are those of the paper, summed (Equation 11). The generic training
    loop of :class:`~mowl.base_models.EmbeddingELModel` is used, with PyKEEN-free negative
    sampling by corrupting the right-hand class or individual. Only these four forms are
    trained: ``gci1``, ``gci3``, the bottom forms and role axioms are not used (a warning
    names the ones the ontology contains). See :class:`~mowl.nn.Concept2BoxModule` for the
    differences from the paper (no BERT, pointwise instead of pairwise hinge).

    :param dataset: mOWL dataset.
    :param embed_dim: Dimension of boxes and vectors.
    :param beta: Gumbel scale.
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

    def __init__(self, dataset, embed_dim=32, beta=1.0, alpha=0.5, kge_margin=0.3,
                 cross_margin=0.15, instance_weight=1.0, cross_weight=1.0,
                 learning_rate=0.005, batch_size=4096, model_filepath=None, device="cpu",
                 neg_sampling_gcis=None, gci2_reading="axiom"):
        super().__init__(dataset, embed_dim, batch_size, extended=True,
                         model_filepath=model_filepath, device=device,
                         learning_rate=learning_rate, neg_sampling_gcis=neg_sampling_gcis)
        self.beta, self.alpha = beta, alpha
        self.kge_margin, self.cross_margin = kge_margin, cross_margin
        self.instance_weight, self.cross_weight = instance_weight, cross_weight
        self.gci2_reading = gci2_reading
        self._warned = False
        self.init_module()

    def init_module(self):
        self.module = Concept2BoxModule(
            len(self.class_index_dict),
            len(self.object_property_index_dict),
            nb_inds=len(self.individual_index_dict) or None,
            embed_dim=self.embed_dim, beta=self.beta, alpha=self.alpha,
            kge_margin=self.kge_margin, cross_margin=self.cross_margin,
            instance_weight=self.instance_weight, cross_weight=self.cross_weight,
            gci2_reading=self.gci2_reading,
        ).to(self.device)

    @property
    def training_datasets(self):
        datasets = super().training_datasets
        usable = {k: v for k, v in datasets.items() if k in self.SUPPORTED_GCIS}
        skipped = sorted(k for k, v in datasets.items()
                         if k not in self.SUPPORTED_GCIS and len(v) > 0)
        if skipped and not self._warned:
            logger.warning("Concept2Box does not use the axiom forms %s.", skipped)
            self._warned = True
        return usable
