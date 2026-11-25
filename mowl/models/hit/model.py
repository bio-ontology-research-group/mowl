"""HiT: Hierarchy Transformer encoders (He, Yuan, Chen, Horrocks, NeurIPS 2024)."""

import logging
from functools import partial

import torch as th

from mowl.models.hit.common import DEFAULT_CENTRI_GRID, HiTBase  # noqa: F401
from mowl.representation.library.hierarchy_regions import ManifoldPoint
from mowl.training.graph import EpochBatches, ValidationPatience
from mowl.training.loop import RegimeTrained, Rows
from mowl.training.text import LinearWarmupDecay

logger = logging.getLogger(__name__)


class HiT(RegimeTrained, HiTBase):
    """Hierarchy Transformer encoder (HiT) [he2024hit]_.

    The model is the combination of parts that the method ``hit`` defines
    (:func:`mowl.representation.library.hierarchy_methods.hit`): the class model is the text
    encoder with the Poincare ball it is read in (the encoder is a part: any module that maps a
    list of strings to vectors, see :mod:`mowl.representation.library.text_encoders`), the
    measure is
    ``distance and depth`` (:math:`d_c + \\lambda\\,\\Delta\\lVert\\cdot\\rVert`), the losses
    are the triplet losses of its geometry
    (:class:`~mowl.representation.library.hierarchy_geometry.TextGeometry`), and there is no
    relation model. The reference implementation that the combination is tested against is
    ``mowl.baselines.models.hit``.

    A text encoder (a sentence-transformers model with mean pooling and no normalisation
    layer) is re-trained so that the embeddings of class labels, read as points of the
    Poincaré ball of radius :math:`\\sqrt{d}` (curvature :math:`c = 1/d`), are clustered
    by the hyperbolic clustering loss and ordered by the hyperbolic centripetal loss
    (Section 3, Eqs. 3-5). A subsumption :math:`C \\sqsubseteq D` is scored by
    :math:`-(d_c(x_C, x_D) + \\lambda(\\lVert x_D\\rVert_c - \\lVert x_C\\rVert_c))` (Eq. 6).

    Training data: the atomic subsumptions :math:`A \\sqsubseteq B` of
    ``dataset.ontology`` (``taxonomy="asserted"``, the edges of the training hierarchy) or
    the direct subsumptions inferred by ELK (``taxonomy="direct"``, as the paper builds
    hierarchies from OWL ontologies, Appendix A). For every positive :math:`(e, e^+)`,
    ``num_negatives`` negative parents are sampled (Section 4.1): random classes
    :math:`e^-` with :math:`e \\not\\sqsubseteq e^-` under the closed-world assumption
    (not in the reflexive-transitive closure of the training edges), or, with
    ``hard_negatives=True``, siblings of :math:`e` first, completed with random ones.

    Optimisation as in the paper (Appendix B): AdamW, learning rate :math:`10^{-5}`, 500
    linear warm-up steps followed by linear decay, batch size 256. After every epoch, if
    ``dataset.validation`` has atomic subsumptions, the weight :math:`\\lambda` is chosen
    on them and the parameters with the best validation MRR are kept (the paper selects
    the model each epoch on validation F1 with a tuned threshold; mOWL uses the ranking
    MRR, as OnT does for HiT, Yang et al. 2025, Section 5.1).

    Class labels are ``rdfs:label`` annotations (fallback: IRI fragment), see
    :mod:`mowl.text.verbalise`. Complex class expressions (when scoring axioms that are
    not atomic) are verbalised with the OnT rules; HiT itself is defined for named
    entities only.

    :param dataset: A :class:`mowl.datasets.Dataset`.
    :param encoder: Hugging Face model name (loaded with sentence-transformers, an optional
        dependency), a ``SentenceTransformer`` object, or any :class:`torch.nn.Module`
        mapping a list of strings to a ``(n, d)`` tensor (for example
        :class:`mowl.representation.library.text_encoders.BagOfWordsEncoder`).
    :param verbaliser: A :class:`mowl.text.Verbaliser`; default: one built from
        ``dataset.ontology`` with ``labels``, ``verbalisation_style`` and
        ``label_fallback``.
    :param labels: ``{iri: label}`` overriding the ontology's labels.
    :param verbalisation_style: ``"paper"``, ``"release"`` or ``"deeponto"``.
    :param label_fallback: ``"fragment"``, ``"camel_case"`` or ``None``.
    :param clustering_margin: :math:`\\alpha` (Eq. 3); the paper uses 5.0.
    :param centripetal_margin: :math:`\\beta` (Eq. 4); the paper uses 0.1.
    :param clustering_weight: Weight of the clustering loss.
    :param centripetal_weight: Weight of the centripetal loss.
    :param curvature: Curvature parameter :math:`c` (default :math:`1/d`).
    :param learning_rate: AdamW learning rate.
    :param weight_decay: AdamW weight decay (0, the Hugging Face default).
    :param warmup_steps: Linear warm-up steps (then linear decay to 0).
    :param batch_size: Training triplets per step.
    :param num_negatives: Negative parents per positive (10 in the paper).
    :param hard_negatives: Sample siblings as negatives first.
    :param taxonomy: ``"asserted"`` or ``"direct"`` (see above).
    :param centri_weight: Fixed :math:`\\lambda`; ``None`` selects it on validation data
        (0 if there is none).
    :param centri_weight_grid: Grid searched for :math:`\\lambda`.
    :param encode_batch_size: Batch size when encoding without gradients.
    :param score_dtype: Precision of the ranking scores, ``"float64"`` (default) or
        ``"float32"`` (as the reference code; much faster on GPUs with slow float64).
    :param seed: Seed of the negative sampling (default: Python's global generator).
    :param device: Torch device.
    :param model_filepath: Where the best parameters are saved.
    """


    #: Default margins of the method (subclasses change them).
    default_margins = (5.0, 0.1)

    axiom_scoring = "method"
    scored_forms = ("gci0",)


    default_validate_every = 1

    # ------------------------------------------------------------------ set-up
    def init_module(self):
        """The module is the combination of parts that the method ``hit`` defines
        (:func:`mowl.representation.library.hierarchy_methods.hit`): a text encoder and a
        Poincare ball as the class model, the distance-and-depth measure, the triplet
        losses."""
        from mowl.nn.lm.hit import HiTModule
        return HiTModule(self.encoder, curvature=self.curvature,
                         clustering_margin=self.clustering_margin,
                         centripetal_margin=self.centripetal_margin,
                         clustering_weight=self.clustering_weight,
                         centripetal_weight=self.centripetal_weight)

    # -- what depends on the module of this side ----------------------------------------
    def _embed_points(self, texts):
        return self.module.classes(texts).centre

    def _cost(self, sub, sup, lam):
        manifold = self.module.class_model.manifold
        return self.module.measure.inside(ManifoldPoint(sub, manifold), ManifoldPoint(sup, manifold),
                                          centri_weight=lam).squeeze(-1)

    def _ranking_scores(self, query, candidates, candidates_are_children, lam):
        manifold = self.module.class_model.manifold
        return self.module.measure.ranking_scores(
            ManifoldPoint(query, manifold), ManifoldPoint(candidates, manifold), candidates_are_children, lam,
            self.score_dtype)

    def _score_pair(self, x, lam):
        manifold = self.module.class_model.manifold
        cost = self.module.measure.inside(ManifoldPoint(x[:1], manifold), ManifoldPoint(x[1:], manifold),
                                          centri_weight=lam)
        return float(-cost.reshape(()))

    # -- training --------------------------------------------------------------------------
    @property
    def default_regimes(self):
        """The regimes that make the training of HiT (Appendix B of the paper): minibatches of a
        fresh shuffle of the triplets per epoch (:class:`~mowl.training.EpochBatches`, with the
        random generator of the model), AdamW with a linear warm-up and decay of the learning rate
        (:class:`~mowl.training.text.LinearWarmupDecay`) and the selection of the best epoch
        by the validation MRR (:class:`~mowl.training.ValidationPatience`), which also chooses
        the weight :math:`\\lambda` of the score."""
        forms = ("triplet",)
        return (partial(EpochBatches, self.batch_size, forms, self.rng),
                partial(LinearWarmupDecay, self.warmup_steps),
                partial(ValidationPatience, self._validation_metric, None, None, True,
                        "validation_history", lambda m: m._validation_rows() is not None,
                        self._best_epoch, False))

    def get_optimizer(self):
        params = [p for p in self.module.parameters() if p.requires_grad]
        return th.optim.AdamW(params, lr=self.learning_rate, weight_decay=self.weight_decay)

    def triplet_rows(self, triplets):
        """The triplets of texts as rows ``(child, parent, negative)`` of indices of the texts
        (:meth:`texts`)."""
        table = self.__dict__.setdefault("_text_ids", {})
        listing = self.__dict__.setdefault("_text_list", [])
        rows = []
        for triplet in triplets:
            row = []
            for text in triplet:
                if text not in table:
                    table[text] = len(listing)
                    listing.append(text)
                row.append(table[text])
            rows.append(row)
        return th.tensor(rows, dtype=th.long, device=self.device).reshape(-1, 3)

    def texts(self, ids):
        """The texts of the indices ``ids`` of :meth:`triplet_rows`."""
        return [self._text_list[i] for i in ids.tolist()]

    @property
    def training_datasets(self):
        """The training triplets, drawn anew at every call of ``train``, as rows of the form
        ``triplet``."""
        self._validated = False
        return {"triplet": Rows(self.triplet_rows(self.training_triplets()))}

    def score_rows(self, form, rows, neg=False):
        """The loss :math:`L_{HiT}` of every triplet of ``rows`` (Eq. 5)."""
        g = self.module
        negatives = g.classes(self.texts(rows[:, 2])).centre
        child, parent = g.classes(self.texts(rows[:, 0])), g.classes(self.texts(rows[:, 1]))
        return g.hierarchy_loss(child, parent, ManifoldPoint(negatives, g.class_model.manifold)
                                ).reshape(-1)

    def on_epoch_start(self, step, steps):
        self._invalidate()

    def after_optimizer_step(self):
        super().after_optimizer_step()
        self._invalidate()

    def record_epoch(self, epoch, mean_loss):
        self.history.append({"epoch": epoch, "loss": mean_loss})

    def _validation_metric(self, model):
        mrr, lam = self.validate()
        self._last_validation = (mrr, lam)
        self.history[-1].update({"valid_mrr": mrr, "centri_weight": lam})
        return mrr

    def _best_epoch(self, model, mrr, epoch):
        lam = self._last_validation[1]
        self.best_validation = {"mrr": mrr, "centri_weight": lam, "epoch": epoch - 1}
        self._validated = True

    def finish_training(self):
        self._invalidate()
        if self._validated:
            if not self._fixed_centri_weight:
                self.centri_weight = self.best_validation["centri_weight"]
        elif self.centri_weight is None:
            self.centri_weight = 0.0
