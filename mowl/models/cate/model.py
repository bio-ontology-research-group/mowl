"""CatE: order embeddings of the lattice of ALC concept descriptions [zhapa2023]_."""

import logging
from functools import partial

import torch as th

from mowl.models.cate.common import CatEBase, UnseenConceptError
from mowl.nn.graph.cate.module import CatEModule
from mowl.training.graph import (EdgeCorruption, EpochBatches, GroupedMarginLoss,
                                 HeadRestrictedHinge, KnownRowMask, ValidationPatience)
from mowl.training.loop import RegimeTrained, Rows
from mowl.training.schedules import CyclicLearningRate

logger = logging.getLogger(__name__)

__all__ = ["CatE", "UnseenConceptError"]


class CatE(RegimeTrained, CatEBase):
    """CatE (Zhapa-Camacho and Hoehndorf, *Lattice-based ALC ontology embeddings with
    saturation*, arXiv:2305.07163v3; NeSy 2024 for the version without saturation).

    **Method.** (i) The training ontology is turned into the preorder
    :math:`(\\tilde C, \\sqsubseteq)` of its concept descriptions and their
    sub-descriptions (Section 3) by
    :class:`~mowl.projection.categorical.model.CategoricalProjector`, wrapped in
    :class:`~mowl.projection.categorical.lattice.CatELattice`; (ii) optionally the lattice
    is partially saturated (Section 3.1, :mod:`mowl.projection.categorical.saturation`);
    (iii) every element gets a vector, and every morphism :math:`A \\sqsubseteq B` is
    trained to satisfy :math:`f(A) \\preceq f(B)` in the product order with the score
    :math:`s(A, B) = \\lVert \\max(0, f(A) - f(B)) \\rVert_2` (Eq. 6) and the margin loss of
    Eq. 7, whose negatives replace :math:`B` by an element drawn uniformly from the
    lattice (:class:`~mowl.nn.graph.cate.CatEModule`). The model is a combination of parts
    (``model.module.parts``), defined once in
    :func:`mowl.representation.library.hierarchy_methods.cate`: free vectors in the product
    order as the class model, lookup of the compound elements, the order violation as the
    measure; the loss of Eq. 7 is :func:`~mowl.representation.library.hierarchy_objectives.
    margin_loss`.

    **Scoring.** A query is scored by :math:`s` between lattice elements (lower is
    better): :math:`C \\sqsubseteq D` by :math:`s(C, D)`, :math:`C(a)` by
    :math:`s(\\{a\\}, C)`, :math:`C \\sqsubseteq \\exists r.D` by :math:`s(C, \\exists r.D)`.
    Both sides must be elements of the lattice. CatE cannot embed other concept
    descriptions (paper, Section 5); scoring one raises
    :class:`~mowl.projection.categorical.lattice.UnseenConceptError`. For ranking
    :math:`C \\sqsubseteq \\exists r.D'` over candidates :math:`D'` the elements
    :math:`\\exists r.D'` must be added before training (``existentials``), as the paper
    does for the PPI task (Section 4.4).

    **Training (from the paper and the reference code).** Adam; triangular cyclic
    learning rate between ``learning_rate / 10`` and ``learning_rate`` with 20 steps per
    half cycle, stepped per batch; weight decay :math:`10^{-4}`; Xavier-uniform
    initialisation; validation every ``validate_every`` epochs by the raw mean
    reciprocal rank of the validation axioms, keeping the best parameters and stopping
    after ``early_stopping_patience`` validations without improvement. The paper states
    only "Adam with a Cyclic Learning Rate scheduler" (Section 4.1); bounds, step size,
    weight decay, initialisation and model selection are taken from the reference code
    (``src/cat.py``).

    **Deviations.** (a) ``negative_aggregation="per_negative"`` (default) is Eq. 7; the
    reference code averages the scores of the ``num_negs`` corruptions before the hinge
    (``"mean"``). Eq. 7 sums over corruptions that are *not* morphisms; by default
    (``filter_negatives=True``) corruptions that are morphisms are discarded, while the
    reference code keeps them (``filter_negatives=False``). Keeping them makes a zero
    loss impossible: every element has the morphism :math:`X \\sqsubseteq \\top`.
    (b) Nodes are identified by unambiguous names; the reference trained on the
    projector's string rendering, which drops parentheses, so distinct descriptions
    could share a vector. (c) ``complete_negation=True`` adds the full negation diagram of
    Figure 2c for complex operands (see
    :func:`~mowl.projection.categorical.saturation.complete_negation`); ``False`` gives the
    projector's lattice. (d) The reference PPI training adds a second loss for the
    morphisms of a batch whose head is a protein, with negatives drawn among the
    :math:`\\exists
    interacts.p` elements; this is not described in the paper and is optional
    here (interaction_loss, part of the cate-code-training regime). (e) Every class and
    individual of the dataset (including those that occur only in the validation or test
    ontology) is an element, with only :math:`\\bot \\sqsubseteq X \\sqsubseteq \\top` (and
    the negation diagram for classes) if it is not in the training ontology.

    :param dataset: mOWL dataset; the lattice is built from ``dataset.ontology``.
    :type dataset: :class:`mowl.datasets.Dataset`
    :param embed_dim: Embedding dimension.
    :type embed_dim: int
    :param margin: :math:`\\gamma` of Eq. 7.
    :type margin: float
    :param num_negs: Corruptions per morphism.
    :type num_negs: int
    :param learning_rate: Upper bound of the cyclic learning rate.
    :type learning_rate: float
    :param batch_size: Morphisms per batch.
    :type batch_size: int
    :param p: Norm of Eq. 6.
    :type p: int
    :param weight_decay: Adam weight decay.
    :type weight_decay: float
    :param cyclic_lr: Use the cyclic learning rate (otherwise a constant one).
    :type cyclic_lr: bool
    :param negative_aggregation: ``"per_negative"`` (Eq. 7) or ``"mean"`` (reference
        code), see :func:`~mowl.representation.library.hierarchy_objectives.margin_loss`.
    :type negative_aggregation: str
    :param filter_negatives: Discard corruptions that are morphisms of the lattice (or
        reflexive), as Eq. 7 sums over non-morphisms only; ``False`` keeps them, as the
        reference code does.
    :type filter_negatives: bool
    :param saturation_steps: Iterations of the saturation rules (0: none, ``None``: until
        a fixed point). ``1`` is the paper's S1.
    :type saturation_steps: int or None
    :param saturation_rules: Rule names of
        :data:`mowl.projection.categorical.saturation.RULES` (default: S1).
    :type saturation_rules: tuple
    :param transitive_closure: Add the transitive closure: ``True`` for the paper's Tr as
        the reference code computes it (not through :math:`\\bot`, :math:`\\top`),
        ``"full"`` for the complete closure, under which a zero loss is attainable.
    :type transitive_closure: bool or str
    :param complete_negation: See deviation (c).
    :type complete_negation: bool
    :param obvious_morphisms: Keep :math:`\\bot \\sqsubseteq X` and
        :math:`X \\sqsubseteq \\top` for complex :math:`X` (the paper keeps them for
        every element). The released training graphs of the ORE1, GO and FoodOn
        benchmarks drop them; ``False`` matches those graphs (the recipes do so).
    :type obvious_morphisms: bool
    :param existentials: Elements :math:`\\exists r.D` to add: a dict from role IRI to
        ``"classes"``, ``"evaluation_classes"`` or an iterable of class IRIs; ``None``;
        or ``"auto"`` (default): if the dataset defines ``get_evaluation_property()`` (the
        PPI and GDA datasets), :math:`\\exists r.D` for that role and every tail
        evaluation class, otherwise none.
    :param interaction_loss: Add the PPI training term of the reference code
        (``src/cat_ppi_evaluator.py``, ``train``): for every morphism of the batch whose
        head is one of the dataset's head evaluation classes (the proteins), the mean
        order violation of the morphism plus a hinge on the violation of a corruption
        whose tail is drawn uniformly among the elements :math:`\\exists r.p` of the
        dataset's evaluation property. Not described in the paper; the published Table 4
        was produced with it.
    :param early_stopping_patience: Validations without improvement before stopping
        (``None``: never stop early).
    :type early_stopping_patience: int or None
    :param model_filepath: Where the parameters are saved.
    :type model_filepath: str, optional
    :param device: Torch device.
    :type device: str
    """

    # ----------------------------------------------------------------- module
    def init_module(self):
        """Create the :class:`~mowl.nn.graph.cate.CatEModule`."""
        lat = self.lattice
        module = CatEModule(len(lat), self.embed_dim, p=self.p, margin=self.margin)
        module.link_margin(self)
        if lat.existential_to_id:
            keys = list(lat.existential_to_id.items())
            module.set_existentials([k[0] for k, _ in keys], [k[1] for k, _ in keys],
                                    [v for _, v in keys])
        self._module = module.to(self.device)

    # --------------------------------------------------------------- training
    save_always = True

    @property
    def default_regimes(self):
        """The regimes that make the training of CatE (Section 4.1 and the options of the
        model): minibatches over the morphisms (:class:`~mowl.training.EpochBatches`),
        corruptions of the superconcept (:class:`~mowl.training.EdgeCorruption`), the mask
        of corruptions that are morphisms (:class:`~mowl.training.KnownRowMask`, if
        ``filter_negatives``), the loss of Eq. 7 (:class:`~mowl.training.GroupedMarginLoss`),
        the cyclic learning rate (:class:`~mowl.training.CyclicLearningRate`, if
        ``cyclic_lr``) and validation with early stopping
        (:class:`~mowl.training.ValidationPatience`) when ``train`` is given
        ``validate_every``."""
        return self._regimes(self.negative_aggregation, self.filter_negatives,
                             self.interaction_loss)

    def _regimes(self, aggregation, filter_negatives, interaction_loss):
        forms = ("edge",)
        parts = [partial(EpochBatches, self.batch_size, forms),
                 partial(EdgeCorruption, self.num_negs, (1,), None, forms)]
        if filter_negatives:
            parts.append(partial(KnownRowMask, forms))
        parts.append(partial(GroupedMarginLoss, None, aggregation, forms))
        if interaction_loss:
            parts.append(partial(HeadRestrictedHinge, self._interaction_ids, None, None,
                                 "edge"))
        if self.cyclic_lr:
            parts.append(partial(CyclicLearningRate, None, self.learning_rate, 20,
                                 range_ratio=10))
        parts.append(partial(ValidationPatience, lambda m: m.validation_mrr(), None,
                             self.early_stopping_patience, True, "validation_history",
                             lambda m: m.dataset.validation is not None
                             and len(m.validation_queries()) > 0))
        return tuple(parts)

    @property
    def training_datasets(self):
        """The morphisms of the lattice as rows ``(A, B)`` of the form ``edge``."""
        rows = getattr(self, "_rows", None)
        if rows is None or rows.data.device != th.device(self.device):
            rows = self._rows = Rows(self.lattice.edges.to(self.device))
        return {"edge": rows}

    @property
    def num_nodes(self):
        return len(self.lattice)

    def get_optimizer(self):
        return th.optim.Adam(self.module.parameters(), lr=self.learning_rate,
                             weight_decay=self.weight_decay)

    def compute_loss(self, pos):
        """The loss of Eq. 7 for a batch of morphisms ``pos`` (shape ``(b, 2)``) as training
        computes it, plus the interaction term if :attr:`interaction_loss` is set."""
        return self.batch_loss({"edge": pos})

# ---------------------------------------------------------------------------------------
# Benchmark runner support
# ---------------------------------------------------------------------------------------

def _register_benchmark_family():
    """Register a :class:`mowl.benchmarks.runner.ModelFamily` for :class:`CatE`, so that
    ``python -m mowl.benchmarks run <benchmark> --method cate`` can build and train it
    (the built-in families handle only ``EmbeddingELModel`` subclasses)."""
    try:
        from mowl.benchmarks import runner
    except ImportError:  # pragma: no cover
        return
    if any(getattr(f, "name", "") == "cate" for f in runner._FAMILIES):
        return

    class CatEFamily(runner.ModelFamily):
        name = "cate"

        def matches(self, cls):
            return isinstance(cls, type) and issubclass(cls, CatE)

        def build(self, cls, dataset, model_kwargs, device, model_filepath):
            kwargs = dict(model_kwargs)
            kwargs.setdefault("device", device)
            kwargs.setdefault("model_filepath", model_filepath)
            return cls(dataset, **kwargs)

        def train(self, model, epochs, protocol, validate_every=None, select_best=True):
            validate = select_best and model.dataset.validation is not None
            if validate and validate_every is None:
                validate_every = max(1, epochs // 10)
            model.train(epochs=epochs, validate_every=validate_every if validate else None)
            validated = validate and bool(getattr(model, "validation_history", []))
            return {"selected": "best validation MRR" if validated else "last epoch",
                    "validation": getattr(model, "validation_history", []),
                    "lattice": {k: v for k, v in model.lattice.stats.items()
                                if isinstance(v, (int, float))}}

    runner.register_family(CatEFamily(), first=True)


_register_benchmark_family()
