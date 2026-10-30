"""PrismEL model class."""
from __future__ import annotations

import torch as th

from mowl.base_models.elmodel import EmbeddingELModel
from mowl.nn import PrismModule


class PrismEL(EmbeddingELModel):
    """Prism (:math:`\\text{triangle} \\times \\text{box}`) :math:`\\mathcal{EL}` embedding model.

    Represents every ontology class :math:`C` as a prism
    :math:`P_C = T_C \\times B_C \\subset \\mathbb{R}^2 \\times \\mathbb{R}^{d-2}`
    with :math:`2d - 1` scalar parameters per class following Federico, Leemhuis
    & Kutz (DL 2026, §4.1 Definition 4 and Proposition 5), trained with ELBE's
    MSE loss on the support-coordinate GCI losses of :class:`~mowl.nn.PrismModule`.

    Guarantees:
    - :meth:`count_class_parameters` returns :math:`2d - 1` per ontology class
      (``2`` for ``tri_center``, ``1`` for ``tri_inradius``, ``d - 2`` for
      ``box_center``, and ``d - 2`` for ``box_offset``).
    - **Exact initialisation and anchor hook:** When ``init_boxes`` is provided,
      :meth:`init_module` applies :meth:`~mowl.geometric.init.ExactBoxInit.apply_to_module`
      after module construction; the initialised prism of each class contains its
      initialisation box (the equilateral triangle in coordinates ``0..1`` with
      centre ``c[:2]`` and inradius :math:`\\max(o_0, o_1) \\sqrt{3}`, which
      suffices to contain the rectangle ``[c_0 - o_0, c_0 + o_0] x [c_1 - o_1, c_1 + o_1]``
      (not minimal), and coordinates ``2..d`` copy the box part). When ``anchor_weight > 0``,
      :meth:`get_regularization_loss` adds
      ``anchor_weight * init_boxes.anchor_loss(self.module)``.

    Non-guarantees:
    - Gradient-based training is non-convex and is not guaranteed to reach a
      zero-loss global optimum from arbitrary random initialisations.
    - Role axiom training (``load_role_axioms=True``) is not supported by
      :class:`~mowl.nn.PrismModule`.

    Parameters
    ----------
    dataset : mowl.datasets.Dataset
        mOWL dataset to use for training and evaluation.
    embed_dim : int, default 50
        The ambient embedding dimension :math:`d \\ge 3`.
    margin : float, default 0.1
        Inclusion margin (positive margin tightens constraints following ELBE).
    init_boxes : mowl.geometric.init.ExactBoxInit, optional
        Optional exact-model box initialisation applied in :meth:`init_module`.
        Rows are matched by IRI against :attr:`class_index_dict`; rows of the
        auxiliary classes introduced by normalisation keep their random
        initialisation.
    anchor_weight : float, default 0.0
        Non-negative weight multiplying ``init_boxes.anchor_loss(self.module)``
        in :meth:`get_regularization_loss` when ``init_boxes`` is provided.
    reg_norm : int, default 1
        Regularisation norm order (retained for interface parity with
        :class:`~mowl.models.ELBE`).
    learning_rate : float, default 0.001
        Adam learning rate.
    batch_size : int, default 32768
        Training batch size.
    model_filepath : str, optional
        Path for saving and loading model checkpoints.
    device : str, default 'cpu'
        PyTorch device string.
    neg_sampling_gcis : list of str, optional
        Explicit list of GCI names for negative sampling, or ``None`` to use
        the module's ``neg_capable_gcis``.
    """

    def __init__(
        self,
        dataset,
        embed_dim=50,
        margin=0.1,
        init_boxes=None,
        anchor_weight=0.0,
        reg_norm=1,
        learning_rate=0.001,
        batch_size=4096 * 8,
        model_filepath=None,
        device="cpu",
        neg_sampling_gcis=None,
    ):
        if float(anchor_weight) < 0:
            raise ValueError(f"anchor_weight must be >= 0, got {anchor_weight!r}")
        super().__init__(
            dataset,
            embed_dim,
            batch_size,
            extended=True,
            model_filepath=model_filepath,
            device=device,
            learning_rate=learning_rate,
            neg_sampling_gcis=neg_sampling_gcis,
        )

        self.margin = margin
        self.reg_norm = reg_norm
        self.init_boxes = init_boxes
        self.anchor_weight = float(anchor_weight)
        self._loaded = False
        self.extended = False
        self._mse_criterion = th.nn.MSELoss()
        self.init_module()

    def init_module(self):
        """Initialise :class:`~mowl.nn.PrismModule` and apply ``init_boxes`` if present."""
        self.module = PrismModule(
            len(self.class_index_dict),
            len(self.object_property_index_dict),
            len(self.individual_index_dict),
            embed_dim=self.embed_dim,
            margin=self.margin,
        ).to(self.device)
        if self.init_boxes is not None:
            self.init_boxes.apply_to_module(
                self.module, class_index_dict=self.class_index_dict
            )

    def count_class_parameters(self) -> int:
        """Return the number of scalar parameters per ontology class (:math:`2d - 1`).

        Sums the per-class widths of ``tri_center`` (2), ``tri_inradius`` (1),
        ``box_center`` (:math:`d - 2`), and ``box_offset`` (:math:`d - 2`).
        """
        return int(
            self.module.tri_center.weight.shape[1]
            + self.module.tri_inradius.weight.shape[1]
            + self.module.box_center.weight.shape[1]
            + self.module.box_offset.weight.shape[1]
        )

    def get_regularization_loss(self):
        """Return module regularisation loss plus weighted anchor loss when active."""
        loss = super().get_regularization_loss()
        if self.init_boxes is not None and self.anchor_weight > 0:
            loss = loss + self.anchor_weight * self.init_boxes.anchor_loss(self.module)
        return loss

    def compute_loss(self, pos_scores, neg_scores=None):
        """Compute MSE loss for PrismEL (matching ELBE).

        Positive samples should have scores close to 0.
        Negative samples should have scores close to 1.
        """
        pos_mean = th.mean(pos_scores)
        loss = self._mse_criterion(
            pos_mean, th.zeros_like(pos_mean, requires_grad=False)
        )

        if neg_scores is not None:
            neg_mean = th.mean(neg_scores)
            loss += self._mse_criterion(
                neg_mean, th.ones_like(neg_mean, requires_grad=False)
            )

        return loss
