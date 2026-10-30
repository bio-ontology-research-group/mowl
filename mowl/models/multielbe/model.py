"""MultiELBE model class."""
from __future__ import annotations

import torch as th

from mowl.base_models.elmodel import EmbeddingELModel
from mowl.nn import MultiELBEModule


class MultiELBE(EmbeddingELModel):
    """Multi-component ELBE embedding model.

    Represents every ontology class as a union of ``n_components`` (:math:`k`)
    axis-parallel boxes in :math:`\\mathbb{R}^d` under ELBE inclusion, gated
    pairwise intersection, and translation role semantics.

    Guarantees:
    - **Exact k = 1 reduction:** With ``n_components=1``,
      ``component_aggregation="hard"``, and ``component_reg=0.0``, the
      underlying :class:`~mowl.nn.MultiELBEModule` computes the exact same
      per-GCI losses and MSE training loss (to within ``float32`` tolerance
      ``1e-6``) as :class:`~mowl.models.ELBE`.
    - **Exact initialisation and anchor hook:** When ``init_boxes`` is provided,
      :meth:`init_module` applies :meth:`~mowl.geometric.init.ExactBoxInit.apply_to_module`
      after module construction; when ``anchor_weight > 0``,
      :meth:`get_regularization_loss` adds
      ``anchor_weight * init_boxes.anchor_loss(self.module)``.

    Non-guarantees:
    - Gradient-based training is non-convex and is not guaranteed to reach a
      zero-loss global optimum from arbitrary random initialisations.
    - Role axiom training (``load_role_axioms=True``) is not supported by
      :class:`~mowl.nn.MultiELBEModule`.

    Parameters
    ----------
    dataset : mowl.datasets.Dataset
        mOWL dataset to use for training and evaluation.
    embed_dim : int, default 50
        The embedding dimension :math:`d`.
    margin : float, default 0.1
        ELBE inclusion margin (positive margin tightens).
    n_components : int, default 1
        Number of box components :math:`k \\ge 1` per ontology class.
    component_reg : float, default 0.0
        Weight for the anti-collapse pairwise component centre regulariser.
    component_aggregation : str, default "hard"
        Component aggregation mode: ``"hard"`` or ``"soft"``.
    tau : float, default 1.0
        Soft-assignment temperature stored on the module.
    diversity_reg : float, default 0.0
        Diversity regularisation weight stored on the module.
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
    init_boxes : mowl.geometric.init.ExactBoxInit, optional
        Optional exact-model box initialisation applied in :meth:`init_module`.
    anchor_weight : float, default 0.0
        Non-negative weight multiplying ``init_boxes.anchor_loss(self.module)``
        in :meth:`get_regularization_loss` when ``init_boxes`` is provided.
    """

    def __init__(
        self,
        dataset,
        embed_dim=50,
        margin=0.1,
        n_components=1,
        component_reg=0.0,
        component_aggregation="hard",
        tau=1.0,
        diversity_reg=0.0,
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
        if component_aggregation not in ("hard", "soft"):
            raise ValueError(
                f"component_aggregation must be 'hard' or 'soft', got {component_aggregation!r}"
            )
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
        self.n_components = n_components
        self.component_reg = component_reg
        self.component_aggregation = component_aggregation
        self.tau = float(tau)
        self.diversity_reg = float(diversity_reg)
        self.reg_norm = reg_norm
        self.init_boxes = init_boxes
        self.anchor_weight = float(anchor_weight)
        self._loaded = False
        self.extended = False
        self._mse_criterion = th.nn.MSELoss()
        self.init_module()

    def init_module(self):
        """Initialise :class:`~mowl.nn.MultiELBEModule` and apply ``init_boxes`` if present."""
        self.module = MultiELBEModule(
            len(self.class_index_dict),
            len(self.object_property_index_dict),
            len(self.individual_index_dict),
            embed_dim=self.embed_dim,
            margin=self.margin,
            n_components=self.n_components,
            component_reg=self.component_reg,
            component_aggregation=self.component_aggregation,
            tau=self.tau,
            diversity_reg=self.diversity_reg,
        ).to(self.device)
        if self.init_boxes is not None:
            self.init_boxes.apply_to_module(
                self.module, class_index_dict=self.class_index_dict
            )

    def get_regularization_loss(self):
        """Return module regularisation loss plus weighted anchor loss when active."""
        loss = super().get_regularization_loss()
        if self.init_boxes is not None and self.anchor_weight > 0:
            loss = loss + self.anchor_weight * self.init_boxes.anchor_loss(self.module)
        return loss

    def compute_loss(self, pos_scores, neg_scores=None):
        """Compute MSE loss for MultiELBE (matching ELBE).

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
