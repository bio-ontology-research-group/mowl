"""PrismEL model class."""
from __future__ import annotations

from typing import Optional, Tuple

import torch as th

from mowl.base_models.elmodel import EmbeddingELModel
from mowl.nn import PrismModule
from mowl.nn.el.structural import (
    DefinitionTable,
    ExistenceTargets,
    build_structural_tables,
)


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
    - Under ``objective="structural"``, only the box GCI losses (``gci0``,
      ``gci0_bot``, ``gci1``, ``gci1_bot``) are squared hinges; the role and
      assertion scores (``gci2``, ``gci3``, ``gci3_bot``, ``class_assertion``,
      ``object_property_assertion``) are unsquared norms/hinges that enter
      linearly in :meth:`compute_loss`.

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
    objective : {"axiom", "structural"}, default "axiom"
        Training objective mode (``"axiom"`` or ``"structural"``). Under
        ``objective="structural"``, :meth:`compute_loss` averages ``pos_scores``
        directly (without an outer squaring): the box structural GCI losses
        (``gci0``, ``gci0_bot``, ``gci1``, ``gci1_bot``) are already squared
        hinges, whereas the prism role and assertion scores (``gci2``, ``gci3``,
        ``gci3_bot``, ``class_assertion``, ``object_property_assertion``) are
        unsquared norms/hinges entering linearly, while keeping negative sampling
        enabled for ``gci2`` and ``object_property_assertion``.
    max_depth : int, default 8
        Maximum definition level :math:`\\text{max\\_level}` passed to
        :meth:`~mowl.nn.el.structural.DefinitionTable.from_normal_forms` under
        ``objective="structural"``.
    structural_definitions : bool, default True
        Whether ``objective="structural"`` compiles acyclic binary conjunction
        definitions :math:`A \\equiv X \\sqcap Y` (including through role
        auxiliaries :math:`A \\sqsubseteq \\exists r.C \\sqsubseteq \\text{Aux}`)
        via :class:`~mowl.nn.el.structural.DefinitionTable`.
    existence_weight : float, default 1.0
        Non-negative weight multiplying :func:`~mowl.nn.el.structural.existence_loss`
        in :meth:`~mowl.nn.PrismModule.structural_extra_loss` under
        ``objective="structural"``.
    size_reg : float, default 1e-3
        Non-negative weight multiplying the primitive prism size regulariser
        ``mean([|r|, 2|o|])`` in :meth:`~mowl.nn.PrismModule.structural_extra_loss`
        under ``objective="structural"``.
    canonical_model : mowl.geometric.completion.CanonicalModel, optional, default None
        Optional pre-computed :class:`~mowl.geometric.completion.CanonicalModel`
        used to determine semantic satisfiability in
        :meth:`~mowl.nn.el.structural.ExistenceTargets.from_normal_forms` under
        ``objective="structural"``.
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
        objective="axiom",
        max_depth=8,
        structural_definitions=True,
        existence_weight=1.0,
        size_reg=1e-3,
        canonical_model=None,
    ):
        if (
            isinstance(anchor_weight, bool)
            or not isinstance(anchor_weight, (int, float))
            or float(anchor_weight) < 0
        ):
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
        self.objective = objective
        self.max_depth = max_depth
        self.structural_definitions = structural_definitions
        self.existence_weight = existence_weight
        self.size_reg = size_reg
        self.canonical_model = canonical_model
        self._structural_tables: Optional[
            Tuple[DefinitionTable, ExistenceTargets, Optional[int]]
        ] = None
        self._loaded = False
        self.extended = False
        self._mse_criterion = th.nn.MSELoss()
        self.init_module()

    def init_module(self):
        """Initialise :class:`~mowl.nn.PrismModule`, apply ``init_boxes`` if present, and re-attach cached structural tables."""
        self.module = PrismModule(
            len(self.class_index_dict),
            len(self.object_property_index_dict),
            len(self.individual_index_dict),
            embed_dim=self.embed_dim,
            margin=self.margin,
            objective=self.objective,
            max_depth=self.max_depth,
            structural_definitions=self.structural_definitions,
            existence_weight=self.existence_weight,
            size_reg=self.size_reg,
        ).to(self.device)
        self.structural_definitions = self.module.structural_definitions
        self.existence_weight = self.module.existence_weight
        self.size_reg = self.module.size_reg
        if self.init_boxes is not None:
            self.init_boxes.apply_to_module(
                self.module, class_index_dict=self.class_index_dict
            )
        if self._structural_tables is not None:
            table, targets, top_idx = self._structural_tables
            self.module.set_structural_tables(table, targets, top_idx=top_idx)

    def train(self, epochs, validate_every=1, epoch_callback=None):
        """Train the model, building structural tables first when ``objective='structural'``.

        Guarantees:
        - When ``objective="structural"``, compiles a :class:`~mowl.nn.el.structural.DefinitionTable`
          (with ``k=1``, ``max_components=2**30``, ``enabled=self.structural_definitions``,
          ``max_level=self.max_depth``) and :class:`~mowl.nn.el.structural.ExistenceTargets`
          via :func:`~mowl.nn.el.structural.build_structural_tables`, caches them on
          the model so :meth:`init_module`, :meth:`load_best_model`, and
          :meth:`from_pretrained` preserve them, and installs them on :attr:`module`
          via :meth:`~mowl.nn.PrismModule.set_structural_tables` before delegating
          to ``super().train(...)``.
        - Raises :class:`ValueError` if any required training GCI tensor has
          fewer columns than expected by its normal form.
        - When ``objective="axiom"``, delegates directly to ``super().train(...)``.

        Non-guarantees:
        - Gradient-based training is non-convex and is not guaranteed to reach a
          global minimum from arbitrary random initialisations.

        Parameters
        ----------
        epochs : int
            Number of training epochs.
        validate_every : int, default 1
            Validation frequency in epochs.
        epoch_callback : callable, optional
            Optional callback ``epoch_callback(epoch, model)`` invoked at the end
            of each 0-based epoch.

        Returns
        -------
        None
            Trains the model in place and returns ``None``.
        """
        if self.objective == "structural":
            self._structural_tables = build_structural_tables(
                self,
                k=1,
                max_components=2**30,
                enabled=self.structural_definitions,
                max_level=self.max_depth,
                cm=self.canonical_model,
            )
            table, targets, top_idx = self._structural_tables
            self.module.set_structural_tables(table, targets, top_idx=top_idx)

        return super().train(
            epochs=epochs,
            validate_every=validate_every,
            epoch_callback=epoch_callback,
        )

    def load_best_model(self):
        """Reload the best checkpoint and re-attach structural tables when ``objective='structural'``."""
        if self.objective == "structural" and self._structural_tables is None:
            self._structural_tables = build_structural_tables(
                self,
                k=1,
                max_components=2**30,
                enabled=self.structural_definitions,
                max_level=self.max_depth,
                cm=self.canonical_model,
            )
        super().load_best_model()

    def from_pretrained(self, model):
        """Load pretrained weights and ensure structural tables are attached when ``objective='structural'``."""
        super().from_pretrained(model)
        if self.objective == "structural":
            if self._structural_tables is None:
                self._structural_tables = build_structural_tables(
                    self,
                    k=1,
                    max_components=2**30,
                    enabled=self.structural_definitions,
                    max_level=self.max_depth,
                    cm=self.canonical_model,
                )
            table, targets, top_idx = self._structural_tables
            self.module.set_structural_tables(table, targets, top_idx=top_idx)

    def additional_training_loss(self, epoch):
        """Return ``module.structural_extra_loss()`` when ``objective='structural'``, else ``None``.

        Guarantees:
        - Returns a scalar ``torch.Tensor`` when ``objective="structural"`` and
          ``None`` when ``objective="axiom"``.

        Non-guarantees:
        - Does not depend on ``epoch``.

        Parameters
        ----------
        epoch : int
            Current 0-based training epoch index.

        Returns
        -------
        Optional[torch.Tensor]
            Extra structural training loss or ``None``.
        """
        if self.objective == "structural":
            return self.module.structural_extra_loss()
        return None

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
        """Compute per-GCI batch loss for PrismEL.

        Under ``objective="structural"``, returns ``pos_scores.mean()`` directly
        (without an outer squaring): the box structural GCI losses (``gci0``,
        ``gci0_bot``, ``gci1``, ``gci1_bot``) are already squared hinges, whereas
        the prism role and assertion scores (``gci2``, ``gci3``, ``gci3_bot``,
        ``class_assertion``, ``object_property_assertion``) are unsquared
        norms/hinges that enter linearly.
        Under ``objective="axiom"``, returns ``MSE(pos_scores.mean(), 0)`` matching ELBE.
        In both modes, if ``neg_scores`` is given, adds ``MSE(neg_scores.mean(), 1)``.
        """
        pos_mean = th.mean(pos_scores)
        if self.objective == "structural":
            loss = pos_mean
        else:
            loss = self._mse_criterion(
                pos_mean, th.zeros_like(pos_mean, requires_grad=False)
            )

        if neg_scores is not None:
            neg_mean = th.mean(neg_scores)
            loss = loss + self._mse_criterion(
                neg_mean, th.ones_like(neg_mean, requires_grad=False)
            )

        return loss
