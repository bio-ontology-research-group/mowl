"""PrismEL model class."""
from __future__ import annotations

import torch as th

from mowl.base_models.elmodel import EmbeddingELModel
from mowl.geometric.normal_forms import BOTTOM, TOP
from mowl.nn import PrismModule
from mowl.nn.el.structural import DefinitionTable, ExistenceTargets


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
    objective : {"axiom", "structural"}, default "axiom"
        Training objective mode (``"axiom"`` or ``"structural"``). Under
        ``objective="structural"``, :meth:`compute_loss` averages the already-squared
        structural GCI losses directly (without an outer squaring) while keeping
        negative sampling enabled for ``gci2`` and ``object_property_assertion``.
    max_depth : int, default 8
        Maximum definition level :math:`\\text{max\\_level}` passed to
        :meth:`~mowl.nn.el.structural.DefinitionTable.from_normal_forms` under
        ``objective="structural"``.
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
        existence_weight=1.0,
        size_reg=1e-3,
        canonical_model=None,
    ):
        if float(anchor_weight) < 0:
            raise ValueError(f"anchor_weight must be >= 0, got {anchor_weight!r}")
        if objective not in ("axiom", "structural"):
            raise ValueError(
                f"objective must be 'axiom' or 'structural', got {objective!r}"
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
        self.reg_norm = reg_norm
        self.init_boxes = init_boxes
        self.anchor_weight = float(anchor_weight)
        self.objective = objective
        self.max_depth = max_depth
        self.existence_weight = float(existence_weight)
        self.size_reg = float(size_reg)
        self.canonical_model = canonical_model
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
            objective=self.objective,
            max_depth=self.max_depth,
            existence_weight=self.existence_weight,
            size_reg=self.size_reg,
        ).to(self.device)
        if self.init_boxes is not None:
            self.init_boxes.apply_to_module(
                self.module, class_index_dict=self.class_index_dict
            )

    def _dataset_rows(self, key: str, cols: int) -> th.Tensor:
        """Return ``self.training_datasets[key][:]`` on CPU sliced to ``cols`` columns, or an empty ``(0, cols)`` LongTensor."""
        ds = self.training_datasets.get(key)
        if ds is None or len(ds) == 0:
            return th.zeros((0, cols), dtype=th.long)
        rows = ds[:].detach().cpu()
        if rows.ndim == 2 and rows.shape[1] > cols:
            rows = rows[:, :cols]
        return rows.to(dtype=th.long)

    def train(self, epochs, validate_every=1, epoch_callback=None):
        """Train the model, building structural tables first when ``objective='structural'``.

        Guarantees:
        - When ``objective="structural"``, compiles a :class:`~mowl.nn.el.structural.DefinitionTable`
          (with ``k=1``, ``max_components=2**30``, ``max_level=self.max_depth``) and
          :class:`~mowl.nn.el.structural.ExistenceTargets` from CPU dataset tensors
          and installs them on :attr:`module` via :meth:`~mowl.nn.PrismModule.set_structural_tables`
          before delegating to ``super().train(...)``.
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
            Optional callback ``(epoch, train_loss, val_loss) -> Optional[bool]``.

        Returns
        -------
        int
            Result returned by :meth:`~mowl.base_models.elmodel.EmbeddingELModel.train`.
        """
        if self.objective == "structural":
            n_classes = len(self.class_index_dict)
            top_idx = self.class_index_dict.get(TOP)
            bottom_idx = self.class_index_dict.get(BOTTOM)
            index_to_iri = {idx: iri for iri, idx in self.class_index_dict.items()}

            gci0 = self._dataset_rows("gci0", 2)
            gci0_bot = self._dataset_rows("gci0_bot", 2)
            gci1 = self._dataset_rows("gci1", 3)
            gci1_bot = self._dataset_rows("gci1_bot", 2)

            table = DefinitionTable.from_normal_forms(
                gci0=gci0,
                gci1=gci1,
                n_classes=n_classes,
                k=1,
                max_components=2**30,
                top_idx=top_idx,
                bottom_idx=bottom_idx,
                enabled=True,
                max_level=self.max_depth,
            )
            targets = ExistenceTargets.from_normal_forms(
                gci0_bot=gci0_bot,
                gci1=gci1,
                gci1_bot=gci1_bot,
                n_classes=n_classes,
                top_idx=top_idx,
                bottom_idx=bottom_idx,
                index_to_iri=index_to_iri,
                cm=self.canonical_model,
            )
            self.module.set_structural_tables(table, targets, top_idx=top_idx)

        return super().train(
            epochs=epochs,
            validate_every=validate_every,
            epoch_callback=epoch_callback,
        )

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
            Current 1-based training epoch index.

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

        Under ``objective="structural"``, ``pos_scores`` are already squared hinge
        losses, so returns ``pos_scores.mean()`` directly (avoiding double squaring).
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
