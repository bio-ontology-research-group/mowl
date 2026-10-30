"""MultiBoxEL model class."""
from __future__ import annotations

from mowl.base_models.elmodel import EmbeddingELModel
from mowl.nn import MultiBoxELModule


class MultiBoxEL(EmbeddingELModel):
    """Multi-component Box\\ :math:`^2`\\ EL embedding model.

    Represents every ontology class as a union of ``n_components`` (:math:`k`)
    axis-parallel boxes in :math:`\\mathbb{R}^d` under Box\\ :math:`^2`\\ EL
    inclusion, gated pairwise intersection, and role semantics.

    Guarantees:
    - **Exact k = 1 reduction:** With ``n_components=1`` and
      ``component_reg=0.0``, the underlying :class:`~mowl.nn.MultiBoxELModule`
      computes the exact same per-GCI losses and regularisation loss (to within
      ``float32`` tolerance ``1e-6``) as :class:`~mowl.models.BoxSquaredEL`.
    - **Exact initialisation and anchor hook:** When ``init_boxes`` is provided,
      :meth:`init_module` applies :meth:`~mowl.geometric.init.ExactBoxInit.apply_to_module`
      after module construction; when ``anchor_weight > 0``,
      :meth:`get_regularization_loss` adds
      ``anchor_weight * init_boxes.anchor_loss(self.module)``.

    Non-guarantees:
    - Gradient-based training is non-convex and is not guaranteed to reach a
      zero-loss global optimum from arbitrary random initialisations.
    - Role axiom training (``load_role_axioms=True``) is not supported by
      :class:`~mowl.nn.MultiBoxELModule`.

    Parameters
    ----------
    dataset : mowl.datasets.Dataset
        mOWL dataset to use for training and evaluation.
    embed_dim : int, default 50
        The embedding dimension :math:`d`.
    n_components : int, default 2
        Number of box components :math:`k \\ge 1` per ontology class.
    margin : float, default 0.02
        Inclusion margin :math:`\\gamma`.
    reg_norm : int, default 1
        Regularisation norm order (retained for interface parity with
        :class:`~mowl.models.BoxSquaredEL`).
    learning_rate : float, default 0.001
        Adam learning rate.
    batch_size : int, default 32768
        Training batch size.
    delta : float, default 2.5
        Negative-sampling distance target :math:`\\delta`.
    reg_factor : float, default 0.2
        Weight for the Box²EL bump regularisation term.
    component_reg : float, default 0.0
        Weight for the anti-collapse pairwise component centre regulariser.
    num_negs : int, default 4
        Number of negative samples per positive (retained for interface parity
        with :class:`~mowl.models.BoxSquaredEL`).
    model_filepath : str, optional
        Path for saving and loading model checkpoints.
    device : str, default 'cpu'
        PyTorch device string.
    neg_sampling_gcis : list of str, optional
        Explicit list of GCI names for negative sampling, or ``None`` to use
        the module's ``neg_capable_gcis``.
    init_boxes : mowl.geometric.init.ExactBoxInit, optional
        Optional exact-model box initialisation applied in :meth:`init_module`.
        Rows are matched by IRI against :attr:`class_index_dict`; rows of the
        auxiliary classes introduced by normalisation keep their random
        initialisation.
    anchor_weight : float, default 0.0
        Non-negative weight multiplying ``init_boxes.anchor_loss(self.module)``
        in :meth:`get_regularization_loss` when ``init_boxes`` is provided.
    component_aggregation : str, default "hard"
        Component aggregation mode: ``"hard"`` or ``"soft"``.
    tau_start : float, default 1.0
        Initial soft-assignment temperature :math:`\\tau_{\\text{start}} > 0`.
    tau_end : float, default 0.01
        Final soft-assignment temperature :math:`\\tau_{\\text{end}} > 0` when
        ``tau_anneal=True``.
    tau_anneal : bool, default False
        Whether to anneal ``module.tau`` geometrically from ``tau_start`` to
        ``tau_end`` over the training epochs in :meth:`on_epoch_start`.
    diversity_reg : float, default 0.0
        Weight for the pairwise component volume overlap regulariser.
    tau : float, optional
        Optional alias for ``tau_start`` when setting a fixed temperature.
    """

    def __init__(self,
                 dataset,
                 embed_dim=50,
                 n_components=2,
                 margin=0.02,
                 reg_norm=1,
                 learning_rate=0.001,
                 batch_size=4096 * 8,
                 delta=2.5,
                 reg_factor=0.2,
                 component_reg=0.0,
                 num_negs=4,
                 model_filepath=None,
                 device='cpu',
                 neg_sampling_gcis=None,
                 init_boxes=None,
                 anchor_weight=0.0,
                 component_aggregation="hard",
                 tau_start=1.0,
                 tau_end=0.01,
                 tau_anneal=False,
                 diversity_reg=0.0,
                 tau=None,
                 ):
        if float(anchor_weight) < 0:
            raise ValueError(f"anchor_weight must be >= 0, got {anchor_weight!r}")
        if component_aggregation not in ("hard", "soft"):
            raise ValueError(
                f"component_aggregation must be 'hard' or 'soft', got {component_aggregation!r}"
            )
        eff_tau_start = float(tau) if tau is not None else float(tau_start)
        eff_tau_end = float(tau_end)
        if eff_tau_start <= 0.0 or eff_tau_end <= 0.0:
            raise ValueError(
                f"tau_start and tau_end must be > 0, got tau_start={eff_tau_start!r}, tau_end={eff_tau_end!r}"
            )
        if float(diversity_reg) < 0.0:
            raise ValueError(f"diversity_reg must be >= 0, got {diversity_reg!r}")
        super().__init__(dataset, embed_dim, batch_size, extended=True,
                         model_filepath=model_filepath, device=device,
                         learning_rate=learning_rate,
                         neg_sampling_gcis=neg_sampling_gcis)

        self.n_components = n_components
        self.margin = margin
        self.reg_norm = reg_norm
        self.delta = delta
        self.reg_factor = reg_factor
        self.component_reg = component_reg
        self.num_negs = num_negs
        self.init_boxes = init_boxes
        self.anchor_weight = float(anchor_weight)
        self.component_aggregation = component_aggregation
        self.tau_start = eff_tau_start
        self.tau_end = eff_tau_end
        self.tau_anneal = bool(tau_anneal)
        self.tau = eff_tau_start
        self.diversity_reg = float(diversity_reg)
        self._loaded = False
        self.extended = False
        self.init_module()

    def init_module(self):
        """Initialise :class:`~mowl.nn.MultiBoxELModule` and apply ``init_boxes`` if present."""
        self.module = MultiBoxELModule(
            len(self.class_index_dict),
            len(self.object_property_index_dict),
            len(self.individual_index_dict),
            embed_dim=self.embed_dim,
            n_components=self.n_components,
            gamma=self.margin,
            delta=self.delta,
            reg_factor=self.reg_factor,
            component_reg=self.component_reg,
            component_aggregation=self.component_aggregation,
            tau=self.tau,
            diversity_reg=self.diversity_reg,
        ).to(self.device)
        if self.init_boxes is not None:
            self.init_boxes.apply_to_module(
                self.module, class_index_dict=self.class_index_dict)

    def on_epoch_start(self, epoch: int, epochs: int) -> None:
        """Update ``self.module.tau`` according to the geometric annealing schedule when ``tau_anneal=True``.

        Guarantees:
        - When ``tau_anneal=True`` and ``epochs > 1``, sets
          ``module.tau = tau_start * (tau_end / tau_start) ** (epoch / (epochs - 1))``
          so epoch ``0`` uses ``tau_start`` and epoch ``epochs - 1`` uses ``tau_end``.
        - When ``tau_anneal=False``, leaves ``module.tau`` unchanged.

        Non-guarantees:
        - When ``epochs <= 1`` and ``tau_anneal=True``, ``module.tau`` remains ``tau_start``.
        """
        if self.tau_anneal:
            if epochs > 1:
                tau = self.tau_start * (
                    (self.tau_end / self.tau_start) ** (epoch / (epochs - 1))
                )
            else:
                tau = self.tau_start
            self.tau = float(tau)
            self.module.tau = float(tau)

    def get_regularization_loss(self):

        """Return module regularisation loss plus weighted anchor loss when active."""
        loss = super().get_regularization_loss()
        if self.init_boxes is not None and self.anchor_weight > 0:
            loss = loss + self.anchor_weight * self.init_boxes.anchor_loss(self.module)
        return loss
