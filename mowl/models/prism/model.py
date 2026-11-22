"""PrismEL model class."""
from __future__ import annotations

from mowl.models.composed.model import ComposedEL
from mowl.nn.el.structural import build_structural_tables
from mowl.representation.library.shapes_training import SquaredMeanLoss


class PrismEL(ComposedEL):
    """Prism (:math:`\\text{triangle} \\times \\text{box}`) :math:`\\mathcal{EL}` embedding model.

    The model is the combination of parts called ``"prism"``
    (:func:`~mowl.representation.library.shapes_methods.prism`): prisms as classes,
    translations as relations, individuals as points, support-coordinate bounds as measure,
    ELBE's mean-squared loss as training regime. It is a :class:`~mowl.models.ComposedEL`,
    so any part can be swapped. The hand-written loss module of the earlier implementation is
    :class:`mowl.baselines.models.prism.model.PrismEL`.

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
    gci2_negative_triangle : {"overlap", "inradius"}, default "overlap"
        See :class:`~mowl.nn.PrismModule`.
    empty_box_reading : {"signed", "absolute"}, default "signed"
        See :class:`~mowl.nn.PrismModule`.
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
        gci2_negative_triangle="overlap",
        canonical_model=None,
        empty_box_reading="signed",
    ):
        if (
            isinstance(anchor_weight, bool)
            or not isinstance(anchor_weight, (int, float))
            or float(anchor_weight) < 0
        ):
            raise ValueError(f"anchor_weight must be >= 0, got {anchor_weight!r}")
        if objective not in ("axiom", "structural"):
            raise ValueError(f"objective must be 'axiom' or 'structural', got {objective!r}.")
        self.reg_norm = reg_norm
        self.init_boxes = init_boxes
        self.anchor_weight = float(anchor_weight)
        self.canonical_model = canonical_model
        super().__init__(
            dataset,
            method="prism",
            embed_dim=embed_dim,
            margin=margin,
            learning_rate=learning_rate,
            batch_size=batch_size,
            model_filepath=model_filepath,
            device=device,
            neg_sampling_gcis=neg_sampling_gcis,
            reading=objective,
            max_depth=max_depth,
            structural_definitions=structural_definitions,
            existence_weight=existence_weight,
            size_reg=size_reg,
            gci2_negative_triangle=gci2_negative_triangle,
            empty_box_reading=empty_box_reading,
        )
        self.extended = False

    @property
    def objective(self):
        """``"axiom"`` or ``"structural"``: the reading of the method (``reading`` of
        :func:`~mowl.representation.library.shapes_methods.prism`)."""
        return getattr(self, "method_options", {}).get("reading", "axiom")

    @objective.setter
    def objective(self, value):
        # (ComposedEL keeps an Objective object under this name; the method names its reading)
        pass

    @property
    def max_depth(self):
        return self.method_options["max_depth"]

    @property
    def structural_definitions(self):
        return self.module.structural_definitions if hasattr(
            self.module, "structural_definitions") else self.method_options[
                "structural_definitions"]

    @property
    def existence_weight(self):
        return self.method_options["existence_weight"]

    @property
    def size_reg(self):
        return self.method_options["size_reg"]

    @property
    def gci2_negative_triangle(self):
        return self.method_options["gci2_negative_triangle"]

    @property
    def empty_box_reading(self):
        return self.method_options["empty_box_reading"]

    def init_module(self):
        """Build the prism module from its parts, apply ``init_boxes`` if present, and attach
        the structural tables when ``objective='structural'``."""
        super().init_module()
        if self.init_boxes is not None:
            self.init_boxes.apply_to_module(
                self.module, class_index_dict=self.class_index_dict
            )

    def _attach_structural_tables(self):
        """As :class:`~mowl.models.ComposedEL`, with the canonical model of the dataset
        (``canonical_model``) for the semantic satisfiability of the existence targets."""
        module = self.module
        if not getattr(module, "wants_structural_tables", False):
            self._structural_tables = None
            return
        tables = getattr(self, "_structural_tables", None)
        if tables is None:
            tables = build_structural_tables(self, cm=self.canonical_model,
                                             **module.structural_table_options)
            self._structural_tables = tables
        table, targets, top_idx = tables
        module.set_structural_tables(table, targets, top_idx)

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
        """The loss of one batch of one axiom form, that of the regime
        :class:`~mowl.representation.library.shapes_training.SquaredMeanLoss` of the method.

        Under ``objective="axiom"``: ``MSE(pos.mean(), 0)`` as ELBE does. Under
        ``objective="structural"``: ``pos.mean()`` (the box scores are squared hinges
        already, the role and assertion scores enter linearly). In both, if ``neg_scores`` is
        given, ``MSE(neg.mean(), 1)`` is added.
        """
        loss = SquaredMeanLoss(0.0, 1.0, self.objective != "structural")
        return loss.combine(None, pos_scores.reshape(-1),
                            None if neg_scores is None else neg_scores.reshape(-1))
