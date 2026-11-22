"""MultiELBE neural module."""
from __future__ import annotations


from mowl.representation.library.structural_points import POINT_READINGS as _VALID_OBJECTIVES
from mowl.representation.geometry import ComposedGeometry
from mowl.representation.library.multibox_methods import multielbe
from mowl.representation.library.multibox_parts import UnionClasses
from mowl.representation.library.multibox_modules import (
    UnionModule, check_union_options, released_names)


@released_names(
    state=("class_embed", "class_offset", "rel_embed", "ind_embed", "ind_offset", "elem_embed"),
    class_embed="class_model.centres", class_center="class_model.centres",
    class_offset="class_model.offsets", rel_embed="relation_model.vectors",
    ind_embed="individual_model.centres", ind_offset="individual_model.offsets",
    elem_embed="point_terms.elem_embed")
class MultiELBEModule(UnionModule, ComposedGeometry):
    """Multi-component ELBE neural module.

    The module is the combination of parts that MultiELBE is
    (:func:`~mowl.representation.library.multibox_methods.multielbe`): every ontology class
    :math:`C` is a union of :math:`k = \\text{n\\_components}` axis-parallel boxes in
    :math:`\\mathbb{R}^d`, stored in ``class_embed`` and ``class_offset`` of shape
    ``(nb_ont_classes, k * d)`` and viewed as ``(B, k, d)``. Roles are single translation
    vectors in ``rel_embed`` of shape ``(nb_rels, d)``, and optional individuals are single
    boxes in ``ind_embed`` and ``ind_offset`` of shape ``(nb_inds, d)``, as in
    :class:`~mowl.nn.el.elbe.module.ELBEModule`. The module is a
    :class:`~mowl.representation.ComposedGeometry` (class model, relation model, measure,
    objective and regularisers are the parts of MultiELBE); models saved under the names of the
    tables load. ``module.objective`` is the :class:`~mowl.representation.Objective` of the
    parts, the name of the reading is ``module.reading``.

    Guarantees:

    - **Exact k = 1 reduction (identical to ELBEModule at equal margin):** When
      ``n_components == 1``, ``component_aggregation == "hard"``,
      ``component_reg == 0.0``, and ``diversity_reg == 0.0`` with weights copied
      from an :class:`~mowl.nn.el.elbe.module.ELBEModule`, every GCI loss
      (positive and negative) is identical to
      :class:`~mowl.nn.el.elbe.module.ELBEModule` at equal ``margin`` to within
      floating-point tolerance ``1e-6``. Both ``MultiELBEModule`` and
      ``ELBEModule`` keep the default ``margin=0.1`` as specified in the brief
      (note that the high-level :class:`~mowl.models.ELBE` model wrapper defaults
      to ``margin=0``, so comparing ``MultiELBE`` and ``ELBE`` models requires
      passing an equal ``margin`` explicitly).
    - **Effective component count:** :meth:`effective_components` counts, for
      each class on CPU via corner arithmetic, the components whose half-widths
      exceed ``eps`` in every coordinate and whose box is not contained in
      another component of the same class (with exact ties broken by keeping
      the lowest-index component).
    - **Checkpointed temperature buffer:** ``tau`` is stored as a registered
      ``float32`` buffer so it is saved and restored in ``state_dict()`` while
      still accepting ``module.tau = float_value`` assignments.

    Non-guarantees:

    - Role axiom losses (``role_inclusion``, ``role_chain``) are not implemented
      (``role_axiom_capable = False``).
    - Negative sampling is only supported for ``gci2`` and
      ``object_property_assertion`` (``neg_capable_gcis``).
    - Under ``component_aggregation="soft"`` at finite ``tau > 0``, the soft
      loss is not zero at a hard model because :func:`~mowl.representation.library.multibox_parts.soft_minimum`
      (:math:`\\le \\text{mean}`) pulls all target components toward each source
      component in proportion to their softmax weights; ``diversity_reg``
      (:class:`~mowl.representation.library.multibox_objective.DiversityPenalty`) exists to counteract this
      collapse pressure.

    Parameters
    ----------
    nb_ont_classes : int
        Number of ontology classes.
    nb_rels : int
        Number of object properties (roles).
    nb_inds : Optional[int], default None
        Number of individuals (ABox).
    embed_dim : int, default 50
        Ambient Euclidean dimension :math:`d`.
    margin : float, default 0.1
        ELBE inclusion margin (positive margin tightens; kept at ``0.1`` as
        specified in the brief, matching :class:`~mowl.nn.el.elbe.module.ELBEModule`).
    n_components : int, default 1
        Number of box components :math:`k \\ge 1` per class.
    component_reg : float, default 0.0
        Weight for the anti-collapse pairwise component centre regulariser
        :math:`\\text{mean}_C \\text{mean}_{i<j} \\text{relu}(\\tau_c - \\|c_{C,i} - c_{C,j}\\|_2)`
        with :math:`\\tau_c = 0.1`.
    component_aggregation : str, default "hard"
        Component aggregation mode: ``"hard"`` or ``"soft"``.
    tau : float, default 1.0
        Soft-assignment temperature stored as a registered buffer on the module.
    diversity_reg : float, default 0.0
        Diversity regularisation weight stored on the module.
    objective : str, default "axiom"
        Training objective mode: ``"axiom"``, ``"point"``, ``"mixed"``, or
        ``"hybrid"``. Under ``objective="point"`` and ``"mixed"``, ABox
        ``class_assertion`` is still trained in the main loop via
        :math:`-\\log m_C(a)` while ``object_property_assertion`` is zeroed
        (handled via :meth:`point_role_loss`).
    beta : float, default 0.1
        Sigmoid temperature :math:`\\beta > 0` for soft box membership under
        ``objective="point"``, ``"mixed"``, or ``"hybrid"``.
    neg_weight : float, default 0.1
        Weight multiplying random-negative BCE terms under ``objective="point"``,
        ``"mixed"``, or ``"hybrid"``.
    volume_reg : float, default 0.0
        Volume regularisation weight stored on the module for ``objective="point"``,
        ``"mixed"``, or ``"hybrid"``.
    ind_halfwidth : float, optional
        Virtual box half-width around individual targets in role pairs under
        ``objective="point"``, ``"mixed"``, or ``"hybrid"``; defaults to
        ``5.0 * beta``.

    References
    ----------
    G. Federico, M. Leemhuis and O. Kutz, "Introducing Multiboxes: Increasing
    the Trustworthiness of Ontology Embeddings", JOWO 2026.
    """

    neg_capable_gcis = frozenset({"gci2", "object_property_assertion"})
    role_axiom_capable = False

    #: ELBE's sign convention: a positive margin requires the inclusion with a gap.
    margin_semantics = "strict"

    #: The order in which the tables were initialised (a seeded module has its numbers).
    released_order = ("class_model.centres", "class_model.offsets", "relation_model.vectors",
                      "individual_model.centres", "individual_model.offsets")
    #: ``per_axiom_column = False``: the flat ``(B,)`` of the formulas for these forms (the
    #: others were columns).
    flat_forms = ("gci0_bot", "gci3_bot")

    def __init__(
        self,
        nb_ont_classes,
        nb_rels,
        nb_inds=None,
        embed_dim=50,
        margin=0.1,
        n_components=1,
        component_reg=0.0,
        component_aggregation="hard",
        tau=1.0,
        diversity_reg=0.0,
        objective="axiom",
        beta=0.1,
        neg_weight=0.1,
        volume_reg=0.0,
        ind_halfwidth=None,
    ):
        tau, beta_f, halfwidth = check_union_options(
            n_components, embed_dim, component_aggregation, tau, diversity_reg, beta,
            ind_halfwidth, neg_weight, volume_reg)
        if objective not in _VALID_OBJECTIVES:
            raise ValueError(f"objective must be one of {_VALID_OBJECTIVES}, got "
                             f"{objective!r}")
        rng_state = self.rng_state()
        multielbe(
            nb_ont_classes, nb_rels, nb_inds or 0, embed_dim=embed_dim, margin=margin,
            n_components=n_components, component_reg=component_reg,
            component_aggregation=component_aggregation, tau=tau,
            diversity_reg=float(diversity_reg), reading=objective, beta=beta_f,
            neg_weight=float(neg_weight), ind_halfwidth=halfwidth,
            volume_reg=float(volume_reg), into=self)
        if nb_inds == 0:
            # the earlier module has (empty) tables of individuals when it is told there are
            # none; so has the module it extends (ELBE)
            self.individual_model = UnionClasses(0, embed_dim, 1)
            self.ind_embed = self.individual_model.centres
        self.initialise_in_released_order(rng_state)
        self._install_tau(tau)
        self.strict_negatives = True
        self.reading = objective
        self.__dict__["_custom_ind_halfwidth"] = ind_halfwidth is not None
        self.nb_ont_classes = nb_ont_classes
        self.nb_rels = nb_rels
        self.nb_inds = nb_inds
        self.embed_dim = embed_dim
