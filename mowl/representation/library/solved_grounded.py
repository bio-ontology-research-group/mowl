"""GroundedEL as a combination of parts: the method that is *grounded*.

GroundedEL positions the elements of a finite domain (one per class, then one per individual)
and gives every class a region of boxes and every relation a region over pairs around them.
The method is the combination (the parts are in
:mod:`~mowl.representation.library.grounded_parts` and
:mod:`~mowl.representation.library.grounded_logic`)

========================  ======================================================================
shared part               ``ElementDomain``: the positions and the deadness of the elements (one
                          per class, then one per individual) and the membership temperature
individual model          ``ElementPoints``: the point of the individual's element
class model               ``BoxUnionClasses``: a union of boxes per class, some inactive
relation model            ``ProductBoxRelations`` (``role_mode="product"``, a union of products of
                          boxes) or ``TranslationBoxRelations`` (``"translation"``)
measure                   ``UnionDepth``: the signed depth in a union of boxes (``agg``)
objective                 ``GroundedObjective`` (``grounded_objective``): per axiom form an
                          element term (a rule of ``mowl.nn.el.grounded.constraints`` in the
                          Łukasiewicz logic, ``logic=`` swaps it) and a region term (a depth
                          hinge), the minimality regulariser; as a linear program over a
                          membership table: ``GroundedProgram`` (``grounded_program``)
samplers                  ``ElementSampler`` and ``RegionSampler`` (``grounded_sampling``):
                          which elements and which points of regions a step reads
training regime           ``mowl.training.GroundedELTraining``: temperature annealing, a
                          learning-rate schedule, idle-component reseeding, dual weights
evaluation regime         ``CrispViolations`` (``grounded_evaluation``, read by ``crisp_report``)
                          and ``evaluate_model`` (``mowl.models.grounded.evaluation``)
========================  ======================================================================

:class:`mowl.nn.GroundedELModule` combines the parts (``shared``, ``class_model``,
``relation_model``, ``individual_model``, ``measure``, ``parts``); it is a
:class:`~mowl.representation.Geometry`. :func:`grounded` is the named combination.
"""

from mowl.representation.methods import register_method

__all__ = ["grounded"]


@register_method("grounded")
def grounded(nb_classes, nb_relations, nb_individuals=0, embed_dim=50, n_components=2,
             n_role_components=2, role_mode="product", tau=1.0, agg="max", agg_tau=0.1,
             top_id=None, bot_id=None, inverse_of=None, class_model=None, relation_model=None,
             measure=None, tables=None):
    """GroundedEL from its parts (module docstring): the element-grounded geometry, which is a
    :class:`mowl.nn.GroundedELModule` over ``nb_classes + nb_individuals`` elements.

    The objective of the method is the grounded constraints, a part of
    :class:`mowl.models.GroundedEL` (it needs the axioms of the ontology to sample from).

    :param n_components, n_role_components: Boxes per class and per role.
    :param role_mode: The relation model: ``"product"`` (a role is a set of pairs of boxes) or
        ``"translation"`` (a role is a box of translations).
    :param agg, agg_tau: The measure (maximum or log-sum-exp over the boxes of a union).
    :param class_model, relation_model, measure: Replace a part (a name of
        :data:`~mowl.representation.library.grounded_parts.GROUNDED_PARTS`, a factory or a
        part; see :class:`mowl.nn.GroundedELModule`).
    :param tables: An existing module (it is its own geometry; returned as it is).
    """
    if tables is not None:
        return tables
    from mowl.nn.el.grounded.module import GroundedELModule
    return GroundedELModule(
        nb_elements=nb_classes + nb_individuals, nb_classes=nb_classes, nb_rels=nb_relations,
        embed_dim=embed_dim, n_components=n_components, n_role_components=n_role_components,
        tau=tau, agg=agg, agg_tau=agg_tau, top_id=top_id, bot_id=bot_id, role_mode=role_mode,
        inverse_of=inverse_of, class_model=class_model, relation_model=relation_model,
        measure=measure)
