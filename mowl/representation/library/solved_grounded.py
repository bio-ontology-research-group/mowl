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

from mowl.representation.dataset_inputs import register_dataset_inputs
from mowl.representation.methods import register_method
from mowl.representation.regions import ConstructorUnavailable

__all__ = ["grounded"]


@register_method("grounded")
def grounded(nb_classes, nb_relations, nb_individuals=0, embed_dim=50, n_components=2,
             n_role_components=2, role_mode="product", tau=1.0, agg="max", agg_tau=0.1,
             top_id=None, bot_id=None, inverse_of=None, class_model=None, relation_model=None,
             measure=None, tables=None, objective=None, silence_bottom=False, tau_end=0.01,
             dead_init_logit=0.0, element_features=None, encoder=None):
    """GroundedEL from its parts (module docstring): the element-grounded geometry, which is a
    :class:`mowl.nn.GroundedELModule` over ``nb_classes + nb_individuals`` elements, with the
    grounded constraints as its joint objective (``geometry.joint_objective``,
    :class:`~mowl.representation.library.grounded_objective.GroundedJointObjective`) and the
    regimes that train it in a general model (``geometry.default_regimes``: the steps of the
    joint objective, :class:`~mowl.training.joint.JointObjectiveSteps`, and the annealing of
    the temperature from ``tau`` to ``tau_end``).

    :class:`mowl.models.ComposedEL` builds it with ``method="grounded"`` (the ids of top and
    bottom come from the dataset, :mod:`mowl.representation.dataset_inputs`) and swaps its
    parts by the names of the general model: ``classes``, ``relations``, ``measure`` and
    ``objective`` are ``class_model``, ``relation_model``, ``measure`` and ``objective`` here
    (:attr:`grounded.part_options`).

    :param n_components, n_role_components: Boxes per class and per role.
    :param role_mode: The relation model: ``"product"`` (a role is a set of pairs of boxes) or
        ``"translation"`` (a role is a box of translations).
    :param agg, agg_tau: The measure (maximum or log-sum-exp over the boxes of a union).
    :param class_model, relation_model, measure: Replace a part (a name of
        :data:`~mowl.representation.library.grounded_parts.GROUNDED_PARTS`, a factory or a
        part; see :class:`mowl.nn.GroundedELModule`). A name that is not a part of the
        grounded geometry is refused with :class:`~mowl.representation.ConstructorUnavailable`.
    :param objective: The joint objective (default: the grounded constraints with their
        defaults); a :class:`~mowl.representation.library.grounded_objective.GroundedObjective`
        is wrapped with the default samplers.
    :param silence_bottom: Bottom has no element and no box (what GroundedEL does after
        building; a general model asks for it).
    :param dead_init_logit, element_features, encoder: The element domain (see
        :class:`mowl.nn.GroundedELModule`).
    :param tables: An existing module (it is its own geometry; returned as it is).
    """
    if tables is not None:
        return tables
    from mowl.nn.el.grounded.module import _DEAD_LOGIT, GroundedELModule
    from mowl.representation.library.grounded_objective import (GroundedJointObjective,
                                                                GroundedObjective)
    from mowl.representation.joint_objective import JointObjective
    from mowl.representation.library.grounded_parts import GROUNDED_PARTS
    for kind, value in (("classes", class_model), ("relations", relation_model),
                        ("measure", measure)):
        if isinstance(value, str) and value not in GROUNDED_PARTS[kind]:
            raise ConstructorUnavailable(
                f"The grounded geometry has {kind} over its element domain "
                f"({sorted(GROUNDED_PARTS[kind])}); {value!r} is not one: its regions are "
                f"not unions of boxes around the elements.")
    if objective is not None and not isinstance(objective, (JointObjective, GroundedObjective)):
        raise ConstructorUnavailable(
            f"The grounded geometry is trained by a joint objective (every axiom form of a "
            f"step read at one draw of elements); {type(objective).__name__} scores one form "
            f"at a time.")
    module = GroundedELModule(
        nb_elements=nb_classes + nb_individuals, nb_classes=nb_classes, nb_rels=nb_relations,
        embed_dim=embed_dim, n_components=n_components, n_role_components=n_role_components,
        tau=tau, agg=agg, agg_tau=agg_tau, dead_init_logit=dead_init_logit,
        element_features=element_features, encoder=encoder, top_id=top_id, bot_id=bot_id,
        role_mode=role_mode, inverse_of=inverse_of, class_model=class_model,
        relation_model=relation_model, measure=measure)
    if silence_bottom and bot_id is not None:
        module.elements.clamp_dead(bot_id, _DEAD_LOGIT)
        module.class_model.silence(bot_id)
    if isinstance(objective, GroundedObjective):
        objective = GroundedJointObjective(objective)
    module.joint_objective = objective if objective is not None else GroundedJointObjective()
    from functools import partial
    from mowl.training.joint import JointObjectiveSteps
    from mowl.training.schedules import TemperatureAnnealing
    module.default_regimes = (partial(JointObjectiveSteps, seed_offset=1337),
                              partial(TemperatureAnnealing, start=tau, end=tau_end))
    return module


#: The part arguments of a general model and the options of :func:`grounded` they are.
grounded.part_options = {"classes": "class_model", "relations": "relation_model",
                         "measure": "measure", "objective": "objective"}


def _grounded_inputs(model):
    """What ``grounded`` takes from a model: the ids of top and bottom among its classes, and
    bottom silenced."""
    from mowl.owlapi.defaults import BOT, TOP
    index = model.class_index_dict
    return dict(top_id=index.get(TOP), bot_id=index.get(BOT), silence_bottom=True)


register_dataset_inputs("grounded")(_grounded_inputs)
