"""GroundedEL as a combination of parts: the method that is *grounded*.

GroundedEL positions the elements of a finite domain (one per class, then one per individual)
and gives every class a region of boxes and every relation a region over pairs around them.
The method is the combination

========================  ======================================================================
domain                    one element per class, then one per individual (``Domain`` of
                          ``solved_grounding``)
individual model          a point: the position of the element
class model               a union of boxes with a deadness per element (``MaskedUnionOfBoxes``)
relation model            a union of product boxes over pairs (``role_mode="product"``) or of
                          translation boxes (``"translation"``)
measure                   bounds per coordinate: the signed depth of a point in a box
objective                 the grounded constraints (``GroundedConstraints`` of
                          ``mowl.models.grounded.objective``): hinge losses of the axioms
                          grounded over sampled elements and region points
training regime           ``mowl.training.GroundedELTraining``: temperature annealing, a
                          learning-rate schedule, idle-component reseeding, dual weights
evaluation regime         ``crisp_report`` and ``evaluate_model`` (``GroundedEvaluation`` of
                          ``mowl.models.grounded.evaluation``)
========================  ======================================================================

The class and relation models of the module share the positions of the elements (a class is
a region *around* the elements that belong to it), which is why they live in one module,
:class:`mowl.nn.GroundedELModule`; the module is a
:class:`~mowl.representation.Geometry` (``classes``, ``individuals``, ``exists``,
``conjunction``, ``inside``, ``apart``, ``member``) and names its parts (``parts``).
:func:`grounded` is the named combination.
"""

from mowl.representation.methods import register_method

__all__ = ["grounded"]


@register_method("grounded")
def grounded(nb_classes, nb_relations, nb_individuals=0, embed_dim=50, n_components=2,
             n_role_components=2, role_mode="product", tau=1.0, agg="max", agg_tau=0.1,
             top_id=None, bot_id=None, inverse_of=None, tables=None):
    """GroundedEL from its parts (module docstring): the element-grounded geometry, which is a
    :class:`mowl.nn.GroundedELModule` over ``nb_classes + nb_individuals`` elements.

    The objective of the method is the grounded constraints, a part of
    :class:`mowl.models.GroundedEL` (it needs the axioms of the ontology to sample from).

    :param n_components, n_role_components: Boxes per class and per role.
    :param role_mode: ``"product"`` (a role is a set of pairs of boxes) or ``"translation"``
        (a role is a box of translations).
    :param tables: An existing module (it is its own geometry; returned as it is).
    """
    if tables is not None:
        return tables
    from mowl.nn.el.grounded.module import GroundedELModule
    return GroundedELModule(
        nb_elements=nb_classes + nb_individuals, nb_classes=nb_classes, nb_rels=nb_relations,
        embed_dim=embed_dim, n_components=n_components, n_role_components=n_role_components,
        tau=tau, agg=agg, agg_tau=agg_tau, top_id=top_id, bot_id=bot_id, role_mode=role_mode,
        inverse_of=inverse_of)
