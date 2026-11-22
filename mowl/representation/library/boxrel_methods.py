"""TransBox, BoxEL, BoxSEL and ELK-Em as combinations of parts, and their adapters.

Every function builds the :class:`~mowl.representation.ComposedGeometry`
of a method, with new tables or (``tables=module``) with those of an existing module, which
is how :func:`mowl.representation.geometry_of` describes the module.
"""

import torch as th
import torch.nn as nn

from mowl.representation.adapters import register_adapter
from mowl.representation.class_models import BoxClasses
from mowl.representation.library.boxrel_boxes import (
    FixableBoxes, MinDeltaBoxes, Points, signed_intersection)
from mowl.representation.library.boxrel_measures import (
    AveragedBounds, MaskedBounds, VolumeRatio)
from mowl.representation.geometry import ComposedGeometry
from mowl.representation.library.boxrel_objectives import (
    BoxSELObjective, ELKEmObjective, TransBoxObjective, boxel_objective)
from mowl.representation.library.boxrel_regularizers import (
    LocationPenalty, SizeCapPenalty, VolumePenalty)
from mowl.representation.library.boxrel_relations import (
    Affine, AffineBoxes, TranslationBoxes)
from mowl.representation.methods import register_method
from mowl.training.regularization import MaxNormPenalty, NormPenalty

__all__ = ["transbox", "boxel", "boxsel", "elkem", "ConditionalGeometry"]


def _joint_tables(size, dim):
    """Centre and offset tables initialised together: one row of :math:`U(-1,1)^{2n}`
    scaled to unit norm, as in the code of TransBox."""
    weight = th.empty(size, 2 * dim)
    nn.init.uniform_(weight, a=-1, b=1)
    weight = weight / th.clamp_min(th.linalg.norm(weight, dim=1, keepdim=True), 1e-8)
    centre, offset = nn.Embedding(size, dim), nn.Embedding(size, dim)
    with th.no_grad():
        centre.weight.copy_(weight[:, :dim])
        offset.weight.copy_(weight[:, dim:])
    return centre, offset


# ==========================================================================================
@register_method("transbox")
def transbox(nb_classes, nb_relations, nb_individuals=0, embed_dim=50, margin=0.1,
             enhanced=True, reg_factor=0.1, formulation="mowl", tables=None):
    """TransBox (Yang et al. 2025): boxes, a box of translations per relation, bounds per
    coordinate and a point per individual.

    :param enhanced: Train :math:`C \\sqsubseteq \\exists r.D` against the semantic
        enhancement.
    :param formulation: ``"mowl"`` or ``"reference"`` (the authors' code), see
        :class:`~mowl.representation.library.boxrel_objectives.TransBoxObjective`.
    :param tables: A :class:`~mowl.nn.TransBoxModule` to share parameters with.
    """
    t = tables
    if t is not None:
        classes = BoxClasses(centres=t.class_center_embedding,
                             offsets=t.class_offset_embedding)
        relations = TranslationBoxes(vectors=t.relation_center_embedding,
                                     offsets=t.relation_offset_embedding)
        individuals = None if t.ind_embedding is None else Points(centres=t.ind_embedding)
    elif formulation == "reference":
        classes = BoxClasses(*(None, None), *_joint_tables(nb_classes, embed_dim))
        relations = TranslationBoxes(None, None, *_joint_tables(nb_relations, embed_dim))
        individuals = Points(nb_individuals, embed_dim) if nb_individuals else None
    else:
        classes = BoxClasses(nb_classes, embed_dim)
        relations = TranslationBoxes(nb_relations, embed_dim)
        individuals = Points(nb_individuals, embed_dim) if nb_individuals else None
    reference = formulation == "reference"
    measure = MaskedBounds(separation="every", empty_weight=1.0 if reference else 0.5,
                           gci1="overlap" if reference else "inclusion")
    regularizers = (MaxNormPenalty("relation_model.offsets", max_norm=1.0, weight=reg_factor),)
    if not reference:
        regularizers = (NormPenalty("class_embed", target=1.0, weight=reg_factor),
                        ) + regularizers
    return ComposedGeometry(classes, relations, measure,
                            TransBoxObjective(margin, enhanced, formulation),
                            individuals=individuals, regularizers=regularizers)


@register_adapter("TransBoxModule")
def _transbox(module):
    return transbox(None, None, embed_dim=module.embed_dim, margin=module.margin,
                    enhanced=module.use_enhancement, reg_factor=module.reg_factor,
                    formulation=module.formulation, tables=module)


# ==========================================================================================
@register_method("boxel")
def boxel(nb_classes, nb_relations, nb_individuals=0, embed_dim=50, margin=0.0,
          temperature=1.0, gci1_bot="volume product", signed=True, tables=None):
    """BoxEL (Xiong et al. 2022): boxes stored as lower corner and log-width, an affine map
    per relation, a smoothed volume ratio, and a penalty that keeps the boxes in the unit
    cube.

    :param margin: Slack of the inclusion (``0``: none, as published; a negative value is a
        slack, a positive one is ignored by a volume ratio).
    :param temperature: :math:`\\beta` of the softplus volume.
    :param gci1_bot: ``"volume product"`` (the BoxEL code) or ``"apart"``, see
        :func:`~mowl.representation.library.boxrel_objectives.boxel_objective`.
    :param signed: Map the corners of a box one by one, which gives inverted boxes for a
        negative scale, as the BoxEL code does (``False``: the box is mirrored).
    :param tables: A :class:`~mowl.nn.BoxELModule` to share parameters with.
    """
    return _boxel_geometry(ComposedGeometry, boxel_objective(margin, _product(gci1_bot)),
                           nb_classes, nb_relations, nb_individuals, embed_dim, temperature,
                           signed, tables, (LocationPenalty(1.0),))


def _product(gci1_bot):
    return "product" if gci1_bot == "volume product" else gci1_bot


def _boxel_geometry(geometry, objective, nb_classes, nb_relations, nb_individuals, embed_dim,
                    temperature, signed, tables, regularizers):
    t = tables
    if t is not None:
        classes = MinDeltaBoxes(mins=t.min_embedding, deltas=t.delta_embedding)
        relations = Affine(vectors=t.relation_embedding, scales=t.scaling_embedding,
                           signed=signed)
        individuals = None if t.ind_embedding is None else MinDeltaBoxes(
            mins=t.ind_embedding, deltas=t.ind_delta_embedding)
    else:
        classes = MinDeltaBoxes(nb_classes, embed_dim)
        relations = Affine(nb_relations, embed_dim, signed=signed)
        individuals = MinDeltaBoxes(nb_individuals, embed_dim) if nb_individuals else None
    return geometry(classes, relations, VolumeRatio(temperature), objective,
                    individuals=individuals, regularizers=regularizers)


@register_adapter("BoxELModule")
def _boxel(module):
    return _boxel_geometry(ComposedGeometry, boxel_objective(0.0), None, None, 0, None,
                           module.temperature, True, module, (LocationPenalty(1.0),))


# ==========================================================================================
class ConditionalGeometry(ComposedGeometry):
    """A geometry that also reads *conditionals* :math:`(D \\mid C)[l, u]` (BoxSEL). The
    class expressions are nested tuples without names: ``("atom",)`` is a class,
    ``("and", e1, e2, ...)`` a conjunction and ``("some", e)`` an existential restriction,
    with the indices of the classes and relations of the expression, left to right, as the
    ``(B, k)`` tensor ``leaves``
    (:func:`mowl.nn.el.boxsel.losses.compile_expression` makes both from names)."""

    def expression(self, struct, leaves, pos=0):
        """``(region, next position)`` of the expression ``struct`` for a batch."""
        kind = struct[0]
        if kind == "atom":
            return self.classes(leaves[:, pos]), pos + 1
        if kind == "some":
            inner, nxt = self.expression(struct[1], leaves, pos + 1)
            return self.exists(leaves[:, pos], inner), nxt
        region = None
        for sub in struct[1:]:
            part, pos = self.expression(sub, leaves, pos)
            region = part if region is None else signed_intersection(region, part)
        return region, pos

    def conditional_loss(self, conclusion, premise, leaves_d, leaves_c, lower, upper):
        """Loss of a batch of conditionals ``(D | C)[l, u]``, ``(B, 1)``."""
        return self.cost("conditional", (conclusion, premise, leaves_d, leaves_c, lower,
                                         upper))

    @th.no_grad()
    def probability(self, conclusion, premise, leaves_d, leaves_c):
        """Proportion of the premise region that lies in the conclusion region, exact."""
        c, _ = self.expression(premise, leaves_c)
        d, _ = self.expression(conclusion, leaves_d)
        return self.measure.probability(c, d)


@register_method("boxsel")
def boxsel(nb_classes, nb_relations, nb_individuals=0, embed_dim=50, margin=0.0,
           temperature=1.0, gci1_bot="volume product", signed=True, beta=1.0,
           location_weight=1.0, volume_weight=0.0, tables=None):
    """BoxSEL (Zhu et al. 2024): BoxEL and the statement type of a conditional probability,
    with the two penalties :math:`L_{loc}` and :math:`L_{vol}`.

    :param beta, location_weight, volume_weight: Side of the cube and the weights of the
        two penalties (a weight of 0 leaves the penalty out).
    :param tables: A :class:`~mowl.nn.BoxSELModule` to share parameters with.
    """
    regularizers = []
    if location_weight:
        regularizers.append(LocationPenalty(beta, weight=location_weight))
    if volume_weight:
        regularizers.append(VolumePenalty(beta, weight=volume_weight))
    return _boxel_geometry(ConditionalGeometry,
                           BoxSELObjective(margin, gci1_bot=_product(gci1_bot)),
                           nb_classes, nb_relations, nb_individuals, embed_dim, temperature,
                           signed, tables, tuple(regularizers))


@register_adapter("BoxSELModule")
def _boxsel(module):
    return boxsel(None, None, temperature=module.temperature, beta=module.beta,
                  location_weight=module.location_weight,
                  volume_weight=module.volume_weight, tables=module)


# ==========================================================================================
@register_method("elkem")
def elkem(nb_classes, nb_relations, nb_individuals=0, embed_dim=50, margin=0.1,
          bot_ids=(), top_ids=(), abox_threshold=0.4, epsilon=1e-5, variant="paper",
          size_weight=0.0, tables=None):
    """ELK-Em (Singh et al. 2026): boxes with classes fixed to the empty or the full box, a
    relation with a range box, an error box and an affine map, bounds averaged over the
    coordinates, points for individuals.

    :param variant: ``"paper"`` or ``"reference"``, see :class:`~mowl.nn.ELKEmModule`.
    :param size_weight: Weight of the box-size penalty of the authors' code
        (:class:`~mowl.representation.library.boxrel_regularizers.SizeCapPenalty`);
        0: none.
    :param tables: An :class:`~mowl.nn.ELKEmModule` to share parameters with.
    """
    if variant not in ("paper", "reference"):
        raise ValueError("variant must be 'paper' or 'reference'.")
    t = tables
    sound = variant == "paper"
    if t is not None:
        classes = FixableBoxes(centres=t.class_center, offsets=t.class_offset,
                               bot_mask=t.bot_mask, top_mask=t.top_mask, epsilon=t.epsilon)
        relations = AffineBoxes(vectors=t.shift, range_centres=t.range_center,
                                range_offsets=t.range_offset, error_offsets=t.error_offset,
                                slopes=t.slope, eps=t.epsilon, sound=sound)
        individuals = None if t.ind_embedding is None else Points(centres=t.ind_embedding)
    else:
        classes = FixableBoxes(nb_classes, embed_dim, epsilon=epsilon)
        classes.fix(bot_ids, top_ids)
        relations = AffineBoxes(nb_relations, embed_dim, eps=epsilon, sound=sound)
        individuals = Points(nb_individuals, embed_dim) if nb_individuals else None
    objective = ELKEmObjective(margin, abox_threshold, gci2_overlap=not sound)
    regularizers = (SizeCapPenalty(weight=size_weight),) if size_weight else ()
    return ComposedGeometry(classes, relations, AveragedBounds(), objective,
                            individuals=individuals, regularizers=regularizers)


@register_adapter("ELKEmModule")
def _elkem(module):
    geometry = elkem(None, None, margin=module.margin, abox_threshold=module.abox_threshold,
                     variant=module.variant, tables=module)
    # The module uses margin 0 in evaluation mode; the geometry starts in the module's mode
    # (the two modes are switched separately afterwards).
    return geometry.train(module.training)
