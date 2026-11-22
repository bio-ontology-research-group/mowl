"""The methods that embed a hierarchy, rebuilt from parts (group ``hierarchy``).

Each function returns the geometry of one method, made of a class model, a measure and an
objective, with new tables or (``tables=``) with the tables of the existing module, which is
how :func:`mowl.representation.geometry_of` describes that module. The options that
reproduce a published loss and are not the semantic default are set here, nowhere else:
the signed energies of shadow cones and disks, the constant of the box lattice loss, the
smoothing details of the Gumbel box and BCBox formulas.

=====================  ==================  ================  ================================
method                 class model         measure           space
=====================  ==================  ================  ================================
``order_embeddings``   order cone          order violation   :math:`\\mathbb{R}^d_+`
``cate``               order cone          order violation  :math:`\\mathbb{R}^d` (lattice)
``box_lattice``        box (unit cube)     volume ratio      unit cube
``gumbel_box``         box (unit cube)     smoothed volume   unit cube
``bcbox``              box with codes      smoothed volume   :math:`\\mathbb{R}^d`
``hyperbolic_cones``   geodesic cone       angle             Poincare ball
``shadow_cones``       point               shadow cone       half-space or ball
``disk_embeddings``    disk                protrusion        hyperboloid, sphere, Euclidean
``hit``                point (from text)   distance, depth   Poincare ball (curvature 1/d)
``ont``                point (from text)   distance, depth   Poincare ball + rotation per role
=====================  ==================  ================  ================================
"""

import torch as th

from mowl.nn.order import cones as C
from mowl.nn.order.manifolds import (Euclidean, Hyperboloid, PoincareBall, Sphere,
                                     UpperHalfSpace)
from mowl.representation.adapters import register_adapter
from mowl.representation.library.hierarchy_classes import (
    CodedBoxes, ConePoints, DiskClasses, OrderPoints, PointClasses, TextPoints,
    UnitCubeBoxes)
from mowl.representation.library.hierarchy_geometry import (
    LatticeGeometry, TextGeometry, hierarchy_geometry)
from mowl.representation.objectives import Objective
from mowl.representation.library.hierarchy_measures import (
    ConeAngle, DistanceDepth, OrderViolation, Protrusion, ShadowCone, SoftVolumeRatio,
    VolumeRatio)
from mowl.representation.library.hierarchy_relations import TextRotation
from mowl.representation.methods import register_method
from mowl.training.regularization import Regularizer

__all__ = ["order_embeddings", "cate", "box_lattice", "gumbel_box", "bcbox",
           "hyperbolic_cones", "shadow_cones", "disk_embeddings", "hit", "ont",
           "MaxCornerPenalty", "CodePenalty"]


class MaxCornerPenalty(Regularizer):
    r"""The "dimension-max regularisation" of the box lattice model: :math:`w \cdot
    \mathrm{mean}(1 - \max)`, which pulls the upper corners to 1 and stops boxes from
    collapsing (Vilnis et al. 2018, Appendix D.1)."""

    kind = "penalty"

    def __init__(self, weight=0.005):
        super().__init__(parameters="class_embed", weight=weight)

    def penalty(self, model):
        classes = model.class_model
        high = classes(th.arange(len(classes), device=classes.device)).high
        return self.weight * (1.0 - high).mean()


class CodePenalty(Regularizer):
    r"""The sparsity penalty of the binary codes of BCBox, :math:`w \lVert 1 - B
    \rVert_1` (Zhang et al. 2022, Section 4.5)."""

    kind = "penalty"

    def __init__(self, weight=1.0):
        super().__init__(parameters="class_embed", weight=weight)

    def penalty(self, model):
        return self.weight * model.class_model.code_regularization()


# -- order embeddings and CatE ----------------------------------------------------------
@register_method("order_embeddings")
def order_embeddings(nb_classes, nb_relations=0, nb_individuals=0, embed_dim=50, margin=1.0,
                     init_scale=1.0, tables=None):
    """Order embeddings (Vendrov et al. 2016): non-negative vectors in the reversed product
    order, the order violation as the cost of a pair, a hinge of margin ``margin`` for a pair
    that is not ordered.

    :param tables: A module with ``class_embed`` to share parameters with.
    """
    classes = OrderPoints(nb_classes, embed_dim, True, "uniform", init_scale) if tables is None \
        else OrderPoints(table=tables.class_embed, nonnegative=True)
    return hierarchy_geometry(classes, None, OrderViolation("reversed", 2),
                              Objective.hierarchy(margin))


@register_method("cate")
def cate(nb_classes, nb_relations=0, nb_individuals=0, embed_dim=50, margin=1.0, p=2,
         tables=None, conjunctions=None):
    """CatE (Zhapa-Camacho and Hoehndorf 2023): free vectors in the product order, one per
    element of the lattice that the categorical projection builds. The compound classes
    :math:`\\exists r.D` and :math:`C \\sqcap D` are elements with embeddings of their own, so
    this geometry offers them by lookup and raises an error for anything that is not an
    element.

    :param tables: A :class:`~mowl.nn.graph.cate.CatEModule` to share parameters (and its
        registered existentials) with.
    :param conjunctions: ``{(i, j): k}``, the element of :math:`C \\sqcap D` for the elements
        ``i``, ``j``.
    :param margin: The margin of the negative cost (the module has none; CatE's default is 1).
    """
    if tables is None:
        classes = OrderPoints(nb_classes, embed_dim, False, "xavier")
        existentials = None
    else:
        classes = OrderPoints(table=tables.node_embedding, nonnegative=False)
        existentials = tables.existential_ids
        p = tables.p
    return LatticeGeometry(classes, OrderViolation("product", p), Objective.hierarchy(margin),
                           existentials=existentials, conjunctions=conjunctions)


# -- boxes ------------------------------------------------------------------------------
@register_method("box_lattice")
def box_lattice(nb_classes, nb_relations=0, nb_individuals=0, embed_dim=50, min_delta=0.01,
                init_side=(0.3, 0.9), max_regularization=0.0, eps=1e-6,
                freeze_delta_negatives=True, disjoint_weight=None, zero_at_inclusion=True,
                tables=None):
    """Box lattice embeddings (Vilnis et al. 2018): boxes in the unit cube, the negative log
    of the conditional probability :math:`P(D \\mid C)` as the cost of a pair, the cross
    entropy with target 0 for a corrupted pair, and the cube as a constraint
    (``geometry.project()`` after each step).

    As published, negative pairs do not update the box sizes (``freeze_delta_negatives``).
    The cost of an included pair is exactly 0 (``zero_at_inclusion=True``); ``False`` is the
    guard of the earlier code, which gives :math:`-\\log(1 + \\varepsilon)`, a constant
    that does not change a gradient.

    :param tables: A :class:`~mowl.nn.el.box_lattice.BoxLatticeModule` to share parameters
        with.
    """
    if tables is None:
        classes = UnitCubeBoxes(nb_classes, embed_dim, min_delta, init_side)
    else:
        classes = UnitCubeBoxes(table=tables.class_embed, min_delta=tables.min_delta)
        eps, disjoint_weight = tables.eps, tables.disjoint_weight
        freeze_delta_negatives = tables.freeze_delta_negatives
        max_regularization = tables.max_regularization
        zero_at_inclusion = tables.zero_at_inclusion
    regularizers = (MaxCornerPenalty(max_regularization),) if max_regularization else ()
    return hierarchy_geometry(
        classes, None, VolumeRatio(eps, disjoint_weight, zero_at_inclusion=zero_at_inclusion),
        Objective.hierarchy(freeze_size_negatives=freeze_delta_negatives),
        regularizers=regularizers)


@register_method("gumbel_box")
def gumbel_box(nb_classes, nb_relations=0, nb_individuals=0, embed_dim=50, beta=0.01,
               min_delta=1e-6, init_side=(0.3, 0.9), max_regularization=0.0,
               log_guard="stable", tables=None):
    """Gumbel box embeddings (Dasgupta et al. 2020): the box lattice's class model with the
    smoothed volume ratio as the measure (Gumbel scale ``beta``, the softplus side length of
    Section 4.2).

    :param log_guard: ``"stable"`` (default): the guards of the binary code boxes (``log
        softplus(z)`` is ``z`` below -15, energy floor 1e-12); ``"epsilon"``: those of the
        earlier code (``log(softplus + 1e-30)``, energy floor 1e-9), which give a constant
        value and no gradient for collapsed sides.
    :param tables: A :class:`~mowl.nn.el.gumbel_box.GumbelBoxModule` to share parameters
        with.
    """
    if log_guard not in ("stable", "epsilon"):
        raise ValueError("log_guard must be 'stable' or 'epsilon'.")
    if tables is None:
        classes = UnitCubeBoxes(nb_classes, embed_dim, min_delta, init_side, soft=True)
    else:
        classes = UnitCubeBoxes(table=tables.class_embed, min_delta=tables.min_delta,
                                soft=True)
        beta, max_regularization = tables.beta, tables.max_regularization
        log_guard = tables.log_guard
    regularizers = (MaxCornerPenalty(max_regularization),) if max_regularization else ()
    stable = log_guard == "stable"
    measure = SoftVolumeRatio(beta, beta, gumbel_shift=True,
                              tail="linear" if stable else "floor",
                              energy_floor=1e-12 if stable else 1e-9)
    return hierarchy_geometry(classes, None, measure, Objective.hierarchy(),
                              regularizers=regularizers)


@register_method("bcbox")
def bcbox(nb_classes, nb_relations=0, nb_individuals=0, embed_dim=50, variant="vbc",
          num_codes=2, code_dims=None, intersection_temp=0.01, volume_temp=1.0,
          code_init=0.0, code_weight=0.0, tables=None):
    """Binary code box embeddings (Zhang et al. 2022): Gumbel boxes whose dimensions are
    switched off by binary codes (per class, ``variant="vbc"``, or global, ``"gbc"``); the
    codes are stored by the class model and read by the measure. The cost of a pair is the
    energy :math:`-\\log P(D \\mid C)` and :math:`-\\log(1 - e^{-E})` for a corrupted pair.

    Directed edges of the paper point from the parent to the child; here, as everywhere in
    this library, ``inside(C, D)`` is the cost of :math:`C \\sqsubseteq D`, which is the
    module's ``energy(D, C)``.

    :param tables: A :class:`~mowl.nn.graph.bcbox.BinaryCodeBoxModule` to share parameters
        with.
    :param code_weight: Weight of the code penalty (the model's ``w_r``).
    """
    if tables is None:
        classes = CodedBoxes(nb_classes, embed_dim, variant, num_codes, code_dims, code_init)
    else:
        classes = CodedBoxes(variant=tables.variant, tables=tables)
        intersection_temp, volume_temp = tables.intersection_temp, tables.volume_temp
    measure = SoftVolumeRatio(intersection_temp, volume_temp, gumbel_shift=False,
                              tail="linear", energy_floor=1e-12, dtype=None)
    regularizers = (CodePenalty(code_weight),) if code_weight else ()
    return hierarchy_geometry(classes, None, measure, Objective.hierarchy(),
                              regularizers=regularizers)


# -- points on manifolds ----------------------------------------------------------------
@register_method("hyperbolic_cones")
def hyperbolic_cones(nb_classes, nb_relations=0, nb_individuals=0, embed_dim=50, K=0.1,
                     margin=0.01, init_max_norm=0.5, tables=None):
    """Hyperbolic entailment cones (Ganea et al. 2018): a point of the Poincare ball with the
    cone of half-aperture :math:`\\psi(x)` at it, the angle against the aperture as the cost,
    a hinge of margin ``margin`` for a pair that is not ordered.

    :param tables: A :class:`~mowl.nn.order.HyperbolicConesModule` to share parameters with.
    """
    if tables is None:
        classes = ConePoints(nb_classes, embed_dim, K, init_max_norm)
    else:
        classes = ConePoints(K=tables.K, table=tables.class_embed)
        margin = tables.margin
    return hierarchy_geometry(classes, None, ConeAngle(), Objective.hierarchy(margin))


@register_method("shadow_cones")
def shadow_cones(nb_classes, nb_relations=0, nb_individuals=0, embed_dim=50,
                 formulation="umbral-half-space", radius=0.05, horosphere_height=20.0,
                 gamma3=1e-4, init_scale=0.01, margin=0.01, negative_form="hinge",
                 tables=None):
    """Shadow cones (Yu et al. 2024): points of the Poincare half-space or ball, the signed
    distance to the shadow cone of the parent as the cost. As published the cost of a stated
    pair is signed (negative inside the cone, ``signed=True``). The cost of a corrupted pair is
    the hinge :math:`\\max(0, \\gamma_1 - E)` of the energy (``negative_form="hinge"``,
    :math:`\\gamma_1` = ``margin``): a loss that is zero where the corrupted pair does not
    hold; ``"energy"`` is the raw energy, the quantity Eq. 1 of the paper is written on (the
    loss of :class:`~mowl.models.ShadowCones` applies the hinge to it).

    :param tables: A :class:`~mowl.nn.order.ShadowConesModule` to share parameters with.
    """
    if tables is not None:
        formulation, radius = tables.formulation, tables.radius
        horosphere_height, gamma3 = tables.horosphere_height, tables.gamma3
        manifold = tables.manifold
        margin, negative_form = tables.margin, tables.negative
    elif formulation.endswith("half-space"):
        manifold = UpperHalfSpace()
    else:
        manifold = PoincareBall(min_norm=C.shadow_min_norm(formulation, radius),
                                max_norm=1 - 1e-5)
    if tables is None:
        classes = PointClasses(nb_classes, embed_dim, manifold, init_scale)
    else:
        classes = PointClasses(manifold=manifold, table=tables.class_embed)
    measure = ShadowCone(formulation, radius, horosphere_height, gamma3, signed=True,
                         negative_form=negative_form)
    return hierarchy_geometry(classes, None, measure, Objective.hierarchy(margin))


_DISK_SPACES = {"hyperbolic": Hyperboloid, "spherical": Sphere, "euclidean": Euclidean}


@register_method("disk_embeddings")
def disk_embeddings(nb_classes, nb_relations=0, nb_individuals=0, embed_dim=50,
                    geometry="hyperbolic", margin=0.01, init_scale=0.1, tables=None):
    """Disk embeddings (Suzuki et al. 2019): a point of a hyperbolic, spherical or Euclidean
    space with a real radius, the protrusion of the child out of the parent as the cost (signed,
    as published), a hinge of margin ``margin`` for a pair that is not ordered.

    :param tables: A :class:`~mowl.nn.order.DiskEmbeddingsModule` to share parameters with.
    """
    if tables is None:
        classes = DiskClasses(nb_classes, embed_dim, _DISK_SPACES[geometry](), init_scale)
    else:
        classes = DiskClasses(manifold=tables.manifold, table=tables.class_embed,
                              radii=tables.radius_embed)
        margin = tables.margin
    return hierarchy_geometry(classes, None, Protrusion(signed=True),
                              Objective.hierarchy(margin))


# -- language models --------------------------------------------------------------------
@register_method("hit")
def hit(nb_classes=None, nb_relations=0, nb_individuals=0, embed_dim=None, margin=0.0, *,
        encoder=None, texts=None, curvature=None, centri_weight=1.0, clustering_margin=5.0,
        centripetal_margin=0.1, clustering_weight=1.0, centripetal_weight=1.0, tables=None):
    """HiT (He et al. 2024): the point of a class is what a text encoder computes from its
    verbalisation, read in the Poincare ball of curvature :math:`1/d`; the cost of a pair is
    :math:`d(C, D) + \\lambda(\\lVert D \\rVert - \\lVert C \\rVert)`. The training loss is the
    triplet loss :meth:`TextGeometry.hierarchy_loss`. The sizes are not used: classes are
    texts.

    :param encoder: The encoder (:mod:`mowl.nn.lm.encoders`).
    :param texts: The verbalisation of every class id (optional; without it classes are given
        as lists of strings).
    :param centri_weight: :math:`\\lambda` (HiT selects it on validation data).
    :param tables: A :class:`~mowl.nn.lm.HiTModule` whose encoder, curvature, margins and
        weights are used.
    """
    if tables is not None:
        encoder, curvature = tables.encoder, tables.c
        clustering_margin, centripetal_margin = tables.clustering_margin, \
            tables.centripetal_margin
        clustering_weight, centripetal_weight = tables.clustering_weight, \
            tables.centripetal_weight
    classes = TextPoints(encoder, texts, curvature=curvature)
    return TextGeometry(classes, None, DistanceDepth(centri_weight), Objective.hierarchy(),
                        clustering_margin, centripetal_margin, clustering_weight,
                        centripetal_weight)


@register_method("ont")
def ont(nb_classes=None, nb_relations=0, nb_individuals=0, embed_dim=None, margin=0.0, *,
        encoder=None, texts=None, role_texts=None, role_model=None, curvature=None,
        centri_weight=1.0, clustering_margin=3.0, centripetal_margin=0.5,
        clustering_weight=1.0, centripetal_weight=1.0, tables=None):
    """OnT (Yang et al. 2025): HiT with a relation model: the region of :math:`\\exists r.D` is
    :math:`f_r(x_D)`, a rotation of the point of :math:`D` whose angles are a linear function
    of the encoding of the name of :math:`r` (:class:`~.hierarchy_relations.TextRotation`).
    Compound classes are also embedded from their verbalisation
    (:meth:`TextGeometry.compound`) and trained to coincide with it
    (:meth:`TextGeometry.existential_loss`).

    :param role_model: A :class:`~mowl.nn.lm.ont.RoleTransformation`; ``None`` builds the
        ablation OnT(w/o r), a geometry without relations.
    :param tables: A :class:`~mowl.nn.lm.OnTModule` to share parameters with.
    """
    if tables is not None:
        encoder, curvature = tables.encoder, tables.c
        clustering_margin, centripetal_margin = tables.clustering_margin, \
            tables.centripetal_margin
        clustering_weight, centripetal_weight = tables.clustering_weight, \
            tables.centripetal_weight
        role_model = tables.role_model if tables.use_roles else None
    classes = TextPoints(encoder, texts, curvature=curvature)
    relations = None if role_model is None else TextRotation(
        role_model, encoder, classes.manifold.c, role_texts)
    return TextGeometry(classes, relations, DistanceDepth(centri_weight), Objective.hierarchy(),
                        clustering_margin, centripetal_margin, clustering_weight,
                        centripetal_weight)


# -- adapters: geometry_of(module) -------------------------------------------------------
@register_adapter("OrderEmbeddingsModule")
def _order_embeddings(module):
    return order_embeddings(None, margin=module.margin, tables=module)


@register_adapter("CatEModule")
def _cate(module):
    return cate(None, tables=module)


@register_adapter("BoxLatticeModule")
def _box_lattice(module):
    return box_lattice(None, tables=module)


@register_adapter("GumbelBoxModule")
def _gumbel_box(module):
    return gumbel_box(None, tables=module)


@register_adapter("BinaryCodeBoxModule")
def _bcbox(module):
    return bcbox(None, variant=module.variant, tables=module)


@register_adapter("HyperbolicConesModule")
def _hyperbolic_cones(module):
    return hyperbolic_cones(None, tables=module)


@register_adapter("ShadowConesModule")
def _shadow_cones(module):
    return shadow_cones(None, tables=module)


@register_adapter("DiskEmbeddingsModule")
def _disk_embeddings(module):
    return disk_embeddings(None, tables=module)


@register_adapter("HiTModule")
def _hit(module):
    return hit(tables=module)


@register_adapter("OnTModule")
def _ont(module):
    return ont(tables=module)
