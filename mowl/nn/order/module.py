r"""Modules of the order-embedding methods: hyperbolic entailment cones, shadow cones and disk
embeddings. Each handles the normal form ``gci0`` (:math:`C \sqsubseteq D`) only: the child is
``gci[:, 0]`` and the parent ``gci[:, 1]``; the score of an axiom (its ``gci0`` loss) is an
energy, lower meaning more plausible.

The modules are *combinations of parts*: each is a :class:`~mowl.representation.ComposedGeometry`
whose class model, measure and objective are those that the function registered for the method
returns (``hyperbolic_cones``, ``shadow_cones``, ``disk_embeddings`` of
:mod:`mowl.representation.library.hierarchy_methods`). The classes here only keep the
constructor and the attributes that the modules have always had; the method is defined in the
function. The direct, hand-written loss modules they were once are the reference of
``tests/baselines`` (:mod:`mowl.baselines.nn.order`).
"""

import contextlib

from mowl.representation.library import cone_regions as C
from mowl.representation.geometry import ComposedGeometry
from mowl.representation.library import hierarchy_methods as HM

__all__ = ["OrderModule", "HyperbolicConesModule", "ShadowConesModule",
           "DiskEmbeddingsModule"]


class OrderModule(ComposedGeometry):
    """Base: a class model of points on a manifold, scored on ``gci0`` only. A subclass
    calls :meth:`_adopt` with the geometry its method function builds."""

    neg_capable_gcis = frozenset({"gci0"})

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("OrderModule is a base class; use HyperbolicConesModule, "
                                  "ShadowConesModule or DiskEmbeddingsModule.")

    def _adopt(self, geometry):
        """Become the combination of parts of ``geometry`` (the parts are shared)."""
        ComposedGeometry.__init__(
            self, geometry.class_model, geometry.relation_model, geometry.measure,
            geometry.objective, regularizers=geometry.regularizers,
            supported_forms=geometry.supported_forms,
            neg_capable_gcis=geometry.neg_capable_gcis)
        # the margin is that of the negative loss, not an inclusion margin: stated pairs have
        # none (the hierarchy objective), so the module declares no margin semantics
        self.margin_semantics = None

    @property
    def nb_classes(self):
        return len(self.class_model)

    @property
    def embed_dim(self):
        return self.class_model.embed_dim

    @property
    def manifold(self):
        return self.class_model.manifold

    @contextlib.contextmanager
    def scoring(self):
        """Raw energies are what ``forward`` returns for positives; this context exists so
        that regimes that re-score negatives (:class:`~mowl.training.SoftmaxEnergyContrastiveLoss`)
        can be used with these modules. It changes nothing."""
        yield

    def forward(self, gci, gci_name, neg=False):
        if gci_name not in self.supported_forms:
            raise NotImplementedError(
                f"{type(self).__name__} embeds the subclass order between named classes "
                f"and has a score for 'gci0' only, not for {gci_name!r}.")
        out = super().forward(gci, gci_name, neg)
        if not self.per_axiom_column and getattr(out, "ndim", 0) == 2 and out.shape[1] == 1:
            return out.reshape(-1)       # the shape of the formula (PerAxiomColumn)
        return out

    def points(self, idx):
        return self.class_embed(idx)

    def energy(self, child, parent):
        """Energy of ``child ⊑ parent`` for index tensors (one value per axiom); lower is more
        plausible: the measure's energy, signed where the method publishes it signed."""
        return self.measure.inside(self.classes(child), self.classes(parent), 0.0).reshape(-1)

    def negative_loss(self, energy):
        return self.measure.negative_of(energy, self.margin)


class HyperbolicConesModule(OrderModule):
    r"""Hyperbolic entailment cones (Ganea et al. 2018): geodesic cones at points of the
    Poincare ball, the angle against the aperture as the cost, a hinge of margin ``margin``
    for a pair that is not ordered (the method ``hyperbolic_cones``).

    :param K: The constant of Eq. 24 (:math:`K = 0.1` in the paper); it fixes the hole of
        radius :math:`\epsilon` around the origin that is kept free of points.
    :param margin: :math:`\gamma` of the max-margin loss (0.01 in the paper).
    :param init_max_norm: Initial points have Euclidean norm uniform in
        :math:`[\epsilon, \text{init\_max\_norm}]`. The paper starts from 100 epochs of
        Poincare embeddings instead (an adaptation, see the model).
    """

    def __init__(self, nb_classes, embed_dim, K=0.1, margin=0.01, init_max_norm=0.5, seed=None):
        self.K = K
        self.init_max_norm = init_max_norm
        self.epsilon = C.entailment_cone_min_norm(K)
        self._adopt(HM.hyperbolic_cones(nb_classes, 0, 0, embed_dim, K=K, margin=margin,
                                        init_max_norm=init_max_norm, seed=seed))


class ShadowConesModule(OrderModule):
    r"""Shadow cones (Yu et al. 2024): points of the Poincare half-space (``formulation``
    ending in ``half-space``) or ball, the signed distance to the shadow cone of the parent
    as the cost (negative inside the cone; the method ``shadow_cones``). Negatives are scored
    with the same energy; the loss of Eq. 1 is in the model.

    :param formulation: ``"umbral-half-space"``, ``"umbral-ball"``, ``"penumbral-half-space"``
        or ``"penumbral-ball"``.
    :param radius: Ball radius :math:`r` (umbral: attached to each point; penumbral-ball:
        the light source at the origin). Not trained.
    :param horosphere_height: :math:`e^h` of the penumbral-half-space light source.
    :param gamma3: Push of the apex into its cone for points of positive altitude.
    :param init_scale: Initial points are uniform in :math:`[-s, s]^n` around the origin
        :math:`(0, \ldots, 0, 1)` of the half-space (Nickel and Kiela initialisation, as in
        the paper); for the ball they are projected out of the hole of radius :math:`r`.
    :param margin: :math:`\gamma_1`, the distance to which a negative is pushed out of the
        cone.
    :param negative: What ``neg=True`` returns. ``"hinge"`` (default): the negative loss
        :math:`\max(0, \gamma_1 - E)`, which is small when the negative does not hold, like
        the negative loss of every other module. ``"energy"``: the raw (signed) energy, as
        earlier mOWL versions did; it is the *score* of the negative, not a loss (it is
        unbounded below and smallest deep inside the cone), and is what Eq. 1 of the paper
        is written on (:meth:`mowl.models.ShadowCones.compute_loss` applies the hinge to it).
    """

    def __init__(self, nb_classes, embed_dim, formulation="umbral-half-space", radius=0.05,
                 horosphere_height=20.0, gamma3=1e-4, init_scale=0.01, seed=None,
                 margin=0.01, negative="hinge"):
        if negative not in ("hinge", "energy"):
            raise ValueError("negative must be 'hinge' or 'energy'.")
        if formulation not in C.FORMULATIONS:
            raise ValueError(f"formulation must be one of {C.FORMULATIONS}.")
        self.formulation = formulation
        self.radius = radius
        self.horosphere_height = horosphere_height
        self.gamma3 = gamma3
        self.init_scale = init_scale
        self.negative = negative
        self.half_space = formulation.endswith("half-space")
        self._adopt(HM.shadow_cones(
            nb_classes, 0, 0, embed_dim, formulation=formulation, radius=radius,
            horosphere_height=horosphere_height, gamma3=gamma3, init_scale=init_scale,
            margin=margin, negative_form=negative, seed=seed))


GEOMETRIES = ("hyperbolic", "spherical", "euclidean")


class DiskEmbeddingsModule(OrderModule):
    r"""Disk embeddings (Suzuki et al. 2019): a formal disk :math:`(x, r)`, a centre in a
    hyperbolic, spherical or Euclidean space and a real radius; the cost of :math:`C
    \sqsubseteq D` is the protrusion :math:`l = d(x_D, x_C) - r_D + r_C` (Eq. 9;
    :math:`\le 0` iff the disk of :math:`D` contains the disk of :math:`C`), the cost of a
    negative :math:`h_+(\mu - l)` (the method ``disk_embeddings``).

    :param geometry: ``"hyperbolic"`` (Lorentz model; ``embed_dim`` is the dimension of the
        hyperbolic space, points have ``embed_dim + 1`` coordinates), ``"spherical"``
        (unit sphere of :math:`\mathbb{R}^{\text{embed\_dim}}`) or ``"euclidean"``.
    :param margin: :math:`\mu`.
    :param init_scale: Initial centres are drawn uniformly in :math:`[-s, s]^n` (and lifted
        to the hyperboloid or normalised onto the sphere); radii start at 0.
    """

    def __init__(self, nb_classes, embed_dim, geometry="hyperbolic", margin=0.01,
                 init_scale=0.1, seed=None):
        if geometry not in GEOMETRIES:
            raise ValueError(f"geometry must be one of {GEOMETRIES}.")
        self.geometry = geometry
        self.init_scale = init_scale
        self._adopt(HM.disk_embeddings(nb_classes, 0, 0, embed_dim, geometry=geometry,
                                       margin=margin, init_scale=init_scale, seed=seed))

    @property
    def radius_embed(self):
        return self.class_model.radii
