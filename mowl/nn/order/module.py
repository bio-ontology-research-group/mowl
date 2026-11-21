r"""Neural modules of the order-embedding methods: hyperbolic entailment cones, shadow cones
and disk embeddings. Each handles the normal form ``gci0`` (:math:`C \sqsubseteq D`) only:
the child is ``gci[:, 0]`` and the parent ``gci[:, 1]``; the score of an axiom (its
``gci0`` loss) is an energy, lower meaning more plausible."""

import contextlib

import torch as th
import torch.nn as nn

from mowl.nn.el.elmodule import ELModule
from mowl.nn.order import cones as C
from mowl.nn.order.manifolds import (Euclidean, Hyperboloid, PoincareBall, Sphere,
                                     UpperHalfSpace)
from mowl.nn.order.disks import protrusion


class OrderModule(ELModule):
    """Base: a table of points on a manifold, scored on ``gci0`` only."""

    neg_capable_gcis = frozenset({"gci0"})

    manifold = None

    def __init__(self, nb_classes, embed_dim, seed=None):
        super().__init__()
        self.nb_classes = nb_classes
        self.embed_dim = embed_dim
        gen = None
        if seed is not None:
            gen = th.Generator().manual_seed(seed)
        self.class_embed = nn.Embedding(nb_classes, self.storage_dim)
        with th.no_grad():
            self.class_embed.weight.copy_(self.init_points(nb_classes, gen))

    @property
    def storage_dim(self):
        return self.embed_dim

    def init_points(self, n, gen):
        raise NotImplementedError

    @contextlib.contextmanager
    def scoring(self):
        """Raw energies are what ``forward`` returns for positives; this context exists so
        that regimes that re-score negatives (:class:`~mowl.training.SoftmaxEnergyContrastiveLoss`)
        can be used with these modules. It changes nothing."""
        yield

    def points(self, idx):
        return self.class_embed(idx)

    def param_groups(self):
        """Parameter groups for :class:`~mowl.nn.order.manifolds.RiemannianSGD`."""
        return [{"params": [self.class_embed.weight], "manifold": self.manifold}]

    def energy(self, child, parent):
        """Energy of ``child ⊑ parent`` for index tensors; lower is more plausible."""
        raise NotImplementedError

    def negative_loss(self, energy):
        return energy

    def gci0_loss(self, gci, neg=False):
        energy = self.energy(gci[:, 0], gci[:, 1])
        return self.negative_loss(energy) if neg else energy

    def forward(self, gci, gci_name, neg=False):
        if gci_name != "gci0":
            raise NotImplementedError(
                f"{type(self).__name__} embeds the subclass order between named classes "
                f"and has a score for 'gci0' only, not for {gci_name!r}.")
        return self.gci0_loss(gci, neg=neg)


class HyperbolicConesModule(OrderModule):
    r"""Hyperbolic entailment cones (Ganea et al. 2018). Each class is a point :math:`x`
    of the Poincare ball with the cone of half-aperture :math:`\psi(x)`; the score of
    :math:`C \sqsubseteq D` is :math:`E(D, C) = \max(0, \Xi(D, C) - \psi(D))`
    (zero iff :math:`C` lies in the cone of :math:`D`). With ``neg=True`` it is the loss of
    a negative pair, :math:`\max(0, \gamma - E)`.

    :param K: The constant of Eq. 24 (:math:`K = 0.1` in the paper); it fixes the hole of
        radius :math:`\epsilon` around the origin that is kept free of points.
    :param margin: :math:`\gamma` of the max-margin loss (0.01 in the paper).
    :param init_max_norm: Initial points have Euclidean norm uniform in
        :math:`[\epsilon, \text{init\_max\_norm}]`. The paper starts from 100 epochs of
        Poincare embeddings instead (an adaptation, see the model).
    """

    def __init__(self, nb_classes, embed_dim, K=0.1, margin=0.01, init_max_norm=0.5, seed=None):
        self.K = K
        self.margin = margin
        self.init_max_norm = init_max_norm
        self.epsilon = C.entailment_cone_min_norm(K)
        self.manifold = PoincareBall(min_norm=self.epsilon, max_norm=1 - 1e-5)
        super().__init__(nb_classes, embed_dim, seed=seed)

    def init_points(self, n, gen):
        d = th.randn(n, self.embed_dim, generator=gen)
        d = d / d.norm(dim=-1, keepdim=True)
        hi = max(self.init_max_norm, self.epsilon * 1.5)
        r = self.epsilon + (hi - self.epsilon) * th.rand(n, 1, generator=gen)
        return d * r

    def energy(self, child, parent):
        return C.entailment_cone_energy(self.points(parent), self.points(child), self.K)

    def negative_loss(self, energy):
        return th.relu(self.margin - energy)


class ShadowConesModule(OrderModule):
    r"""Shadow cones (Yu et al. 2024). Each class is a point of the Poincare half-space
    (``formulation`` ending in ``half-space``) or ball; the score of :math:`C \sqsubseteq
    D` is the signed distance :math:`d(C, \mathrm{Cone}(D))` (negative when :math:`C` is
    inside the shadow cone of :math:`D`). Negatives are scored with the same energy; the
    loss of Eq. 1 is in the model.

    :param formulation: ``"umbral-half-space"``, ``"umbral-ball"``, ``"penumbral-half-space"``
        or ``"penumbral-ball"``.
    :param radius: Ball radius :math:`r` (umbral: attached to each point; penumbral-ball:
        the light source at the origin). Not trained.
    :param horosphere_height: :math:`e^h` of the penumbral-half-space light source.
    :param gamma3: Push of the apex into its cone for points of positive altitude.
    :param init_scale: Initial points are uniform in :math:`[-s, s]^n` around the origin
        :math:`(0, \ldots, 0, 1)` of the half-space (Nickel and Kiela initialisation, as in
        the paper); for the ball they are projected out of the hole of radius :math:`r`.
    """

    def __init__(self, nb_classes, embed_dim, formulation="umbral-half-space", radius=0.05,
                 horosphere_height=20.0, gamma3=1e-4, init_scale=0.01, seed=None):
        if formulation not in C.FORMULATIONS:
            raise ValueError(f"formulation must be one of {C.FORMULATIONS}.")
        if embed_dim < 2:
            raise ValueError("Shadow cones need embed_dim >= 2.")
        self.formulation = formulation
        self.radius = radius
        self.horosphere_height = horosphere_height
        self.gamma3 = gamma3
        self.init_scale = init_scale
        self.half_space = formulation.endswith("half-space")
        if self.half_space:
            self.manifold = UpperHalfSpace()
        else:
            self.manifold = PoincareBall(min_norm=C.shadow_min_norm(formulation, radius),
                                         max_norm=1 - 1e-5)
        super().__init__(nb_classes, embed_dim, seed=seed)

    def init_points(self, n, gen):
        x = (th.rand(n, self.embed_dim, generator=gen) * 2 - 1) * self.init_scale
        if self.half_space:
            x[:, -1] += 1.0
            return self.manifold.project(x)
        return self.manifold.project(x)

    def energy(self, child, parent):
        return C.shadow_cone_energy(self.points(parent), self.points(child),
                                    self.formulation, self.radius, self.horosphere_height,
                                    self.gamma3)

    def negative_loss(self, energy):
        return energy


GEOMETRIES = ("hyperbolic", "spherical", "euclidean")


class DiskEmbeddingsModule(OrderModule):
    r"""Disk embeddings (Suzuki et al. 2019). Each class is a formal disk :math:`(x, r)`:
    a centre in a hyperbolic, spherical or Euclidean space and a real radius. The score of
    :math:`C \sqsubseteq D` is the protrusion :math:`l = d(x_D, x_C) - r_D + r_C`
    (Eq. 9; :math:`\le 0` iff the disk of :math:`D` contains the disk of :math:`C`). With
    ``neg=True`` it is :math:`h_+(\mu - l)`.

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
        self.margin = margin
        self.init_scale = init_scale
        self.manifold = {"hyperbolic": Hyperboloid, "spherical": Sphere,
                         "euclidean": Euclidean}[geometry]()
        super().__init__(nb_classes, embed_dim, seed=seed)
        self.radius_embed = nn.Embedding(nb_classes, 1)
        nn.init.zeros_(self.radius_embed.weight)

    @property
    def storage_dim(self):
        return self.embed_dim + 1 if self.geometry == "hyperbolic" else self.embed_dim

    def init_points(self, n, gen):
        x = (th.rand(n, self.embed_dim, generator=gen) * 2 - 1) * self.init_scale
        if self.geometry == "hyperbolic":
            return Hyperboloid.from_spatial(x)
        if self.geometry == "spherical":
            x = x + 1e-3 * th.randn(n, self.embed_dim, generator=gen)
            return self.manifold.project(x)
        return x

    def param_groups(self):
        return [{"params": [self.class_embed.weight], "manifold": self.manifold},
                {"params": [self.radius_embed.weight], "manifold": None}]

    def energy(self, child, parent):
        return protrusion(self.manifold, self.points(parent), self.radius_embed(parent)[:, 0],
                          self.points(child), self.radius_embed(child)[:, 0])

    def negative_loss(self, energy):
        return th.relu(self.margin - energy)
