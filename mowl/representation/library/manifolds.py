"""Manifolds and the Riemannian SGD used by the order-embedding methods of this package.

``geoopt`` is not a dependency of mOWL, so the few operations the methods need are written
here with PyTorch: the Riemannian gradient (the Euclidean gradient rescaled by the inverse
metric), a retraction or exponential map, and a projection that keeps points inside the
region where the method's geometry is defined. Every manifold has curvature :math:`-1`
(hyperbolic) or :math:`+1` (sphere), as in the papers.

* :class:`PoincareBall`: update by the exponential map (Ganea et al. 2018 use the
  retraction :math:`x + v`; Yu et al. 2024 use the exponential map of HTorch).
* :class:`UpperHalfSpace`: the Poincare half-space :math:`\\{x_n > 0\\}`, metric
  :math:`I / x_n^2`. The update is a retraction, linear in the horizontal coordinates and
  multiplicative in the last one (so the height stays positive); it agrees with the
  exponential map to first order.
* :class:`Hyperboloid`: the Lorentz model (Suzuki et al. 2019, Section 3.4), exact
  exponential map.
* :class:`Sphere`, :class:`Euclidean`: the other two geometries of Disk Embeddings.
"""

import torch as th

from mowl.nn.lm import hyperbolic as H


def _exact_zero(near, value_at_floor, function, arg):
    """``function(arg)``, but exactly 0 (and with zero gradient) where ``near``.

    The distance functions need a floor on their argument, because their derivative is
    infinite at coincident points; the floor used to make the distance of a point to itself
    a small positive number, so a region inside itself cost slightly above zero. The points
    that were pushed to the floor now have distance 0. The argument is kept off the
    singularity where ``near`` so the gradient stays finite."""
    safe = th.where(near, th.full_like(arg, value_at_floor), arg)
    return th.where(near, th.zeros_like(arg), function(safe))


class Manifold:
    """Interface: ``rgrad``, ``retract``, ``project``, ``distance``, ``random_points``."""

    def rgrad(self, x, grad):
        raise NotImplementedError

    def retract(self, x, v):
        raise NotImplementedError

    def project(self, x):
        return x

    def distance(self, x, y):
        raise NotImplementedError


class Euclidean(Manifold):
    def rgrad(self, x, grad):
        return grad

    def retract(self, x, v):
        return x + v

    def distance(self, x, y):
        d2 = (x - y).pow(2).sum(-1)
        return _exact_zero(d2 <= 1e-12, 1.0, th.sqrt, d2)


class Sphere(Manifold):
    """The unit sphere in :math:`\\mathbb{R}^n`, spherical distance."""

    def rgrad(self, x, grad):
        return grad - (x * grad).sum(-1, keepdim=True) * x

    def retract(self, x, v):
        n = v.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        y = x * th.cos(n) + v / n * th.sin(n)
        return self.project(y)

    def project(self, x):
        return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    def distance(self, x, y):
        cos = (x * y).sum(-1).clamp(max=1.0)
        # below the float32 resolution of the cosine (points closer than about 1.4e-3 rad)
        # the distance is 0
        cos = th.where(cos >= 1 - 1e-6, th.ones_like(cos), cos.clamp_min(-1 + 1e-6))
        return _exact_zero(cos >= 1.0, 0.0, th.acos, cos)


class PoincareBall(Manifold):
    """Poincare ball of curvature -1.

    :param min_norm: Points are kept at Euclidean norm at least ``min_norm`` (the
        :math:`\\epsilon`-hole of entailment cones, the light-source hole of shadow cones).
    :param max_norm: Points are kept at norm at most ``max_norm``.
    """

    def __init__(self, min_norm=0.0, max_norm=1 - 1e-5):
        self.min_norm = min_norm
        self.max_norm = max_norm

    def rgrad(self, x, grad):
        return grad * ((1 - (x * x).sum(-1, keepdim=True)) ** 2 / 4)

    def retract(self, x, v):
        return self.project(H.expmap(x, v, 1.0))

    def project(self, x):
        n = x.norm(dim=-1, keepdim=True).clamp_min(1e-15)
        target = n.clamp(self.min_norm, self.max_norm)
        return x / n * target

    def distance(self, x, y):
        d = H.distance(x, y, 1.0)
        # rounding in the Moebius addition leaves ~1e-8 for coincident points
        return th.where((x == y).all(dim=-1), th.zeros_like(d), d)


class UpperHalfSpace(Manifold):
    """Poincare half-space :math:`\\{x \\in \\mathbb{R}^n : x_n > 0\\}`, curvature -1."""

    def __init__(self, min_height=1e-6):
        self.min_height = min_height

    def rgrad(self, x, grad):
        return grad * x[..., -1:] ** 2

    def retract(self, x, v):
        h = x[..., -1:]
        height = h * th.exp((v[..., -1:] / h).clamp(-10, 10))
        return self.project(th.cat([x[..., :-1] + v[..., :-1], height], dim=-1))

    def project(self, x):
        return th.cat([x[..., :-1], x[..., -1:].clamp_min(self.min_height)], dim=-1)

    def distance(self, x, y):
        diff = ((x - y) ** 2).sum(-1)
        arg = 1 + diff / (2 * x[..., -1] * y[..., -1])
        return _exact_zero(arg <= 1.0 + 1e-7, 1.0 + 1e-6, th.acosh, arg)


def lorentz_inner(x, y):
    return -x[..., 0] * y[..., 0] + (x[..., 1:] * y[..., 1:]).sum(-1)


class Hyperboloid(Manifold):
    """Lorentz model :math:`\\{x : \\langle x, x\\rangle_L = -1, x_0 > 0\\}` of
    :math:`\\mathbb{H}^{n}` stored as vectors of :math:`n + 1` coordinates."""

    def rgrad(self, x, grad):
        h = grad.clone()
        h[..., 0] = -h[..., 0]
        return h + lorentz_inner(x, h).unsqueeze(-1) * x

    def retract(self, x, v):
        vn = lorentz_inner(v, v).clamp_min(0).sqrt().clamp(1e-12, 20.0).unsqueeze(-1)
        y = th.cosh(vn) * x + th.sinh(vn) * v / vn
        return self.project(y)

    def project(self, x):
        spatial = x[..., 1:]
        x0 = (1 + (spatial * spatial).sum(-1, keepdim=True)).sqrt()
        return th.cat([x0, spatial], dim=-1)

    def distance(self, x, y):
        # -<x, y> = 1 + <x - y, x - y> / 2 on the hyperboloid; the difference form has no
        # cancellation for nearby points and is exactly 0 for coincident ones
        arg = 1.0 + 0.5 * lorentz_inner(x - y, x - y).clamp_min(0.0)
        return _exact_zero(arg <= 1.0 + 1e-7, 1.0 + 1e-6, th.acosh, arg)

    @staticmethod
    def from_spatial(spatial):
        x0 = (1 + (spatial * spatial).sum(-1, keepdim=True)).sqrt()
        return th.cat([x0, spatial], dim=-1)


class RiemannianSGD(th.optim.Optimizer):
    """Riemannian stochastic gradient descent (Bonnabel 2013). A parameter group with a
    ``manifold`` entry is updated as :math:`x \\leftarrow R_x(-\\eta\\, \\mathrm{grad}_R)`; a
    group without one by plain SGD.

    :param param_groups: List of dicts ``{"params": [...], "manifold": Manifold or None}``.
    :param lr: Step size :math:`\\eta`.
    """

    def __init__(self, param_groups, lr=0.01):
        groups = []
        for g in param_groups:
            g = dict(g)
            g.setdefault("manifold", None)
            groups.append(g)
        super().__init__(groups, dict(lr=lr))

    @th.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with th.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, manifold = group["lr"], group["manifold"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                g = th.nan_to_num(g)
                if manifold is None:
                    p.add_(g, alpha=-lr)
                else:
                    p.copy_(manifold.retract(p, -lr * manifold.rgrad(p, g)))
        return loss


def library_manifold(manifold):
    """The manifold as an object of this module. A direct implementation keeps its own copy of
    the manifolds (``mowl.baselines``, which the library does not import); a geometry that
    shares the tables of such a module takes the same manifold, as this module's class with
    the same name and attributes, so that ``isinstance`` checks of the measures hold."""
    if manifold is None or type(manifold).__module__ == __name__:
        return manifold
    cls = globals().get(type(manifold).__name__)
    if not (isinstance(cls, type) and issubclass(cls, Manifold)):
        return manifold
    copy = cls.__new__(cls)
    copy.__dict__.update(vars(manifold))
    return copy
