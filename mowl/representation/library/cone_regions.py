r"""Cones with an apex and an aperture, on hyperbolic manifolds (entailment cones, shadow cones).

A cone is a *region*: its apex is a point of a manifold, and the light that casts it says
where the cone opens. The classes here are written on the distance of the manifold
(:mod:`mowl.nn.order.manifolds`) and on the Möbius addition of
:mod:`mowl.representation.library.poincare_ball`; they do not use the formulas of the direct modules in
:mod:`mowl.baselines`. The measures that compare a point with a cone are in
:mod:`~mowl.representation.library.hierarchy_measures`.

*Light sources* (:class:`OriginLight`, :class:`HorosphereLight`, :class:`InfinityLight`) are the
part that differs between the formulations of the shadow cones paper (Yu et al. 2024, Sections 3
and 4): the origin of the Poincaré ball (a ball of radius :math:`r`), a horosphere of the
half-space, a point at infinity of the half-space. Each light gives

* the half-aperture :math:`\theta_u` of the cone of a *point* apex (penumbra),
* the angle :math:`\varphi(u, v)` at the apex between the axis (away from the light) and the
  geodesic to a point :math:`v`,
* the offset of a point from the boundary of the cone of a *ball* of radius :math:`r` around the
  apex (umbra), and whether the point is closer to the light than the tangent point,
* the apex moved into the cone along the axis.

*The distance of a point to a cone* (:meth:`Shadow.signed_distance`) is the one of Section 4:
with :math:`t = \varphi - \theta` the angle at the apex between the point and the boundary,
:math:`d = \operatorname{arsinh}(\sinh d(u, v) \sin t)` (negative inside the cone) when
:math:`t < \pi/2`, and the distance to the apex otherwise; for umbral cones the boundary line
does not pass through the apex, and the point is at the apex's distance when it lies beyond the
tangent point towards the light.

Everything has curvature :math:`-1`, as in the papers.
"""

import math
from dataclasses import dataclass

import torch as th

from mowl.representation.library import poincare_ball as H
from mowl.nn.order.manifolds import PoincareBall, UpperHalfSpace
from mowl.representation.library.hierarchy_regions import Point
from mowl.representation.regions import ConstructorUnavailable

__all__ = ["Light", "OriginLight", "HorosphereLight", "InfinityLight", "Shadow",
           "FORMULATIONS", "light_for", "entailment_cone_min_norm", "shadow_min_norm",
           "cone_half_aperture", "axis_angle", "entailment_cone_energy",
           "shadow_cone_energy"]

FORMULATIONS = ("penumbral-ball", "penumbral-half-space", "umbral-ball", "umbral-half-space")

#: Guard of :func:`torch.asin` and of :func:`torch.sinh`: a point at the light source has no
#: cone (the aperture saturates) and a distance beyond 30 would overflow the sine.
_ASIN_MAX = 1 - 1e-6
_DISTANCE_MAX = 30.0
#: Points closer than this (squared Euclidean distance) are the same point.
_SAME_POINT = 1e-12


def _unit(x):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def _angle_between(a, b, cos_clamp=0.0):
    """The angle between two vectors in :math:`[0, \\pi]`. Exact (``atan2`` of the sine and the
    cosine) unless ``cos_clamp > 0``, which reads the arccosine of the cosine clamped to
    :math:`[-1 + c, 1 - c]` (the form of the published formulas)."""
    if cos_clamp > 0:
        cos = (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1)).clamp_min(1e-12)
        return th.acos(cos.clamp(-1 + cos_clamp, 1 - cos_clamp))
    a, b = _unit(a), _unit(b)
    dot = (a * b).sum(-1)
    off = (b - dot.unsqueeze(-1) * a).norm(dim=-1)
    return th.atan2(off, dot)


class Light:
    """What casts a cone: see the module docstring. ``manifold_type`` is the space the light
    lives in."""

    manifold_type = None

    def check(self, manifold):
        if not isinstance(manifold, self.manifold_type):
            raise ConstructorUnavailable(
                f"{type(self).__name__} needs points of {self.manifold_type.__name__}, got "
                f"{type(manifold).__name__}.")

    def half_aperture(self, manifold, apex):
        raise NotImplementedError

    def axis_angle(self, manifold, apex, x, cos_clamp=0.0):
        raise NotImplementedError

    def push(self, manifold, apex, s):
        """``apex`` moved into its cone along the axis by the hyperbolic distance ``s``."""
        raise NotImplementedError

    def umbra(self, manifold, apex, x, radius):
        """``(t, closer)``: :math:`t = \\sinh` of the signed distance of ``x`` to the boundary
        of the shadow of the ball of radius ``radius`` around ``apex`` (negative inside), and
        whether the foot of ``x`` on the boundary is closer to the light than the tangent
        point of the ball."""
        raise NotImplementedError


class OriginLight(Light):
    """A ball of radius ``radius`` at the origin of the Poincaré ball (Formulations 3 and 5 of
    Yu et al.). For a point apex :math:`u` the cone is that of half-aperture :math:`\\arcsin
    (\\sinh r / \\sinh d(u, O))` (law of sines in the triangle of the apex, the origin and the
    tangent point), which is the entailment cone of Ganea et al. with :math:`K = \\sinh(r)/2`
    (Theorem 3.2 of Yu et al.)."""

    manifold_type = PoincareBall

    def __init__(self, radius):
        self.radius = float(radius)

    @staticmethod
    def _sinh_distance(x):
        """:math:`\\sinh d(O, x) = 2\\lVert x\\rVert / (1 - \\lVert x\\rVert^2)`."""
        n2 = (x * x).sum(-1)
        return 2 * n2.clamp_min(1e-12).sqrt() / (1 - n2).clamp_min(1e-9)

    def half_aperture(self, manifold, apex):
        return th.asin((math.sinh(self.radius) / self._sinh_distance(apex)).clamp(max=_ASIN_MAX))

    def axis_angle(self, manifold, apex, x, cos_clamp=0.0):
        # The Moebius translation taking the apex to the origin is conformal, so it keeps the
        # angle at the apex. It takes the origin to -apex, i.e. the axis (away from the light)
        # to the direction +apex, and the point to (-apex) (+) x.
        moved = H.mobius_add(-apex, x, 1.0)
        return _angle_between(apex, moved, cos_clamp)

    def push(self, manifold, apex, s):
        # distance from the origin 2 atanh(n): the apex moves outward by s
        n = apex.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return apex / n * th.tanh(th.atanh(n.clamp(max=1 - 1e-7)) + s / 2)

    def umbra(self, manifold, apex, x, radius):
        sinh_u, sinh_x = self._sinh_distance(apex), self._sinh_distance(x)
        # the boundary of the shadow of B(u, r) lit from the origin is the geodesic through the
        # origin tangent to the ball, at angle beta from the direction of u
        beta = th.asin((math.sinh(radius) / sinh_u).clamp(max=_ASIN_MAX))
        alpha = _angle_between(apex, x)             # angle at the origin between u and x
        t = sinh_x * th.sin(alpha - beta)
        cosh_u, cosh_x = th.sqrt(1 + sinh_u ** 2), th.sqrt(1 + sinh_x ** 2)
        # distances from the origin along the boundary to the foot of x and to the tangent point
        foot = th.acosh((cosh_x / th.sqrt(1 + t ** 2)).clamp_min(1.0))
        tangent = th.acosh((cosh_u / math.cosh(radius)).clamp_min(1.0))
        return t, foot < tangent


class HorosphereLight(Light):
    """The horosphere :math:`x_n = e^h` of the upper half-space as the light source (penumbral
    half-space, Formulation 4); the cone of a point apex has half-aperture
    :math:`\\arcsin(u_n / e^h)` and its axis is the vertical ray below the apex."""

    manifold_type = UpperHalfSpace

    def __init__(self, height):
        self.height = float(height)

    def half_aperture(self, manifold, apex):
        return th.asin((apex[..., -1] / self.height).clamp(max=_ASIN_MAX))

    def axis_angle(self, manifold, apex, x, cos_clamp=0.0):
        # The geodesic from u to x is a semicircle centred on the boundary at distance lam
        # from u's horizontal position towards x, with lam**2 + u_n**2 = (rho - lam)**2 + x_n**2.
        # Its tangent at u is (u_n, lam) (horizontal, vertical), and the axis is (0, -1), so
        # the angle is atan2(u_n, -lam) = atan2(2 rho u_n, u_n**2 - x_n**2 - rho**2).
        un, xn = apex[..., -1], x[..., -1]
        rho = ((apex[..., :-1] - x[..., :-1]) ** 2).sum(-1).clamp_min(1e-12).sqrt()
        return th.atan2(2 * rho * un, un ** 2 - xn ** 2 - rho ** 2)

    def push(self, manifold, apex, s):
        return th.cat([apex[..., :-1], apex[..., -1:] * math.exp(-s)], dim=-1)


class InfinityLight(HorosphereLight):
    """A point at infinity of the upper half-space (the light of the umbral half-space
    formulation, the best performing one in Yu et al.): the rays are vertical, the cone of a
    ball of radius :math:`r` is the vertical cylinder of horizontal half-width :math:`u_n \\sinh
    r` below it, whose wall is a hyperbolic hyperplane."""

    def __init__(self):
        super().__init__(height=float("inf"))

    def half_aperture(self, manifold, apex):
        raise ConstructorUnavailable("The light at infinity casts umbral cones only.")

    def umbra(self, manifold, apex, x, radius):
        un, xn = apex[..., -1], x[..., -1]
        rho = ((apex[..., :-1] - x[..., :-1]) ** 2).sum(-1).clamp_min(1e-12).sqrt()
        t = (rho - un * math.sinh(radius)) / xn
        # the foot of x on the wall is at height x_n sqrt(1 + t**2), the tangent point of the
        # ball at height u_n cosh r; higher is closer to the light
        return t, xn ** 2 * (1 + t ** 2) > (un * math.cosh(radius)) ** 2


def light_for(formulation, radius=0.05, horosphere_height=20.0):
    """The light source and the kind (umbral or penumbral) of a formulation of
    :data:`FORMULATIONS`: ``(light, umbral)``."""
    if formulation == "penumbral-ball":
        return OriginLight(radius), False
    if formulation == "umbral-ball":
        return OriginLight(radius), True
    if formulation == "penumbral-half-space":
        return HorosphereLight(horosphere_height), False
    if formulation == "umbral-half-space":
        return InfinityLight(), True
    raise ValueError(f"formulation must be one of {FORMULATIONS}, got {formulation!r}.")


@dataclass(frozen=True)
class Shadow(Point):
    """The cone at the apex ``centre`` cast by ``light`` (``umbral``: of a ball of radius
    ``radius`` around the apex, else of the point). ``gamma3`` is the push of the apex into the
    cone that the distance uses where a point lies beyond the apex (Appendix G of Yu et al.:
    such points are pulled into the cone)."""

    light: Light = None
    umbral: bool = False
    radius: float = 0.05
    gamma3: float = 1e-4

    def half_aperture(self):
        """Half-aperture of the cone of a point apex (penumbra), ``(B,)``."""
        return self.light.half_aperture(self.manifold, self.centre)

    def signed_distance(self, points):
        """The distance of ``points`` to the cone (Section 4 of Yu et al.), negative inside,
        positive outside; exactly 0 for a point that is the apex, where which side of the
        boundary applies is decided by rounding."""
        manifold, apex, light = self.manifold, self.centre, self.light
        light.check(manifold)
        pushed = light.push(manifold, apex, self.gamma3)
        beyond_apex = manifold.distance(pushed, points)
        if self.umbral:
            t, closer = light.umbra(manifold, apex, points, self.radius)
            signed = th.asinh(t) + self.radius
        else:
            t = light.axis_angle(manifold, apex, points) - light.half_aperture(manifold, apex)
            sinh_d = th.sinh(manifold.distance(apex, points).clamp(max=_DISTANCE_MAX))
            signed = th.asinh(sinh_d * th.sin(t))
            closer = t > math.pi / 2
        energy = th.where(closer, beyond_apex, signed)
        same = ((apex - points) ** 2).sum(-1) <= _SAME_POINT
        return th.where(same, th.zeros_like(energy), energy)


# -- entailment cones ---------------------------------------------------------------------
def entailment_cone_min_norm(K):
    """The radius :math:`\\epsilon = 2K / (1 + \\sqrt{1 + 4K^2})` of the hole around the origin
    that the apertures of the entailment cones leave out (Ganea et al. 2018, Eq. 25)."""
    return 2 * K / (1 + math.sqrt(1 + 4 * K * K))


def shadow_min_norm(formulation, radius):
    """Euclidean norm below which a point of the Poincaré ball lies in the hole of radius
    ``radius`` around the light source (Remark 1 of Yu et al.); ``None`` for the half-space."""
    if formulation.endswith("half-space"):
        return None
    return math.tanh(radius / 2) * 1.02 + 1e-4


def _entailment_light(K):
    return OriginLight(math.asinh(2 * K))


def cone_half_aperture(x, K):
    """:math:`\\psi(x) = \\arcsin(K(1 - \\lVert x\\rVert^2)/\\lVert x\\rVert)` (Ganea et al.,
    Eq. 26): the half-aperture of the cone of a point apex of the Poincaré ball, the
    penumbral cone of a light of radius :math:`\\operatorname{arsinh}(2K)`."""
    return _entailment_light(K).half_aperture(None, x)


def axis_angle(x, y, cos_clamp=0.0):
    """:math:`\\Xi(x, y)`: the angle at :math:`x` between the cone axis (the ray from the origin
    through :math:`x`) and the geodesic to :math:`y` (Ganea et al., Eq. 28); ``cos_clamp`` as
    in :func:`_angle_between`."""
    return OriginLight(1.0).axis_angle(None, x, y, cos_clamp)


def entailment_cone_energy(u, v, K=0.1):
    """:math:`\\max(0, \\Xi(u, v) - \\psi(u))` (Eq. 33)."""
    return th.relu(axis_angle(u, v) - cone_half_aperture(u, K))


def shadow_cone_energy(u, v, formulation="umbral-half-space", radius=0.05,
                       horosphere_height=20.0, gamma3=1e-4):
    """:meth:`Shadow.signed_distance` of the points ``v`` to the cones of the apexes ``u`` of a
    formulation; the manifold is the ball or the half-space that the formulation lives on."""
    light, umbral = light_for(formulation, radius, horosphere_height)
    manifold = UpperHalfSpace() if formulation.endswith("half-space") else PoincareBall()
    return Shadow(u, manifold, None, light, umbral, radius, gamma3).signed_distance(v)
