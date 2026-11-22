r"""Energy functions of hyperbolic entailment cones and shadow cones, in PyTorch.

All functions take a *parent* ``u`` (the apex of the cone; the class :math:`D` of an axiom
:math:`C \sqsubseteq D`) and a *child* ``v`` (:math:`C`), arrays of shape ``(..., d)``, and
return one number per pair. Curvature is :math:`-1`, as in the papers.

Entailment cones (Ganea et al. 2018, Poincare ball):

.. math::
    \psi(x) = \arcsin\frac{K(1-\lVert x\rVert^2)}{\lVert x\rVert},\qquad
    \Xi(x, y) = \arccos\frac{\langle x,y\rangle(1+\lVert x\rVert^2) - \lVert x\rVert^2
    (1+\lVert y\rVert^2)}{\lVert x\rVert\,\lVert x-y\rVert\sqrt{1+\lVert x\rVert^2
    \lVert y\rVert^2-2\langle x,y\rangle}},\qquad
    E(u, v) = \max(0, \Xi(u, v) - \psi(u)).

Shadow cones (Yu et al. 2024): the energy is the signed hyperbolic distance of :math:`v` to
the cone of :math:`u`, :math:`d(v, \mathrm{Cone}(u))`, with the two cases of Section 4:
:math:`d_{\mathbb H}(u, v)` when the relative altitude :math:`H(v, u) > 0`, and the signed
distance to the cone boundary otherwise (negative inside the cone). Four formulations:
``penumbral-ball``, ``penumbral-half-space``, ``umbral-ball``, ``umbral-half-space``.
"""

import math

import torch as th

from mowl.nn.lm import hyperbolic as H

_ACOS_CLAMP = 1e-6


def entailment_cone_min_norm(K):
    """The radius :math:`\\epsilon = 2K / (1 + \\sqrt{1 + 4K^2})` of the hole around the
    origin that the cone apertures leave out (Ganea et al. 2018, Eq. 25)."""
    return 2 * K / (1 + math.sqrt(1 + 4 * K * K))


def cone_half_aperture(x, K):
    """:math:`\\psi(x)` of Eq. 26."""
    n = x.norm(dim=-1).clamp_min(1e-9)
    arg = K * (1 - n * n) / n
    return th.asin(arg.clamp(max=1 - 1e-6))


def axis_angle(x, y, eps=1e-6):
    """:math:`\\Xi(x, y) = \\pi - \\angle O x y`: the angle at :math:`x` between the
    geodesic to :math:`y` and the cone axis (the ray from the origin through :math:`x`),
    Eq. 28."""
    xx = (x * x).sum(-1)
    yy = (y * y).sum(-1)
    xy = (x * y).sum(-1)
    num = xy * (1 + xx) - xx * (1 + yy)
    den = xx.clamp_min(1e-12).sqrt() * (x - y).pow(2).sum(-1).clamp_min(1e-12).sqrt()
    den = den * (1 + xx * yy - 2 * xy).clamp_min(1e-12).sqrt()
    return th.acos((num / den).clamp(-1 + _ACOS_CLAMP, 1 - _ACOS_CLAMP))


def entailment_cone_energy(u, v, K=0.1):
    """:math:`E(u, v) = \\max(0, \\Xi(u, v) - \\psi(u))` (Eq. 33); zero iff :math:`v` lies
    in the entailment cone of :math:`u`."""
    return th.relu(axis_angle(u, v) - cone_half_aperture(u, K))


# ---------------------------------------------------------------------------------------
# Shadow cones
# ---------------------------------------------------------------------------------------

FORMULATIONS = ("penumbral-ball", "penumbral-half-space", "umbral-ball", "umbral-half-space")


def _asinh(x):
    return th.asinh(x)


def _push_outward_ball(u, gamma3):
    """Move :math:`u` away from the origin along its ray by hyperbolic distance gamma3."""
    n = u.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    new = th.tanh(th.atanh(n.clamp(max=1 - 1e-7)) + gamma3 / 2)
    return u / n * new


def _push_down_half_space(u, gamma3):
    return th.cat([u[..., :-1], u[..., -1:] * math.exp(-gamma3)], dim=-1)


def _unit_angle(a, b):
    """The angle between two vectors, exactly 0 for equal directions (and zero gradient there).

    ``2 atan2(|a/|a| - b/|b||, |a/|a| + b/|b||)`` is accurate for every angle, unlike
    ``acos`` of the clamped cosine, whose clamp gave coincident points the angle 1.4e-3."""
    a = a / a.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    b = b / b.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    dif = ((a - b) ** 2).sum(-1)
    near = dif <= 1e-12
    dif = th.where(near, th.ones_like(dif), dif)
    angle = 2 * th.atan2(dif.sqrt(), ((a + b) ** 2).sum(-1).sqrt())
    return th.where(near, th.zeros_like(angle), angle)


def half_space_distance(x, y):
    """The hyperbolic distance in the upper half-space; exactly 0 (zero gradient) for points
    that coincide up to rounding, as in :mod:`mowl.nn.order.manifolds`."""
    diff = ((x - y) ** 2).sum(-1)
    arg = 1 + diff / (2 * x[..., -1] * y[..., -1])
    near = arg <= 1.0 + 1e-7
    safe = th.where(near, th.full_like(arg, 1.0 + 1e-6), arg)
    return th.where(near, th.zeros_like(arg), th.acosh(safe))


def penumbral_ball_half_aperture(u, r):
    """:math:`\\theta_u = \\arcsin(\\sinh r / \\sinh d_{\\mathbb H}(u, O))` (Formulation 3)."""
    n2 = (u * u).sum(-1)
    sinh_d = 2 * n2.clamp_min(1e-12).sqrt() / (1 - n2).clamp_min(1e-9)
    return th.asin((math.sinh(r) / sinh_d).clamp(max=1 - 1e-6))


def penumbral_half_space_half_aperture(u, horosphere_height):
    """:math:`\\theta_u = \\arcsin(u_n / e^h)` for the horosphere light source
    :math:`\\{x_n = e^h\\}` (Formulation 4, curvature -1)."""
    return th.asin((u[..., -1] / horosphere_height).clamp(max=1 - 1e-6))


def half_space_axis_angle(u, v):
    """The angle :math:`\\varphi(v, u)` at :math:`u` between the geodesic towards
    :math:`v` and the cone axis :math:`-e_n` (the vertical ray below :math:`u`), from the
    logarithm map of the half-space (Appendix F.1) without evaluating :math:`\\sinh s`."""
    un, vn = u[..., -1], v[..., -1]
    rho2 = ((u[..., :-1] - v[..., :-1]) ** 2).sum(-1)
    cosh_s = 1 + (rho2 + (un - vn) ** 2) / (2 * un * vn)
    a = cosh_s - un / vn
    cos_phi = -a / (rho2 / vn ** 2 + a * a).clamp_min(1e-12).sqrt()
    return th.acos(cos_phi.clamp(-1 + _ACOS_CLAMP, 1 - _ACOS_CLAMP))


def _penumbral_distance(dist_uv, dist_uprime_v, t):
    """Theorem F.2: ``dist(u', v)`` if t > pi/2 (altitude H = t - pi/2 > 0) else the signed
    distance to the boundary ``asinh(sinh d(u, v) sin t)``."""
    inside = _asinh(th.sinh(dist_uv.clamp(max=30)) * th.sin(t))
    return th.where(t > math.pi / 2, dist_uprime_v, inside)


def shadow_cone_energy(u, v, formulation="umbral-half-space", radius=0.05,
                       horosphere_height=20.0, gamma3=1e-4):
    """The distance of :math:`v` to the shadow cone of :math:`u`, :math:`d(v,
    \\mathrm{Cone}(u))` of Yu et al. (2024), Section 4 and Appendix F; negative when
    :math:`v` is strictly inside the cone, positive outside.

    In the case :math:`H(v, u) > 0` the paper's code uses :math:`d(u', v)` instead of
    :math:`d(u, v)`, with :math:`u'` the apex moved into the cone along its axis by
    ``gamma3`` (Appendix G), so that such points are pulled into the cone.

    :param formulation: One of :data:`FORMULATIONS`.
    :param radius: Radius :math:`r` of the ball light source (penumbral-ball) or of the
        balls attached to the points (umbral cones).
    :param horosphere_height: :math:`e^{h}` of the horosphere of penumbral-half-space.
    :param gamma3: See above.

    A point that coincides with the apex (up to rounding) is the apex itself and costs exactly
    0: the formulations evaluate it on the boundary of a branch, where rounding decides which
    branch applies (and the clamps below used to leave a small positive cost).
    """
    energy = _shadow_cone_energy(u, v, formulation, radius, horosphere_height, gamma3)
    coincident = ((u - v) ** 2).sum(-1) <= 1e-12
    return th.where(coincident, th.zeros_like(energy), energy)


def _shadow_cone_energy(u, v, formulation, radius, horosphere_height, gamma3):
    if formulation == "penumbral-ball":
        theta = penumbral_ball_half_aperture(u, radius)
        t = axis_angle(u, v) - theta
        up = _push_outward_ball(u, gamma3)
        return _penumbral_distance(H.distance(u, v, 1.0), H.distance(up, v, 1.0), t)
    if formulation == "penumbral-half-space":
        theta = penumbral_half_space_half_aperture(u, horosphere_height)
        t = half_space_axis_angle(u, v) - theta
        up = _push_down_half_space(u, gamma3)
        return _penumbral_distance(half_space_distance(u, v), half_space_distance(up, v), t)
    if formulation == "umbral-half-space":
        un, vn = u[..., -1], v[..., -1]
        rho = ((u[..., :-1] - v[..., :-1]) ** 2).sum(-1).clamp_min(1e-12).sqrt()
        t = (rho - un * math.sinh(radius)) / vn
        altitude = vn ** 2 * (1 + t ** 2) - un ** 2 * math.cosh(radius) ** 2
        signed = _asinh(t) + radius
        up = _push_down_half_space(u, gamma3)
        return th.where(altitude > 0, half_space_distance(up, v), signed)
    if formulation == "umbral-ball":
        nu = (u * u).sum(-1)
        nv = (v * v).sum(-1)
        sinh_du = 2 * nu.clamp_min(1e-12).sqrt() / (1 - nu).clamp_min(1e-9)
        sinh_dv = 2 * nv.clamp_min(1e-12).sqrt() / (1 - nv).clamp_min(1e-9)
        # the antipodal end keeps the published clamp (alpha <= acos(-1 + 1e-6)); the coincident
        # end is exact
        alpha = _unit_angle(u, v).clamp(max=math.acos(-1 + _ACOS_CLAMP))
        beta = th.asin((math.sinh(radius) / sinh_du).clamp(max=1 - 1e-6))
        t = sinh_dv * th.sin(alpha - beta)
        signed = _asinh(t) + radius
        cosh_du = th.sqrt(1 + sinh_du ** 2)
        cosh_dv = th.sqrt(1 + sinh_dv ** 2)
        h_v = th.acosh((cosh_dv / th.sqrt(1 + t ** 2)).clamp_min(1.0))
        h_u = th.acosh((cosh_du / math.cosh(radius)).clamp_min(1.0))
        up = _push_outward_ball(u, gamma3)
        # H(v) and H(u) are the distances from the light source O to the feet of v and u on
        # the light path; the altitude of v above u grows towards the light, so it is
        # H(u) - H(v) (the authors' code; the printed H(v, u) has the opposite sign).
        return th.where(h_u - h_v > 0, H.distance(up, v, 1.0), signed)
    raise ValueError(f"formulation must be one of {FORMULATIONS}, got {formulation!r}.")


def shadow_min_norm(formulation, radius):
    """Euclidean norm below which a point of the Poincare ball lies in the hole of radius
    ``radius`` around the light source (Remark 1)."""
    if formulation.endswith("half-space"):
        return None
    return math.tanh(radius / 2) * 1.02 + 1e-4
