"""Operations on the Poincaré ball, written with PyTorch.

The :math:`d`-dimensional Poincaré ball of curvature :math:`-c` (:math:`c > 0`) is the open
ball :math:`\\mathbb{B}^d_c = \\{x \\in \\mathbb{R}^d : c\\lVert x\\rVert^2 < 1\\}` of radius
:math:`1/\\sqrt{c}`. The functions follow the definitions used by HiT (He et al., NeurIPS
2024, Section 2.2, Eqs. 1-2) and OnT (Yang et al., ISWC 2025, Section 3.2, Eqs. 2-3), which
take them from Ganea et al. (2018, "Hyperbolic neural networks"):

* Möbius addition
  :math:`x \\oplus_c y = \\frac{(1 + 2c\\langle x, y\\rangle + c\\lVert y\\rVert^2)x +
  (1 - c\\lVert x\\rVert^2)y}{1 + 2c\\langle x, y\\rangle + c^2\\lVert x\\rVert^2
  \\lVert y\\rVert^2}`,
* distance :math:`d_c(x, y) = \\frac{2}{\\sqrt{c}}\\tanh^{-1}(\\sqrt{c}\\lVert -x \\oplus_c y
  \\rVert)`, equal to OnT's :math:`\\frac{1}{\\sqrt{c}}\\operatorname{arcosh}(1 +
  \\frac{2c\\lVert x - y\\rVert^2}{(1 - c\\lVert x\\rVert^2)(1 - c\\lVert y\\rVert^2)})`,
* hyperbolic norm :math:`\\lVert x\\rVert_c = d_c(x, 0)`,
* exponential and logarithmic maps, Möbius scalar multiplication (OnT Eq. 3 for
  :math:`c = 1`), and the block rotations OnT uses for roles (Section 4.2).

Numerical safety follows the reference implementations, which use ``geoopt``:
:math:`\\tanh^{-1}` is evaluated on arguments clamped to :math:`[-1 + 10^{-7}, 1 - 10^{-7}]`,
norms in denominators are clamped below by :math:`10^{-15}`, and :func:`project` pulls
points back inside the ball with a relative margin of :math:`4 \\cdot 10^{-3}` (float32) or
:math:`10^{-5}` (float64). As in the reference code, encoder outputs are not projected
before the losses are computed (HiT, Section 3, argues that the outputs of a ``tanh``
layer lie in the hypercube that the ball of radius :math:`\\sqrt{d}` circumscribes); points
outside the ball saturate in :func:`artanh`.
"""

import math

import torch as th

#: Clamp of the argument of :math:`\\tanh^{-1}` (as in ``geoopt``).
ARTANH_EPS = 1e-7
#: Lower bound of norms used as denominators (as in ``geoopt``).
MIN_NORM = 1e-15
#: Relative distance from the boundary kept by :func:`project`, by dtype (``geoopt``).
PROJECTION_EPS = {th.float32: 4e-3, th.float64: 1e-5}


def curvature_for_dim(dim):
    """Curvature :math:`c = 1/d` of the ball of radius :math:`\\sqrt{d}` that circumscribes
    the hypercube :math:`[-1, 1]^d` (HiT, Section 3).

    :param dim: Embedding dimension :math:`d`.
    :rtype: float
    """
    if dim <= 0:
        raise ValueError("dim must be positive.")
    return 1.0 / dim


def _check_c(c):
    if not c > 0:
        raise ValueError(f"The curvature parameter c must be positive, got {c}.")
    return c


def artanh(x):
    """:math:`\\tanh^{-1}` with the argument clamped to :math:`(-1, 1)`."""
    return th.atanh(x.clamp(-1 + ARTANH_EPS, 1 - ARTANH_EPS))


def tanh(x):
    """:math:`\\tanh` with the argument clamped to :math:`[-15, 15]`."""
    return th.tanh(x.clamp(-15, 15))


def _norm(x, keepdim=True):
    return x.norm(dim=-1, p=2, keepdim=keepdim).clamp_min(MIN_NORM)


def mobius_add(x, y, c):
    """Möbius addition :math:`x \\oplus_c y` (HiT, Eq. 2).

    :param x: Tensor of shape ``(..., d)``.
    :param y: Tensor broadcastable to ``x``.
    :param c: Curvature parameter :math:`c > 0`.
    """
    _check_c(c)
    xy = (x * y).sum(dim=-1, keepdim=True)
    x2 = (x * x).sum(dim=-1, keepdim=True)
    y2 = (y * y).sum(dim=-1, keepdim=True)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    den = 1 + 2 * c * xy + c ** 2 * x2 * y2
    return num / den.clamp_min(MIN_NORM)


def distance(x, y, c):
    """Hyperbolic distance :math:`d_c(x, y) = \\frac{2}{\\sqrt{c}}\\tanh^{-1}(\\sqrt{c}
    \\lVert -x \\oplus_c y\\rVert)` (HiT, Eq. 1).

    :return: Tensor of shape ``(...)`` (the last dimension is reduced).
    """
    sqrt_c = math.sqrt(_check_c(c))
    diff = mobius_add(-x, y, c).norm(dim=-1, p=2)
    return 2.0 / sqrt_c * artanh(sqrt_c * diff)


def pairwise_distance(x, y, c, dtype=th.float64):
    """All distances between the rows of ``x`` (``(n, d)``) and ``y`` (``(m, d)``), shape
    ``(n, m)``, without materialising ``(n, m, d)`` tensors. Uses the identity
    :math:`\\lVert -x \\oplus_c y\\rVert^2 = \\frac{\\lVert x - y\\rVert^2}{1 -
    2c\\langle x, y\\rangle + c^2\\lVert x\\rVert^2\\lVert y\\rVert^2}`; computed and
    returned in ``dtype`` (float64 by default, for ranking)."""
    sqrt_c = math.sqrt(_check_c(c))
    x64, y64 = x.to(dtype), y.to(dtype)
    x2 = (x64 * x64).sum(dim=-1, keepdim=True)
    y2 = (y64 * y64).sum(dim=-1, keepdim=True).transpose(0, 1)
    xy = x64 @ y64.transpose(0, 1)
    diff2 = (x2 + y2 - 2 * xy).clamp_min(0)
    den = (1 - 2 * c * xy + c ** 2 * x2 * y2).clamp_min(MIN_NORM)
    return 2.0 / sqrt_c * artanh(sqrt_c * th.sqrt(diff2 / den))


def distance_arcosh(x, y, c):
    """The same distance in the closed form of OnT, Eq. 2:
    :math:`\\frac{1}{\\sqrt{c}}\\operatorname{arcosh}\\left(1 + \\frac{2c\\lVert x -
    y\\rVert^2}{(1 - c\\lVert x\\rVert^2)(1 - c\\lVert y\\rVert^2)}\\right)`.

    Less accurate than :func:`distance` for nearby points; provided as a cross-check.
    """
    sqrt_c = math.sqrt(_check_c(c))
    diff2 = ((x - y) ** 2).sum(dim=-1)
    den = ((1 - c * (x * x).sum(dim=-1)) * (1 - c * (y * y).sum(dim=-1))).clamp_min(MIN_NORM)
    arg = 1 + 2 * c * diff2 / den
    return th.acosh(arg.clamp_min(1.0)) / sqrt_c


def hyperbolic_norm(x, c):
    """Hyperbolic norm :math:`\\lVert x\\rVert_c = d_c(x, 0) = \\frac{2}{\\sqrt{c}}
    \\tanh^{-1}(\\sqrt{c}\\lVert x\\rVert)` (HiT, Eq. 4)."""
    sqrt_c = math.sqrt(_check_c(c))
    return 2.0 / sqrt_c * artanh(sqrt_c * x.norm(dim=-1, p=2))


def conformal_factor(x, c):
    """:math:`\\lambda^c_x = 2 / (1 - c\\lVert x\\rVert^2)`, shape ``(..., 1)``."""
    return 2.0 / (1 - c * (x * x).sum(dim=-1, keepdim=True)).clamp_min(MIN_NORM)


def expmap0(u, c):
    """Exponential map at the origin, :math:`\\tanh(\\sqrt{c}\\lVert u\\rVert)
    \\frac{u}{\\sqrt{c}\\lVert u\\rVert}`."""
    sqrt_c = math.sqrt(_check_c(c))
    n = _norm(u)
    return tanh(sqrt_c * n) * u / (sqrt_c * n)


def logmap0(y, c):
    """Logarithmic map at the origin, the inverse of :func:`expmap0`:
    :math:`\\tanh^{-1}(\\sqrt{c}\\lVert y\\rVert)\\frac{y}{\\sqrt{c}\\lVert y\\rVert}`."""
    sqrt_c = math.sqrt(_check_c(c))
    n = _norm(y)
    return artanh(sqrt_c * n) * y / (sqrt_c * n)


def expmap(x, u, c):
    """Exponential map at :math:`x`: :math:`x \\oplus_c \\left(\\tanh\\left(\\frac{\\sqrt{c}
    \\lambda^c_x\\lVert u\\rVert}{2}\\right)\\frac{u}{\\sqrt{c}\\lVert u\\rVert}\\right)`."""
    sqrt_c = math.sqrt(_check_c(c))
    n = _norm(u)
    second = tanh(sqrt_c * conformal_factor(x, c) * n / 2) * u / (sqrt_c * n)
    return mobius_add(x, second, c)


def logmap(x, y, c):
    """Logarithmic map at :math:`x`, the inverse of :func:`expmap`:
    :math:`\\frac{2}{\\sqrt{c}\\lambda^c_x}\\tanh^{-1}(\\sqrt{c}\\lVert -x \\oplus_c y\\rVert)
    \\frac{-x \\oplus_c y}{\\lVert -x \\oplus_c y\\rVert}`."""
    sqrt_c = math.sqrt(_check_c(c))
    sub = mobius_add(-x, y, c)
    n = _norm(sub)
    return 2 / (sqrt_c * conformal_factor(x, c)) * artanh(sqrt_c * n) * sub / n


def mobius_scalar_mul(r, x, c):
    """Möbius scalar multiplication :math:`r \\otimes_c x = \\tanh(r\\tanh^{-1}(\\sqrt{c}
    \\lVert x\\rVert))\\frac{x}{\\sqrt{c}\\lVert x\\rVert}` (OnT, Eq. 3, for :math:`c = 1`).

    :param r: Scalar or tensor broadcastable to ``x[..., :1]``.
    """
    sqrt_c = math.sqrt(_check_c(c))
    n = _norm(x)
    if isinstance(r, th.Tensor) and r.dim() == x.dim() - 1:
        r = r.unsqueeze(-1)
    return tanh(r * artanh(sqrt_c * n)) * x / (sqrt_c * n)


def project(x, c, eps=None):
    """Pull points whose norm exceeds :math:`(1 - \\epsilon)/\\sqrt{c}` back onto that
    sphere; points inside are returned unchanged.

    :param eps: Relative margin; defaults to :data:`PROJECTION_EPS` for the dtype.
    """
    sqrt_c = math.sqrt(_check_c(c))
    if eps is None:
        eps = PROJECTION_EPS.get(x.dtype, 1e-5)
    maxnorm = (1 - eps) / sqrt_c
    n = _norm(x)
    return th.where(n > maxnorm, x / n * maxnorm, x)


def rotate(x, angles, pairing="half"):
    """Rotate every point by a block-diagonal rotation :math:`R(\\Theta)` made of
    :math:`m = d/2` planar rotations (OnT, Section 4.2, Eq. 4).

    Rotations about the origin are Euclidean isometries that fix the origin, hence also
    isometries of the Poincaré ball: they preserve hyperbolic distances and norms (OnT,
    Proposition 1).

    :param x: Tensor of shape ``(..., d)`` with ``d`` even.
    :param angles: Tensor of shape ``(..., d/2)`` (broadcastable).
    :param pairing: Which coordinates form a plane. ``"half"`` rotates
        :math:`(x_i, x_{i+m})`, as the reference implementation of OnT does;
        ``"adjacent"`` rotates :math:`(x_{2i}, x_{2i+1})`, the literal block-diagonal
        matrix printed in the paper. Both are rotations; they differ by a fixed
        permutation of the coordinates.
    """
    d = x.shape[-1]
    if d % 2:
        raise ValueError("Rotations need an even embedding dimension.")
    cos, sin = th.cos(angles), th.sin(angles)
    if pairing == "half":
        a, b = x[..., : d // 2], x[..., d // 2:]
        return th.cat([a * cos - b * sin, a * sin + b * cos], dim=-1)
    if pairing == "adjacent":
        a, b = x[..., 0::2], x[..., 1::2]
        out = th.stack([a * cos - b * sin, a * sin + b * cos], dim=-1)
        return out.reshape(x.shape)
    raise ValueError(f"Unknown pairing {pairing!r}; use 'half' or 'adjacent'.")
