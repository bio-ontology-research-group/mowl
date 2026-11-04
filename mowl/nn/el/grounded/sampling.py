"""Element and region samplers for GroundedEL (§3–§4).

Provides deterministic-with-generator sampling utilities for:

1. **Element sampling (:func:`sample_elements`, :func:`sample_filler_elements`):**
   Combines the mandatory LHS element(s) :math:`x_A`, uniform random domain
   elements drawn from ``[0, nb_elements)``, and optional in-batch elements.
2. **Region component sampling (:func:`sample_active_component`,
   :func:`sample_class_region`):** Draws points uniformly inside a randomly
   selected active non-empty component box of a class region.
3. **Component-pair intersection sampling (:func:`sample_component_intersection`,
   :func:`sample_pair_region`):** Draws points uniformly inside the intersection
   box of a randomly selected non-empty active component pair :math:`A_i \\cap B_j`,
   marking rows with no non-empty intersection as ``valid=False`` so callers
   skip them.
4. **Role product-box sampling (:func:`sample_role_region`):** Draws pairs
   :math:`(x, y)` uniformly inside a randomly selected active product box
   :math:`H_{r,k} \\times T_{r,k}`.
5. **Adversarial signed-gradient ascent (:func:`adversarial_ascent`):** Performs
   ``n_adv`` steps of signed-gradient ascent on a hinge function clipped to the
   sampled component or intersection box.

Guarantees:
- **Explicit generator discipline:** Every random draw uses the caller-supplied
  :class:`torch.Generator`, ensuring exact reproducibility without mutating
  global RNG state.
- **Strict box containment:** Uniformly sampled points and adversarially updated
  points always lie inside the closed bounding box ``[lower, upper]`` of the
  selected active component or intersection; when ``n_adv == 0`` and ``w > 0``,
  sampled points lie strictly inside the open interior ``(lower, upper)``.
- **Empty-region safety:** Rows with no non-empty active component (or no
  non-empty pairwise intersection) return ``valid[b] == False`` and finite zero
  coordinates.

Non-guarantees:
- Uniform choice over active non-empty components weights components equally
  regardless of their Euclidean volume.
"""
from __future__ import annotations

from typing import Callable, NamedTuple, Optional, Sequence, Tuple, Union

import torch as th

from mowl.nn.el.grounded.module import GroundedELModule
from mowl.nn.el.structural import Regions

__all__ = [
    "RegionSample",
    "sample_elements",
    "sample_filler_elements",
    "adversarial_ascent",
    "sample_active_component",
    "sample_component_intersection",
    "sample_class_region",
    "sample_pair_region",
    "sample_role_region",
]

_INTERIOR_EPS: float = 1e-6


class RegionSample(NamedTuple):
    """Sampled region points together with validity mask and enclosing box bounds.

    Attributes
    ----------
    points : torch.Tensor
        Sampled point coordinates of shape ``(B, d)``.
    valid : torch.Tensor
        Boolean mask of shape ``(B,)`` indicating which rows had at least one
        non-empty active box (or intersection box).
    lower : torch.Tensor
        Lower box bounds of shape ``(B, d)`` for the selected component.
    upper : torch.Tensor
        Upper box bounds of shape ``(B, d)`` for the selected component.
    """

    points: th.Tensor
    valid: th.Tensor
    lower: th.Tensor
    upper: th.Tensor


def _validate_generator(generator: th.Generator) -> th.Generator:
    """Validate that ``generator`` is an instance of :class:`torch.Generator`."""
    if not isinstance(generator, th.Generator):
        raise TypeError(
            f"generator must be a torch.Generator instance, got {type(generator).__name__}"
        )
    return generator


def _rand_with_generator(
    shape: Tuple[int, ...],
    dtype: th.dtype,
    device: th.device,
    generator: th.Generator,
) -> th.Tensor:
    """Draw uniform ``[0, 1)`` tensor using ``generator`` and move to ``device``."""
    gen_dev = generator.device
    u = th.rand(shape, dtype=dtype, device=gen_dev, generator=generator)
    return u if u.device == device else u.to(device=device)


def _randint_with_generator(
    low: int,
    high: int,
    shape: Tuple[int, ...],
    device: th.device,
    generator: th.Generator,
) -> th.Tensor:
    """Draw uniform integer tensor in ``[low, high)`` using ``generator`` and move to ``device``."""
    gen_dev = generator.device
    out = th.randint(low, high, shape, dtype=th.long, device=gen_dev, generator=generator)
    return out if out.device == device else out.to(device=device)


def sample_elements(
    lhs_elements: th.Tensor,
    nb_elements: int,
    n_random: int,
    generator: th.Generator,
    batch_elements: Optional[th.Tensor] = None,
) -> th.Tensor:
    """Sample candidate domain elements combining LHS elements, uniform random elements, and optional batch elements.

    Implements §3 element sampling: for each row ``b`` of ``lhs_elements``,
    concatenates (i) ``lhs_elements[b]`` (always included), (ii) ``n_random``
    elements drawn uniformly from ``[0, nb_elements)``, and (iii) ``batch_elements``
    when provided.

    Guarantees:
    - Returns a 2-D ``torch.long`` tensor of shape ``(B, L + n_random + M)`` on
      ``lhs_elements.device``.
    - Leading column(s) ``[:, :L]`` equal ``lhs_elements``.
    - Uses only ``generator`` for random draws.

    Non-guarantees:
    - Does not deduplicate element indices within a row.

    Parameters
    ----------
    lhs_elements : torch.Tensor
        1-D ``(B,)`` or 2-D ``(B, L)`` tensor of LHS element indices.
    nb_elements : int
        Total number of domain elements :math:`|\\Delta| \\ge 1`.
    n_random : int
        Number of uniform random elements :math:`\\ge 0` per row.
    generator : torch.Generator
        Explicit PyTorch RNG.
    batch_elements : Optional[torch.Tensor], default None
        Optional 1-D ``(M,)`` or 2-D ``(B, M)`` tensor of in-batch element indices.

    Returns
    -------
    torch.Tensor
        2-D tensor of sampled element indices of shape ``(B, S)``.
    """
    _validate_generator(generator)
    if not isinstance(lhs_elements, th.Tensor):
        raise TypeError(
            f"lhs_elements must be a torch.Tensor, got {type(lhs_elements).__name__}"
        )
    if isinstance(nb_elements, bool) or not isinstance(nb_elements, int) or nb_elements < 1:
        raise ValueError(f"nb_elements must be an int >= 1, got {nb_elements!r}")
    if isinstance(n_random, bool) or not isinstance(n_random, int) or n_random < 0:
        raise ValueError(f"n_random must be an int >= 0, got {n_random!r}")

    device = lhs_elements.device
    lhs = lhs_elements.to(dtype=th.long)
    if lhs.ndim == 1:
        lhs = lhs.unsqueeze(1)
    elif lhs.ndim != 2:
        raise ValueError(
            f"lhs_elements must be 1-D or 2-D, got shape {tuple(lhs_elements.shape)}"
        )

    bsz = int(lhs.shape[0])
    parts = [lhs]

    if n_random > 0:
        rand_elems = _randint_with_generator(
            0, nb_elements, (bsz, n_random), device=device, generator=generator
        )
        parts.append(rand_elems)

    if batch_elements is not None and batch_elements.numel() > 0:
        be = batch_elements.to(device=device, dtype=th.long)
        if be.ndim == 1:
            parts.append(be.unsqueeze(0).expand(bsz, be.shape[0]))
        elif be.ndim == 2 and be.shape[0] == bsz:
            parts.append(be)
        else:
            raise ValueError(
                f"batch_elements must have shape (M,) or ({bsz}, M), got {tuple(be.shape)}"
            )

    return th.cat(parts, dim=1)


def sample_filler_elements(
    filler_elements: th.Tensor,
    nb_elements: int,
    n_random: int,
    generator: th.Generator,
    batch_size: int = 1,
) -> th.Tensor:
    """Sample target elements :math:`y` for :math:`\\exists r.B \\sqsubseteq C` combining filler elements and random elements.

    Guarantees:
    - Returns a 2-D ``torch.long`` tensor of shape ``(batch_size, F + n_random)``
      containing all ``filler_elements`` in the leading ``F`` columns followed
      by ``n_random`` uniform random elements per row.

    Non-guarantees:
    - Does not deduplicate element indices within a row.

    Parameters
    ----------
    filler_elements : torch.Tensor
        1-D ``(F,)`` or 2-D ``(batch_size, F)`` tensor of filler element indices.
    nb_elements : int
        Total number of domain elements :math:`|\\Delta| \\ge 1`.
    n_random : int
        Number of uniform random elements :math:`\\ge 0` per row.
    generator : torch.Generator
        Explicit PyTorch RNG.
    batch_size : int, default 1
        Number of rows when ``filler_elements`` is 1-D.

    Returns
    -------
    torch.Tensor
        2-D tensor of target element indices of shape ``(batch_size, F + n_random)``.
    """
    _validate_generator(generator)
    if not isinstance(filler_elements, th.Tensor):
        raise TypeError(
            f"filler_elements must be a torch.Tensor, got {type(filler_elements).__name__}"
        )
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError(f"batch_size must be an int >= 1, got {batch_size!r}")

    fe = filler_elements.to(dtype=th.long)
    if fe.ndim == 1:
        fe = fe.unsqueeze(0).expand(batch_size, fe.shape[0])
    elif fe.ndim == 2:
        batch_size = int(fe.shape[0])
    else:
        raise ValueError(
            f"filler_elements must be 1-D or 2-D, got shape {tuple(filler_elements.shape)}"
        )

    if n_random == 0:
        return fe.clone()
    rand_y = _randint_with_generator(
        0, nb_elements, (batch_size, n_random), device=fe.device, generator=generator
    )
    return th.cat([fe, rand_y], dim=1)


def adversarial_ascent(
    points: th.Tensor,
    lower: th.Tensor,
    upper: th.Tensor,
    hinge_fn: Callable[[th.Tensor], th.Tensor],
    n_adv: int = 1,
    step_size: Optional[Union[float, th.Tensor]] = None,
    valid: Optional[th.Tensor] = None,
) -> th.Tensor:
    """Run ``n_adv`` steps of signed-gradient ascent on ``hinge_fn`` clipped to ``[lower, upper]``.

    Guarantees:
    - Returns a detached tensor of the same shape, dtype, and device as ``points``.
    - Every output coordinate satisfies ``lower <= out <= upper`` on valid rows.
    - When ``n_adv == 0``, returns ``points.detach()`` unchanged.

    Non-guarantees:
    - Does not retain autograd history across the ascent steps; callers must
      re-evaluate the hinge at the returned points to backpropagate into model
      parameters.

    Parameters
    ----------
    points : torch.Tensor
        Initial point coordinates of shape ``(B, d)``.
    lower : torch.Tensor
        Lower box bounds of shape ``(B, d)``.
    upper : torch.Tensor
        Upper box bounds of shape ``(B, d)``.
    hinge_fn : Callable[[torch.Tensor], torch.Tensor]
        Function mapping ``(B, d)`` points to per-row hinge values.
    n_adv : int, default 1
        Number of signed-gradient ascent steps :math:`\\ge 0`.
    step_size : Optional[Union[float, torch.Tensor]], default None
        Per-step ascent size; defaults to ``0.25 * (upper - lower)`` when ``None``.
    valid : Optional[torch.Tensor], default None
        Optional boolean mask of shape ``(B,)``; invalid rows are left unchanged.

    Returns
    -------
    torch.Tensor
        Adversarially refined points of shape ``(B, d)`` clipped to ``[lower, upper]``.
    """
    if isinstance(n_adv, bool) or not isinstance(n_adv, int) or n_adv < 0:
        raise ValueError(f"n_adv must be an int >= 0, got {n_adv!r}")
    if points.shape != lower.shape or points.shape != upper.shape:
        raise ValueError(
            f"points, lower, and upper must have identical shapes, got "
            f"{tuple(points.shape)}, {tuple(lower.shape)}, {tuple(upper.shape)}"
        )
    if n_adv == 0 or points.numel() == 0:
        return points.detach()

    lo = lower.detach()
    hi = upper.detach()
    if step_size is None:
        step = 0.25 * (hi - lo)
    elif isinstance(step_size, th.Tensor):
        step = step_size.detach().to(device=points.device, dtype=points.dtype)
    else:
        if isinstance(step_size, bool) or not isinstance(step_size, (int, float)) or float(step_size) <= 0.0:
            raise ValueError(f"step_size must be > 0, got {step_size!r}")
        step = float(step_size)

    z = points.detach().clone()
    orig = z.clone()
    for _ in range(n_adv):
        with th.enable_grad():
            z_var = z.detach().requires_grad_(True)
            val = hinge_fn(z_var)
            grad = th.autograd.grad(val.sum(), z_var, create_graph=False)[0]
        z_next = z + step * grad.sign()
        z = th.minimum(th.maximum(z_next, lo), hi).detach()

    if valid is not None:
        v_mask = valid.to(device=z.device, dtype=th.bool).unsqueeze(-1)
        z = th.where(v_mask, z, orig)
    return z


def sample_active_component(
    regions: Regions,
    generator: th.Generator,
    n_adv: int = 0,
    hinge_fn: Optional[Callable[[th.Tensor], th.Tensor]] = None,
    step_size: Optional[Union[float, th.Tensor]] = None,
) -> RegionSample:
    """Sample one point per row uniformly inside a randomly chosen active non-empty component of ``regions``.

    Optionally performs ``n_adv`` signed-gradient ascent steps on ``hinge_fn``
    clipped to the chosen component box.

    Guarantees:
    - Returns a :class:`RegionSample` with ``points``, ``lower``, ``upper`` of
      shape ``(B, d)`` and ``valid`` of shape ``(B,)``.
    - For every row ``b`` with ``valid[b] == True``, ``points[b]`` lies inside
      ``[lower[b], upper[b]]`` of an active non-empty slot of ``regions[b]``
      (strictly inside ``(lower[b], upper[b])`` when ``n_adv == 0``).
    - Rows with no active non-empty slot have ``valid[b] == False`` and zero
      ``points[b]``, ``lower[b]``, ``upper[b]``.

    Non-guarantees:
    - Requires ``hinge_fn`` to be provided when ``n_adv > 0``.

    Parameters
    ----------
    regions : Regions
        Batch of regions of shape ``(B, K, d)``.
    generator : torch.Generator
        Explicit PyTorch RNG.
    n_adv : int, default 0
        Number of signed-gradient ascent steps :math:`\\ge 0`.
    hinge_fn : Optional[Callable[[torch.Tensor], torch.Tensor]], default None
        Hinge function used when ``n_adv > 0``.
    step_size : Optional[Union[float, torch.Tensor]], default None
        Optional step size for adversarial ascent.

    Returns
    -------
    RegionSample
        Sampled points, validity mask, and selected component box bounds.
    """
    _validate_generator(generator)
    if isinstance(n_adv, bool) or not isinstance(n_adv, int) or n_adv < 0:
        raise ValueError(f"n_adv must be an int >= 0, got {n_adv!r}")
    if n_adv > 0 and hinge_fn is None:
        raise ValueError("hinge_fn must be provided when n_adv > 0")

    cen = regions.cen
    w = regions.w
    active = regions.active
    if cen.ndim != 3 or w.shape != cen.shape or active.shape != cen.shape[:2]:
        raise ValueError(
            f"Invalid Regions shapes: cen={tuple(cen.shape)}, w={tuple(w.shape)}, "
            f"active={tuple(active.shape)}"
        )

    bsz, k_slots, dim = cen.shape
    device = cen.device
    dtype = cen.dtype

    if bsz == 0:
        empty_pts = th.zeros((0, dim), dtype=dtype, device=device)
        empty_val = th.zeros((0,), dtype=th.bool, device=device)
        return RegionSample(
            points=empty_pts, valid=empty_val, lower=empty_pts, upper=empty_pts
        )

    valid_k = active & (th.amin(w, dim=-1) > 0.0)  # (B, K)
    valid = valid_k.any(dim=1)  # (B,)

    u_slot = _rand_with_generator((bsz, k_slots), dtype=dtype, device=device, generator=generator)
    slot_scores = th.where(valid_k, u_slot, th.full_like(u_slot, -1.0))
    chosen_k = th.argmax(slot_scores, dim=1)  # (B,)
    gather_idx = chosen_k.view(bsz, 1, 1).expand(bsz, 1, dim)

    c_sel = th.gather(cen, 1, gather_idx).squeeze(1)  # (B, d)
    w_sel = th.gather(w, 1, gather_idx).squeeze(1)  # (B, d)

    valid_d = valid.unsqueeze(-1)
    c_sel = th.where(valid_d, c_sel, th.zeros_like(c_sel))
    w_sel = th.where(valid_d, w_sel, th.zeros_like(w_sel))

    lower = c_sel - w_sel
    upper = c_sel + w_sel

    u_pt = _rand_with_generator((bsz, dim), dtype=dtype, device=device, generator=generator)
    u_pt = u_pt.clamp(min=_INTERIOR_EPS, max=1.0 - _INTERIOR_EPS)
    points = th.where(valid_d, lower + (upper - lower) * u_pt, th.zeros_like(lower))

    if n_adv > 0 and hinge_fn is not None and bool(valid.any().item()):
        points = adversarial_ascent(
            points,
            lower,
            upper,
            hinge_fn=hinge_fn,
            n_adv=n_adv,
            step_size=step_size,
            valid=valid,
        )

    return RegionSample(points=points, valid=valid, lower=lower, upper=upper)


def sample_component_intersection(
    regions_a: Regions,
    regions_b: Regions,
    generator: th.Generator,
    n_adv: int = 0,
    hinge_fn: Optional[Callable[[th.Tensor], th.Tensor]] = None,
    step_size: Optional[Union[float, th.Tensor]] = None,
) -> RegionSample:
    """Sample one point per row uniformly inside a random non-empty active component intersection :math:`A_i \\cap B_j`.

    Implements §4 conjunction region sampling: forms all :math:`K_a \\times K_b`
    pairwise intersection boxes, selects a uniformly random active pair with
    strictly positive intersection width in every coordinate, and marks rows
    without any non-empty intersection as ``valid=False``.

    Guarantees:
    - Returns a :class:`RegionSample` with ``points``, ``lower``, ``upper`` of
      shape ``(B, d)`` and ``valid`` of shape ``(B,)``.
    - For every row ``b`` with ``valid[b] == True``, ``points[b]`` lies inside
      both an active component of ``regions_a[b]`` and an active component of
      ``regions_b[b]``.
    - Rows where all active component pairs are disjoint (or touch only on a
      boundary) have ``valid[b] == False``.

    Non-guarantees:
    - Allocates an intermediate ``(B, K_a * K_b, d)`` tensor of pairwise bounds.

    Parameters
    ----------
    regions_a : Regions
        First conjunct regions of shape ``(B, K_a, d)``.
    regions_b : Regions
        Second conjunct regions of shape ``(B, K_b, d)``.
    generator : torch.Generator
        Explicit PyTorch RNG.
    n_adv : int, default 0
        Number of signed-gradient ascent steps :math:`\\ge 0`.
    hinge_fn : Optional[Callable[[torch.Tensor], torch.Tensor]], default None
        Hinge function used when ``n_adv > 0``.
    step_size : Optional[Union[float, torch.Tensor]], default None
        Optional step size for adversarial ascent.

    Returns
    -------
    RegionSample
        Sampled intersection points, validity mask, and intersection box bounds.
    """
    _validate_generator(generator)
    if (
        regions_a.cen.ndim != 3
        or regions_b.cen.ndim != 3
        or regions_a.cen.shape[0] != regions_b.cen.shape[0]
        or regions_a.cen.shape[2] != regions_b.cen.shape[2]
    ):
        raise ValueError(
            f"Incompatible region shapes {tuple(regions_a.cen.shape)} and {tuple(regions_b.cen.shape)}"
        )

    bsz, ka, dim = regions_a.cen.shape
    kb = regions_b.cen.shape[1]
    total_pairs = ka * kb

    ca = regions_a.cen.unsqueeze(2)  # (B, ka, 1, d)
    wa = regions_a.w.unsqueeze(2)
    cb = regions_b.cen.unsqueeze(1)  # (B, 1, kb, d)
    wb = regions_b.w.unsqueeze(1)

    lo_ij = th.maximum(ca - wa, cb - wb)
    hi_ij = th.minimum(ca + wa, cb + wb)
    cen_ij = 0.5 * (lo_ij + hi_ij)
    w_ij = 0.5 * (hi_ij - lo_ij)
    act_ij = regions_a.active.unsqueeze(2) & regions_b.active.unsqueeze(1)

    inter_regions = Regions(
        cen=cen_ij.reshape(bsz, total_pairs, dim),
        w=w_ij.reshape(bsz, total_pairs, dim),
        active=act_ij.reshape(bsz, total_pairs),
    )
    return sample_active_component(
        inter_regions,
        generator=generator,
        n_adv=n_adv,
        hinge_fn=hinge_fn,
        step_size=step_size,
    )


def sample_class_region(
    module: GroundedELModule,
    class_ids: Union[int, Sequence[int], th.Tensor],
    generator: th.Generator,
    n_samples: int = 1,
    n_adv: int = 0,
    hinge_fn: Optional[Callable[[th.Tensor], th.Tensor]] = None,
    step_size: Optional[Union[float, th.Tensor]] = None,
) -> RegionSample:
    """Sample region points from active components of ``class_ids`` in ``module``.

    When ``class_ids`` is a scalar (or length-1) and ``n_samples > 1``, draws
    ``n_samples`` independent points from that class.

    Guarantees:
    - Returns a :class:`RegionSample` of batch size ``B`` (or ``n_samples`` when
      ``class_ids`` is scalar).

    Non-guarantees:
    - Raises :class:`ValueError` if ``n_samples < 1``.

    Parameters
    ----------
    module : GroundedELModule
        GroundedEL neural module.
    class_ids : Union[int, Sequence[int], torch.Tensor]
        Class index or 1-D tensor of class indices.
    generator : torch.Generator
        Explicit PyTorch RNG.
    n_samples : int, default 1
        Number of samples when ``class_ids`` has length 1.
    n_adv : int, default 0
        Number of signed-gradient ascent steps :math:`\\ge 0`.
    hinge_fn : Optional[Callable[[torch.Tensor], torch.Tensor]], default None
        Optional hinge function for adversarial ascent.
    step_size : Optional[Union[float, torch.Tensor]], default None
        Optional step size for adversarial ascent.

    Returns
    -------
    RegionSample
        Sampled points, validity mask, and box bounds.
    """
    if isinstance(n_samples, bool) or not isinstance(n_samples, int) or n_samples < 1:
        raise ValueError(f"n_samples must be an int >= 1, got {n_samples!r}")
    idx = module._to_index_tensor(class_ids, module.nb_classes, "class_ids")
    if idx.numel() == 1 and n_samples > 1:
        idx = idx.expand(n_samples)
    regions = module.class_regions(idx)
    return sample_active_component(
        regions,
        generator=generator,
        n_adv=n_adv,
        hinge_fn=hinge_fn,
        step_size=step_size,
    )


def sample_pair_region(
    module: GroundedELModule,
    class_a_ids: Union[int, Sequence[int], th.Tensor],
    class_b_ids: Union[int, Sequence[int], th.Tensor],
    generator: th.Generator,
    n_samples: int = 1,
    n_adv: int = 0,
    hinge_fn: Optional[Callable[[th.Tensor], th.Tensor]] = None,
    step_size: Optional[Union[float, th.Tensor]] = None,
) -> RegionSample:
    """Sample region points from active component intersections of ``class_a_ids`` and ``class_b_ids`` in ``module``.

    Guarantees:
    - Returns a :class:`RegionSample` of batch size ``B`` (or ``n_samples`` when
      both inputs are scalar).

    Non-guarantees:
    - Raises :class:`ValueError` if ``class_a_ids`` and ``class_b_ids`` have
      different lengths.

    Parameters
    ----------
    module : GroundedELModule
        GroundedEL neural module.
    class_a_ids : Union[int, Sequence[int], torch.Tensor]
        First conjunct class index or 1-D tensor.
    class_b_ids : Union[int, Sequence[int], torch.Tensor]
        Second conjunct class index or 1-D tensor.
    generator : torch.Generator
        Explicit PyTorch RNG.
    n_samples : int, default 1
        Number of samples when both inputs have length 1.
    n_adv : int, default 0
        Number of signed-gradient ascent steps :math:`\\ge 0`.
    hinge_fn : Optional[Callable[[torch.Tensor], torch.Tensor]], default None
        Optional hinge function for adversarial ascent.
    step_size : Optional[Union[float, torch.Tensor]], default None
        Optional step size for adversarial ascent.

    Returns
    -------
    RegionSample
        Sampled intersection points, validity mask, and intersection box bounds.
    """
    if isinstance(n_samples, bool) or not isinstance(n_samples, int) or n_samples < 1:
        raise ValueError(f"n_samples must be an int >= 1, got {n_samples!r}")
    idx_a = module._to_index_tensor(class_a_ids, module.nb_classes, "class_a_ids")
    idx_b = module._to_index_tensor(class_b_ids, module.nb_classes, "class_b_ids")
    if idx_a.shape != idx_b.shape:
        raise ValueError(
            f"class_a_ids and class_b_ids must have the same shape, got {tuple(idx_a.shape)} vs {tuple(idx_b.shape)}"
        )
    if idx_a.numel() == 1 and n_samples > 1:
        idx_a = idx_a.expand(n_samples)
        idx_b = idx_b.expand(n_samples)
    reg_a = module.class_regions(idx_a)
    reg_b = module.class_regions(idx_b)
    return sample_component_intersection(
        reg_a,
        reg_b,
        generator=generator,
        n_adv=n_adv,
        hinge_fn=hinge_fn,
        step_size=step_size,
    )


def sample_role_region(
    module: GroundedELModule,
    rel_ids: Union[int, Sequence[int], th.Tensor],
    generator: th.Generator,
    n_samples: int = 1,
) -> Tuple[RegionSample, RegionSample]:
    """Sample point pairs ``(x_sample, y_sample)`` uniformly inside a random active product component :math:`H_{r,k} \\times T_{r,k}` of ``rel_ids``.

    Guarantees:
    - Returns ``(head_sample, tail_sample)`` drawn from the *same* randomly
      chosen active product component :math:`k` per row so that
      ``(head_sample.points[b], tail_sample.points[b])`` lies inside
      :math:`H_{r,k} \\times T_{r,k}` whenever ``head_sample.valid[b]`` is ``True``.

    Non-guarantees:
    - Roles with no active non-empty product component have ``valid[b] == False``.

    Parameters
    ----------
    module : GroundedELModule
        GroundedEL neural module.
    rel_ids : Union[int, Sequence[int], torch.Tensor]
        Role index or 1-D tensor of role indices.
    generator : torch.Generator
        Explicit PyTorch RNG.
    n_samples : int, default 1
        Number of samples when ``rel_ids`` has length 1.

    Returns
    -------
    Tuple[RegionSample, RegionSample]
        ``(head_sample, tail_sample)`` for :math:`x` and :math:`y`.
    """
    _validate_generator(generator)
    if isinstance(n_samples, bool) or not isinstance(n_samples, int) or n_samples < 1:
        raise ValueError(f"n_samples must be an int >= 1, got {n_samples!r}")
    idx = module._to_index_tensor(rel_ids, module.nb_rels, "rel_ids")
    if idx.numel() == 1 and n_samples > 1:
        idx = idx.expand(n_samples)

    h_reg, t_reg = module.role_regions(idx)
    bsz, kr, dim = h_reg.cen.shape
    device = h_reg.cen.device
    dtype = h_reg.cen.dtype

    # Concatenate (H_{r,k}, T_{r,k}) as a 2d-dimensional box in R^{2d} so the same k is chosen
    prod_regions = Regions(
        cen=th.cat([h_reg.cen, t_reg.cen], dim=-1),
        w=th.cat([h_reg.w, t_reg.w], dim=-1),
        active=h_reg.active,
    )
    prod_sample = sample_active_component(prod_regions, generator=generator)
    h_sample = RegionSample(
        points=prod_sample.points[:, :dim],
        valid=prod_sample.valid,
        lower=prod_sample.lower[:, :dim],
        upper=prod_sample.upper[:, :dim],
    )
    t_sample = RegionSample(
        points=prod_sample.points[:, dim:],
        valid=prod_sample.valid,
        lower=prod_sample.lower[:, dim:],
        upper=prod_sample.upper[:, dim:],
    )
    return h_sample, t_sample
