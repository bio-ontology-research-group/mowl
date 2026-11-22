"""Re-seeding of the components of a structural union of boxes (a training regime's move).

Gradient steps cannot move a component that covers nothing, and a conjunction needs its two
classes to overlap somewhere. The relocation rule below moves such components, under
``no_grad``, to where they are needed: it changes the layout of the tables of a
:class:`~mowl.representation.library.structural_regions.SlotClasses` between optimisation
steps and is about unions of boxes, not about an axiom, so it is a function of the geometry
(:class:`~mowl.representation.library.structural_methods.StructuralGeometry.reseed`) that the
regime :class:`~mowl.training.structure.IdleComponentReseeding` calls.
"""

import numpy as np
import torch as th

from mowl.nn.el.structural import nonempty_measure, pair_size, slot_components

__all__ = ["reseed_components"]


def _regions(geometry, idx):
    return geometry.classes(idx).slots


def _clamp_targets(targets, def_mask, off_view, margin):
    """The largest component of every primitive target class that is still empty is made at
    least ``margin`` wide. Returns the number of changes."""
    moves = 0
    if targets.classes.numel() > 0:
        for c_raw in targets.classes.detach().cpu().tolist():
            c_idx = int(c_raw)
            if def_mask[c_idx]:
                continue
            w_c = th.abs(off_view[c_idx])
            ne_c = nonempty_measure(w_c)
            if float(ne_c.max().item()) < margin:
                best_j = int(th.argmax(ne_c).item())
                off_view[c_idx, best_j].copy_(th.clamp(w_c[best_j], min=float(margin)))
                moves += 1
    return moves


def reseed_components(geometry, rng):
    """Relocate dead or idle primitive components of ``geometry`` (a
    :class:`~mowl.representation.library.structural_methods.StructuralGeometry` of slots of
    boxes) under ``no_grad``; returns the number of moves.

    ``geometry.reseed_options["mode"]``:

    ``"dead"``
        Dead components (nested in another component of the class, ties broken by the lower
        index, or thinner than :math:`10^{-3}` somewhere) are moved, in the order of a
        permutation drawn from ``rng``, to the partner of a conjunction pair that is
        violated (:math:`\\mathrm{NE}(C \\cap D) < m`): the largest active non-empty slot
        of the partner, with half its width; each dead component is used once. Then the
        largest component of every primitive target class that is still empty is clamped to
        width :math:`\\max(|o|, m)`.
    ``"idle"`` (needs ``witness_weight > 0`` and witness targets)
        Dead defined classes (:math:`\\mathrm{NE}(A) < m`) are processed in the order of a
        permutation, at most ``reseed_options["batch"]`` per call: for each primitive leaf
        :math:`X` of the definition, unless a component already contains the witness
        point :math:`p` of :math:`A` by at least :math:`m`, the component of :math:`X` that
        the fewest live defined classes use (:func:`~mowl.nn.el.structural.slot_components`)
        becomes a box of half-width ``reseed_options["size"]`` around :math:`p` plus noise
        drawn from ``rng``. Then the clamp step of ``"dead"``.

    Defined classes' rows are never modified. The optimiser moments of the moved rows are
    not reset.
    """
    options = geometry.reseed_options
    base = geometry._slot_classes
    targets = geometry.existence_targets
    margin = geometry.objective.margin
    n, k, dim = len(base), base.components, base.embed_dim
    table = geometry.definition_table
    if options["mode"] == "idle":
        if (geometry.witness_weight <= 0.0 or geometry.witness_embed is None
                or targets is None or targets.witness is None):
            raise ValueError("reseed_mode='idle' requires witness_weight > 0 and attached "
                             "witness targets.")
        if n == 0:
            geometry.last_reseed_count = 0
            return 0
        device, dtype = base.centres.weight.device, base.centres.weight.dtype
        with th.no_grad():
            cen_view = base.centres.weight.data.view(n, k, dim)
            off_view = base.offsets.weight.data.view(n, k, dim)
            def_mask = table.def_mask.detach().cpu().numpy() if table is not None \
                else np.zeros(n, dtype=bool)
            moves = 0
            if table is not None and table.n_defined > 0:
                wit_row = {int(c): i for i, c in enumerate(
                    targets.witness.classes.detach().cpu().tolist())}
                defined = [int(a) for a in np.flatnonzero(def_mask)]
                r_def = _regions(geometry, th.tensor(defined, device=device, dtype=th.long))
                ne = nonempty_measure(r_def.w).masked_fill(~r_def.active, float("-inf"))
                best_slots = th.argmax(ne, dim=1).detach().cpu().tolist()
                best_nes = th.amax(ne, dim=1).detach().cpu().tolist()
                usage = np.zeros((n, k), dtype=np.int64)
                dead = []
                for a_cls, best_s, best_ne in zip(defined, best_slots, best_nes):
                    if float(best_ne) >= margin:
                        for prim_x, comp_c in slot_components(table, int(a_cls), int(best_s),
                                                              k).items():
                            if 0 <= prim_x < n and 0 <= comp_c < k:
                                usage[prim_x, comp_c] += 1
                    elif int(a_cls) in wit_row:
                        dead.append(int(a_cls))
                if dead:
                    perm = rng.permutation(len(dead))
                    for p_idx in perm[:min(len(dead), int(options["batch"]))]:
                        a_cls = dead[int(p_idx)]
                        p = geometry.witness_embed.weight.data[wit_row[a_cls]]
                        for x_cls in list(slot_components(table, a_cls, 0, k).keys()):
                            w_x = th.abs(off_view[x_cls])
                            inside = th.amin(w_x - th.abs(p.unsqueeze(0) - cen_view[x_cls]),
                                             dim=-1)
                            if bool((inside >= margin).any().item()):
                                continue
                            c_star = int(np.argmin(usage[x_cls]))
                            noise = th.as_tensor(rng.standard_normal(dim), dtype=dtype,
                                                 device=device) * 0.01
                            cen_view[x_cls, c_star].copy_(p + noise)
                            off_view[x_cls, c_star].fill_(float(options["size"]))
                            usage[x_cls, c_star] += 1
                            moves += 1
            moves += _clamp_targets(targets, def_mask, off_view, margin)
            geometry.last_reseed_count = moves
            return moves

    if targets is None or n == 0:
        geometry.last_reseed_count = 0
        return 0
    eps = 1e-3
    device = base.centres.weight.device
    with th.no_grad():
        cen_view = base.centres.weight.data.view(n, k, dim)
        off_view = base.offsets.weight.data.view(n, k, dim)
        cen_np = cen_view.detach().cpu().numpy()
        off_np = th.abs(off_view.detach()).cpu().numpy()
        lo_np, hi_np = cen_np - off_np, cen_np + off_np
        valid_np = np.all(off_np >= eps, axis=-1)
        def_mask = table.def_mask.detach().cpu().numpy() if table is not None \
            else np.zeros(n, dtype=bool)
        dead_queues = {}
        for c in range(n):
            if def_mask[c]:
                dead_queues[c] = []
                continue
            valid, lo_c, hi_c = valid_np[c], lo_np[c], hi_np[c]
            dead_c = []
            for i in range(k):
                if not valid[i]:
                    dead_c.append(i)
                    continue
                contained = False
                for j in range(k):
                    if i == j or not valid[j]:
                        continue
                    if np.all(lo_c[j] <= lo_c[i]) and np.all(hi_c[i] <= hi_c[j]):
                        equal = np.all(lo_c[i] <= lo_c[j]) and np.all(hi_c[j] <= hi_c[i])
                        if not equal or j < i:
                            contained = True
                            break
                if contained:
                    dead_c.append(i)
            dead_queues[c] = dead_c
        moves = 0
        if targets.pairs.numel() > 0:
            pairs = targets.pairs.to(device=device, dtype=th.long)
            sizes = pair_size(_regions(geometry, pairs[:, 0]), _regions(geometry, pairs[:, 1]))
            violated = np.flatnonzero((sizes < margin).detach().cpu().numpy())
            if violated.size > 0:
                perm = rng.permutation(violated.size)
                pairs_list = targets.pairs.detach().cpu().tolist()
                for p_idx in violated[perm]:
                    c_idx, d_idx = int(pairs_list[p_idx][0]), int(pairs_list[p_idx][1])
                    if dead_queues.get(c_idx):
                        mover, partner = c_idx, d_idx
                    elif dead_queues.get(d_idx):
                        mover, partner = d_idx, c_idx
                    else:
                        continue
                    r_p = _regions(geometry, th.tensor([partner], device=device,
                                                       dtype=th.long))
                    ne_p = nonempty_measure(r_p.w[0]).masked_fill(~r_p.active[0],
                                                                  float("-inf"))
                    best_j = int(th.argmax(ne_p).item())
                    if float(ne_p[best_j].item()) <= 0.0:
                        continue
                    dead_slot = dead_queues[mover].pop(0)
                    cen_view[mover, dead_slot].copy_(r_p.cen[0, best_j])
                    off_view[mover, dead_slot].copy_(0.5 * th.abs(r_p.w[0, best_j]))
                    moves += 1
        moves += _clamp_targets(targets, def_mask, off_view, margin)
        geometry.last_reseed_count = moves
        return moves
