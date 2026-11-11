# flake8: noqa: W503, E203
"""Sparse least-model saturation over mOWL's indexed EL++ tables.

Guarantees
----------
The iteration method computes the least Boolean fixed point of the supplied
completion rules.  The LP method exposes their literal finite Horn relaxation.
The checker and interval construction operate on the resulting finite model.

Non-guarantees
--------------
The LP method is intended only for small finite domains.  A valid finite
interpretation does not imply that every class is satisfiable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from scipy import sparse


@dataclass
class LeastModel:
    nb_classes: int
    nb_rels: int
    nb_individuals: int
    element_map: np.ndarray
    n_blocks: int
    labels: sparse.csr_matrix
    edges: np.ndarray
    dead: np.ndarray
    raw_labels: sparse.csr_matrix
    raw_edges: np.ndarray
    n_sweeps: int
    method: str
    top_id: Optional[int] = None
    bot_id: Optional[int] = None

    @property
    def consistent(self):
        return not (
            self.dead[self.element_map[self.nb_classes:]].any()
            or (self.top_id is not None and self.dead[self.element_map[self.top_id]])
        )

    def unsatisfiable_classes(self):
        ids = np.flatnonzero(self.dead[self.element_map[: self.nb_classes]])
        if self.bot_id is not None:
            ids = ids[ids != self.bot_id]
        return ids.astype(np.int64)

    def holds(self, sub, sup):
        a = self.labels[:, int(sub)]
        b = self.labels[:, int(sup)]
        return (a - a.multiply(b)).nnz == 0

    def subsumption_matrix(self, class_ids=None):
        ids = (
            np.arange(self.nb_classes, dtype=np.int64)
            if class_ids is None
            else np.asarray(class_ids, dtype=np.int64)
        )
        if ids.ndim != 1 or np.any(ids < 0) or np.any(ids >= self.nb_classes):
            raise ValueError(
                "class_ids must be a one-dimensional array of valid class ids"
            )
        # For each nonempty source column, its supersets are found by sparse
        # intersections; chunking bounds temporary storage for large signatures.
        out_r, out_c = [], []
        cols = self.labels[:, ids].tocsc()
        row_frequency = np.asarray(cols.getnnz(axis=1)).ravel()
        for start in range(0, len(ids), 256):
            for j in range(start, min(start + 256, len(ids))):
                src = cols[:, j]
                if src.nnz == 0:
                    out_r.extend([j] * len(ids))
                    out_c.extend(range(len(ids)))
                    continue
                # Every superset contains the rarest element in the source.
                anchor = int(src.indices[np.argmin(row_frequency[src.indices])])
                candidates = cols.getrow(anchor).indices
                candidate_cols = cols[:, candidates].astype(np.int32)
                counts = (src.T.astype(np.int32) @ candidate_cols).toarray().ravel()
                valid = candidates[counts == src.nnz]
                out_r.extend([j] * len(valid))
                out_c.extend(valid.tolist())
        return sparse.csr_matrix(
            (np.ones(len(out_r), dtype=bool), (out_r, out_c)),
            shape=(len(ids), len(ids)),
        )

    def instances(self, class_id):
        blocks = set(self.labels[:, int(class_id)].nonzero()[0].tolist())
        return np.asarray(
            [
                i
                for i in range(self.nb_individuals)
                if int(self.element_map[self.nb_classes + i]) in blocks
            ],
            dtype=np.int64,
        )

    def dense_labels(self):
        return self.labels[self.element_map].toarray().astype(bool, copy=False)

    def element_edges(self):
        if not np.array_equal(self.element_map, np.arange(len(self.element_map))):
            raise ValueError("element_edges requires an identity element_map")
        return self.edges.copy()


@dataclass
class IntervalRealisation:
    order: np.ndarray
    runs_per_class: np.ndarray
    intervals: Dict[int, np.ndarray]
    role_products: int

    def n_parameters(self):
        return 2 * int(self.runs_per_class.sum()) + 3 * int(self.role_products)


_TABLE_WIDTHS = {
    "gci0": 2,
    "gci1": 3,
    "gci2": 3,
    "gci3": 3,
    "gci0_bot": 2,
    "gci1_bot": 3,
    "gci3_bot": 3,
    "role_inclusion": 2,
    "role_chain": 3,
    "class_assertion": 2,
    "object_property_assertion": 3,
}


def _table(value, width, name):
    if value is None:
        return np.empty((0, width), dtype=np.int64)
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    raw = np.asarray(value)
    if raw.size and raw.dtype.kind not in "iu":
        raise ValueError(f"{name} must contain integer indices")
    a = np.asarray(raw, dtype=np.int64)
    if a.size == 0:
        return np.empty((0, width), dtype=np.int64)
    if a.ndim != 2 or a.shape[1] != width:
        raise ValueError(f"{name} must have shape (m, {width})")
    return a


def _as_sparse(rows, cols, shape):
    if len(rows) == 0:
        return sparse.csr_matrix(shape, dtype=bool)
    x = sparse.coo_matrix(
        (np.ones(len(rows), dtype=bool), (rows, cols)), shape=shape
    ).tocsr()
    x.data[:] = True
    return x


def least_model(
    nb_classes,
    nb_rels=0,
    nb_individuals=0,
    *,
    gci0=None,
    gci1=None,
    gci2=None,
    gci3=None,
    gci0_bot=None,
    gci1_bot=None,
    gci3_bot=None,
    role_inclusion=None,
    role_chain=None,
    class_assertion=None,
    object_property_assertion=None,
    top_id=None,
    bot_id=None,
    inverse_of=None,
    element_map=None,
    extra_facts=None,
    method="iteration",
    max_sweeps=None,
    objective_weights=None,
    lp_max_variables=2_000_000,
):
    """Compute the finite least model by sparse completion or literal LP."""
    dims = (nb_classes, nb_rels, nb_individuals)
    if (
        any(
            isinstance(x, (bool, np.bool_))
            or not isinstance(x, (int, np.integer))
            or x < 0
            for x in dims
        )
        or nb_classes < 1
    ):
        raise ValueError(
            "nb_classes must be positive and nb_rels/nb_individuals nonnegative integers"
        )
    nc, nr, ni = map(int, dims)
    ne = nc + ni
    if (
        isinstance(lp_max_variables, bool)
        or not isinstance(lp_max_variables, (int, np.integer))
        or lp_max_variables < 1
    ):
        raise ValueError("lp_max_variables must be a positive integer")
    if max_sweeps is not None and (
        isinstance(max_sweeps, bool)
        or not isinstance(max_sweeps, (int, np.integer))
        or max_sweeps < 1
    ):
        raise ValueError("max_sweeps must be a positive integer or None")
    # Normalize optional tables to a consistent integer-array representation.
    vals = dict(
        gci0=gci0,
        gci1=gci1,
        gci2=gci2,
        gci3=gci3,
        gci0_bot=gci0_bot,
        gci1_bot=gci1_bot,
        gci3_bot=gci3_bot,
        role_inclusion=role_inclusion,
        role_chain=role_chain,
        class_assertion=class_assertion,
        object_property_assertion=object_property_assertion,
    )
    tables = {k: _table(vals[k], w, k) for k, w in _TABLE_WIDTHS.items()}
    for name, arr in tables.items():
        limits = {
            "gci0": (nc, nc),
            "gci1": (nc, nc, nc),
            "gci2": (nc, nr, nc),
            "gci3": (nr, nc, nc),
            "gci0_bot": (nc, nc),
            "gci1_bot": (nc, nc, nc),
            "gci3_bot": (nr, nc, nc),
            "role_inclusion": (nr, nr),
            "role_chain": (nr, nr, nr),
            "class_assertion": (ni, nc),
            "object_property_assertion": (ni, nr, ni),
        }[name]
        if (
            limits
            and arr.size
            and (
                np.any(arr < 0)
                or any(np.any(arr[:, i] >= lim) for i, lim in enumerate(limits))
            )
        ):
            raise ValueError(f"{name} contains an out-of-range index")
    if top_id is not None and top_id == bot_id:
        raise ValueError("top_id and bot_id must be distinct")
    for name, val in (("top_id", top_id), ("bot_id", bot_id)):
        if val is not None and (
            isinstance(val, bool)
            or not isinstance(val, (int, np.integer))
            or not 0 <= val < nc
        ):
            raise ValueError(f"{name} must be a valid class id or None")
    if element_map is None:
        em = np.arange(ne, dtype=np.int64)
    else:
        raw_element_map = np.asarray(element_map)
        if raw_element_map.dtype.kind not in "iu":
            raise ValueError("element_map must contain integer block ids")
        em = np.asarray(raw_element_map, dtype=np.int64)
        if em.shape != (ne,) or np.any(em < 0):
            raise ValueError(
                "element_map must have one nonnegative block id per element"
            )
        em = np.unique(em, return_inverse=True)[1].astype(np.int64)
    nb = int(em.max() + 1) if len(em) else 0
    if inverse_of is None:
        inv = None
    else:
        raw_inverse = np.asarray(inverse_of)
        if raw_inverse.dtype.kind not in "iu":
            raise ValueError("inverse_of must contain integer role ids")
        inv = np.asarray(raw_inverse, dtype=np.int64)
    if inv is not None and (inv.shape != (nr,) or np.any((inv < -1) | (inv >= nr))):
        raise ValueError(
            "inverse_of must have length nb_rels and contain -1 or role ids"
        )
    if inv is not None and any(
        partner >= 0 and inv[int(partner)] != role for role, partner in enumerate(inv)
    ):
        raise ValueError("inverse_of partners must be symmetric")
    # Split folded bottom heads.
    for key, botkey, hcol in (
        ("gci0", "gci0_bot", 1),
        ("gci1", "gci1_bot", 2),
        ("gci3", "gci3_bot", 2),
    ):
        if bot_id is not None and tables[key].size:
            mask = tables[key][:, hcol] == bot_id
            extra = tables[key][mask]
            if extra.size:
                tables[botkey] = np.vstack((tables[botkey], extra))
                tables[key] = tables[key][~mask]
    # GCI3 bot table uses (r,B,bot); bottom-rule tables use normalized third column too.
    g0, g1, g2, g3 = (tables[k] for k in ("gci0", "gci1", "gci2", "gci3"))
    g0b, g1b, g3b = (tables[k] for k in ("gci0_bot", "gci1_bot", "gci3_bot"))
    initial_dead = {int(em[bot_id])} if bot_id is not None else set()
    class_ids = np.arange(nc, dtype=np.int64)
    if bot_id is not None:
        class_ids = class_ids[class_ids != bot_id]
    label_rows = [em[class_ids]]
    label_cols = [class_ids]
    if top_id is not None:
        label_rows.append(np.arange(nb, dtype=np.int64))
        label_cols.append(np.full(nb, top_id, dtype=np.int64))
    ca = tables["class_assertion"]
    if ca.size:
        if bot_id is not None:
            initial_dead.update(em[nc + ca[ca[:, 1] == bot_id, 0]].tolist())
            ca = ca[ca[:, 1] != bot_id]
        if ca.size:
            label_rows.append(em[nc + ca[:, 0]])
            label_cols.append(ca[:, 1])
    if extra_facts is not None:
        ef = _table(extra_facts, 2, "extra_facts")
        if ef.size and (
            np.any(ef[:, 0] >= nb) or np.any(ef[:, 1] >= nc) or np.any(ef < 0)
        ):
            raise ValueError("extra_facts index out of range")
        if ef.size:
            if bot_id is not None:
                initial_dead.update(ef[ef[:, 1] == bot_id, 0].tolist())
                ef = ef[ef[:, 1] != bot_id]
            if ef.size:
                label_rows.append(ef[:, 0])
                label_cols.append(ef[:, 1])
    L = _as_sparse(np.concatenate(label_rows), np.concatenate(label_cols), (nb, nc))
    opa = tables["object_property_assertion"]
    # Sparse selector matrices make conjunctions a vectorized pair of products.
    if g1.size:
        g1_rows = np.arange(len(g1), dtype=np.int64)
        g1_left = _as_sparse(g1[:, 0], g1_rows, (nc, len(g1)))
        g1_right = _as_sparse(g1[:, 1], g1_rows, (nc, len(g1)))
        g1_heads = _as_sparse(g1_rows, g1[:, 2], (len(g1), nc))
    else:
        g1_left = sparse.csr_matrix((nc, 0), dtype=bool)
        g1_right = sparse.csr_matrix((nc, 0), dtype=bool)
        g1_heads = sparse.csr_matrix((0, nc), dtype=bool)
    # gci2 source class -> role/filler block adjacency, precomputed as sparse class maps.
    g2_by_role = []
    for r in range(nr):
        rr = g2[g2[:, 1] == r] if g2.size else g2
        g2_by_role.append(_as_sparse(rr[:, 0], rr[:, 2], (nc, nc)))
    # Relation matrices per role are maintained sparsely as (source,target) matrices.
    roles = [sparse.csr_matrix((nb, nb), dtype=bool) for _ in range(nr)]
    for r in range(nr):
        assertions = opa[opa[:, 1] == r] if opa.size else opa
        if assertions.size:
            sources = em[nc + assertions[:, 0]]
            targets = em[nc + assertions[:, 2]]
            roles[r] = _as_sparse(sources, targets, (nb, nb))
    # Utility monotonically unions sparse boolean matrices.
    dead = np.zeros(nb, dtype=bool)
    if initial_dead:
        dead[list(initial_dead)] = True

    def union(a, b):
        if b.nnz == 0:
            return a, False
        z = (a + b).tocsr()
        z.data[:] = True
        z.eliminate_zeros()
        return z, z.nnz != a.nnz

    def do_iteration():
        nonlocal L, dead
        old_dead_count = int(dead.sum())
        changed = False
        if g0.size:
            S = _as_sparse(g0[:, 0], g0[:, 1], (nc, nc))
            L, ch = union(L, L @ S)
            changed |= ch
        if g1.size:
            left_present = L @ g1_left
            right_present = L @ g1_right
            pair_present = left_present.multiply(right_present)
            conclusions = (pair_present @ g1_heads).tocsr()
            L, ch = union(L, conclusions)
            changed |= ch
        for r in range(nr):
            if g2_by_role[r].nnz:
                targets = (L @ g2_by_role[r]).tocoo()
                if targets.nnz:
                    dst = em[targets.col]
                    add = _as_sparse(targets.row, dst, (nb, nb))
                    roles[r], ch = union(roles[r], add)
                    changed |= ch
            if inv is not None and inv[r] >= 0:
                rev = roles[r].T.tocsr()
                roles[int(inv[r])], ch = union(roles[int(inv[r])], rev)
                changed |= ch
        # Role inclusions, chains and inverse closure, grouped by roles.
        inclusions = tables["role_inclusion"]
        for r in range(nr):
            targets = (
                np.unique(inclusions[inclusions[:, 0] == r, 1])
                if inclusions.size
                else []
            )
            for s in targets:
                roles[int(s)], ch = union(roles[int(s)], roles[r])
                changed |= ch
        chains = tables["role_chain"]
        if chains.size:
            pairs = np.unique(chains[:, :2], axis=0)
            for r1, r2 in pairs:
                prod = (roles[int(r1)] @ roles[int(r2)]).tocsr()
                for s in np.unique(
                    chains[(chains[:, 0] == r1) & (chains[:, 1] == r2), 2]
                ):
                    roles[int(s)], ch = union(roles[int(s)], prod)
                    changed |= ch
        if g3.size:
            for r in range(nr):
                rules = g3[g3[:, 0] == r]
                if rules.size:
                    S = _as_sparse(rules[:, 1], rules[:, 2], (nc, nc))
                    conclusions = (roles[r] @ L @ S).tocsr()
                    L, ch = union(L, conclusions)
                    changed |= ch
        # Bottom rules and backwards propagation across existential edges.
        if g0b.size:
            As = np.unique(g0b[:, 0])
            dead |= L[:, As].getnnz(axis=1) > 0
        if g1b.size:
            for a, b, _ in g1b:
                x = L[:, int(a)].multiply(L[:, int(b)]).nonzero()[0]
                dead[x] = True
        if g3b.size:
            for r in range(nr):
                rules = g3b[g3b[:, 0] == r]
                if rules.size:
                    Bs = np.unique(rules[:, 1])
                    dead |= np.asarray((roles[r] @ L[:, Bs]).getnnz(axis=1) > 0).ravel()
        for r in range(nr):
            if roles[r].nnz:
                dead |= np.asarray(roles[r] @ dead.astype(np.int8)).ravel() > 0
        return changed or int(dead.sum()) != old_dead_count

    if method == "iteration":
        sweeps = 0
        while True:
            if max_sweeps is not None and sweeps >= max_sweeps:
                raise RuntimeError("max_sweeps exceeded before reaching fixpoint")
            sweeps += 1
            changed = do_iteration()
            if not changed:
                break
    elif method == "lp":
        return _least_model_lp(
            nc,
            nr,
            ni,
            em,
            nb,
            tables,
            top_id,
            bot_id,
            inv,
            extra_facts,
            objective_weights,
            lp_max_variables,
        )
    else:
        raise ValueError("method must be 'iteration' or 'lp'")
    rawL = L.tocsr()
    raw_edges = _role_edges(roles)
    pruned = rawL.copy().tolil()
    pruned[np.flatnonzero(dead), :] = False
    pruned = pruned.tocsr()
    edges = (
        raw_edges[~(dead[raw_edges[:, 0]] | dead[raw_edges[:, 1]])]
        if len(raw_edges)
        else raw_edges
    )
    if bot_id is not None:
        pruned[:, bot_id] = False
        rawL[:, bot_id] = False
    return LeastModel(
        nc,
        nr,
        ni,
        em,
        nb,
        pruned,
        edges,
        dead,
        rawL,
        raw_edges,
        sweeps,
        "iteration",
        top_id,
        bot_id,
    )


def _role_edges(roles):
    rows = []
    for r, m in enumerate(roles):
        coo = m.tocoo()
        rows.extend((int(x), int(y), r) for x, y in zip(coo.row, coo.col))
    return np.asarray(sorted(set(rows)), dtype=np.int64).reshape((-1, 3))


def least_model_from_model(model, **kwargs):
    """Build a least model from GroundedEL or another indexed EL model."""
    tables = getattr(model, "training_datasets", {})
    if callable(tables):
        tables = tables()
    aliases = {
        "nb_individuals": ("nb_individuals", "nb_inds"),
        "nb_classes": ("nb_classes",),
        "nb_rels": ("nb_rels",),
    }
    dims = {
        k: next((getattr(model, n) for n in names if hasattr(model, n)), None)
        for k, names in aliases.items()
    }
    if dims["nb_individuals"] is None:
        dims["nb_individuals"] = 0
    if any(v is None for v in dims.values()):
        raise ValueError(
            "model must expose nb_classes, nb_rels and nb_individuals/nb_inds"
        )
    for key in _TABLE_WIDTHS:
        if key not in kwargs and key in tables:
            val = tables[key]
            kwargs[key] = getattr(val, "data", val)
    for key in ("top_id", "bot_id", "inverse_of"):
        if key not in kwargs and hasattr(model, key):
            kwargs[key] = getattr(model, key)
    return least_model(**dims, **kwargs)


def check_model(
    lm,
    *,
    gci0=None,
    gci1=None,
    gci2=None,
    gci3=None,
    gci0_bot=None,
    gci1_bot=None,
    gci3_bot=None,
    role_inclusion=None,
    role_chain=None,
    class_assertion=None,
    object_property_assertion=None,
    pruned=True,
):
    """Count violated ground instances by direct finite set evaluation."""
    names = _TABLE_WIDTHS
    vals = locals()
    t = {k: _table(vals[k], w, k) for k, w in names.items()}
    for key, botkey, hcol in (
        ("gci0", "gci0_bot", 1),
        ("gci1", "gci1_bot", 2),
        ("gci3", "gci3_bot", 2),
    ):
        if lm.bot_id is not None and t[key].size:
            mask = t[key][:, hcol] == lm.bot_id
            t[botkey] = np.vstack((t[botkey], t[key][mask]))
            t[key] = t[key][~mask]
    L = lm.labels if pruned else lm.raw_labels
    labels = [set(L[:, c].nonzero()[0]) for c in range(lm.nb_classes)]
    edges = lm.edges if pruned else lm.raw_edges
    es = set(map(tuple, edges.tolist()))
    violations = {k: 0 for k in names}
    outgoing = {}
    for x, y, r in es:
        outgoing.setdefault((r, x), set()).add(y)
    witnessed = {}
    for x, y, r in es:
        for c in L.getrow(y).indices:
            witnessed.setdefault((r, int(c)), set()).add(x)
    for a, b in t["gci0"]:
        violations["gci0"] += len(labels[int(a)] - labels[int(b)])
    for a, b, c in t["gci1"]:
        violations["gci1"] += len((labels[int(a)] & labels[int(b)]) - labels[int(c)])
    for a, r, b in t["gci2"]:
        violations["gci2"] += len(
            labels[int(a)] - witnessed.get((int(r), int(b)), set())
        )
    for r, b, c in t["gci3"]:
        for x, y, rr in es:
            if rr == r and y in labels[int(b)] and x not in labels[int(c)]:
                violations["gci3"] += 1
    for a, _ in t["gci0_bot"]:
        violations["gci0_bot"] += len(labels[int(a)])
    for a, b, _ in t["gci1_bot"]:
        violations["gci1_bot"] += len(labels[int(a)] & labels[int(b)])
    for r, b, _ in t["gci3_bot"]:
        violations["gci3_bot"] += sum(
            1 for x, y, rr in es if rr == r and y in labels[int(b)]
        )
    for r, s in t["role_inclusion"]:
        violations["role_inclusion"] += sum(
            (x, y, int(r)) in es and (x, y, int(s)) not in es for x, y, _ in es
        )
    for r1, r2, s in t["role_chain"]:
        for x, m, r in es:
            if r == r1:
                violations["role_chain"] += sum(
                    (x, y, int(s)) not in es for y in outgoing.get((int(r2), m), ())
                )
    for a, c in t["class_assertion"]:
        block = lm.element_map[lm.nb_classes + int(a)]
        violations["class_assertion"] += int(block not in labels[int(c)])
    for a, r, b in t["object_property_assertion"]:
        x = lm.element_map[lm.nb_classes + int(a)]
        y = lm.element_map[lm.nb_classes + int(b)]
        violations["object_property_assertion"] += int((x, y, int(r)) not in es)
    # Non-degenerate: every non-dead class element except bot has itself in its class.
    violations["non_degenerate_violations"] = sum(
        not lm.dead[lm.element_map[c]]
        and c != lm.bot_id
        and lm.element_map[c] not in labels[c]
        for c in range(lm.nb_classes)
    )
    return violations


def interval_realisation(lm, order="dfs"):
    """Represent the pruned finite model as unions of intervals on a line."""
    live = np.flatnonzero(~lm.dead)
    if isinstance(order, str) and order == "identity":
        ordering = live
    elif isinstance(order, str) and order == "dfs":
        # Build the immediate containment forest from live class columns.
        sets = [
            set(lm.labels[:, c].nonzero()[0]) & set(live.tolist())
            for c in range(lm.nb_classes)
        ]
        class_parent = {c: None for c in range(lm.nb_classes)}
        for c in range(lm.nb_classes):
            supers = [d for d in range(lm.nb_classes) if sets[c] < sets[d]]
            if supers:
                class_parent[c] = min(supers, key=lambda d: (len(sets[d]), d))
        class_children = {c: [] for c in range(lm.nb_classes)}
        for c, p in class_parent.items():
            if p is not None:
                class_children[p].append(c)
        for children_c in class_children.values():
            children_c.sort(key=lambda c: (-len(sets[c]), c))
        ordering = []
        seen = set()

        def visit_class(c):
            b = int(lm.element_map[c])
            if b in live and b not in seen:
                seen.add(b)
                ordering.append(b)
            for child in class_children[c]:
                visit_class(child)

        for c in range(lm.nb_classes):
            if class_parent[c] is None:
                visit_class(c)
        # Individuals, quotient blocks, and class elements without a distinct
        # named-superclass path are placed deterministically after the forest.
        for b in live:
            if int(b) not in seen:
                seen.add(int(b))
                ordering.append(int(b))
        ordering = np.asarray(ordering, dtype=np.int64)
    elif not isinstance(order, str):
        ordering = np.asarray(order, dtype=np.int64)
        if (
            ordering.ndim != 1
            or len(ordering) != len(live)
            or set(ordering.tolist()) != set(live.tolist())
        ):
            raise ValueError("explicit order must permute live block ids")
    else:
        raise ValueError("order must be 'dfs', 'identity', or a permutation")
    rank = np.full(lm.n_blocks, -1, dtype=np.int64)
    rank[ordering] = np.arange(len(ordering))
    runs = np.zeros(lm.nb_classes, dtype=np.int64)
    intervals = {}
    for c in range(lm.nb_classes):
        pos = np.sort(rank[lm.labels[:, c].nonzero()[0]])
        pos = pos[pos >= 0]
        if len(pos):
            cuts = np.r_[0, np.flatnonzero(np.diff(pos) > 1) + 1, len(pos)]
            ints = np.asarray(
                [[pos[cuts[i]], pos[cuts[i + 1] - 1]] for i in range(len(cuts) - 1)],
                dtype=np.int64,
            )
        else:
            ints = np.empty((0, 2), dtype=np.int64)
        runs[c] = len(ints)
        intervals[c] = ints
    # For each role and target, source positions are covered in maximal runs.
    products = 0
    for r in range(lm.nb_rels):
        re = lm.edges[lm.edges[:, 2] == r] if len(lm.edges) else np.empty((0, 3), int)
        for target in np.unique(re[:, 1]) if len(re) else []:
            p = np.sort(rank[re[re[:, 1] == target, 0]])
            p = p[p >= 0]
            if len(p):
                products += 1 + int(np.sum(np.diff(p) > 1))
    result = IntervalRealisation(ordering, runs, intervals, products)
    # Exact membership guarantee.
    recovered = np.zeros((lm.n_blocks, lm.nb_classes), dtype=bool)
    for c, ints in intervals.items():
        for a, b in ints:
            recovered[a:b + 1, c] = True
    if not np.array_equal(recovered, lm.labels[ordering].toarray()):
        raise RuntimeError("interval membership reconstruction failed")
    return result


def _least_model_lp(
    nc,
    nr,
    ni,
    em,
    nb,
    tables,
    top_id,
    bot_id,
    inv,
    extra_facts,
    objective_weights,
    limit,
):
    """Construct and solve the literal dense-indexed Horn LP for small domains."""
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix

    g0, g1, g2, g3 = (tables[k] for k in ("gci0", "gci1", "gci2", "gci3"))
    g0b, g1b, g3b = (tables[k] for k in ("gci0_bot", "gci1_bot", "gci3_bot"))
    # Dense indexing is intentional here: this backend is a small-instance oracle.
    nl = nb * nc
    nrvars = nb * nb * nr
    nd = nb
    nvar = nl + nrvars + nd
    if nvar > limit:
        raise ValueError(f"LP has {nvar} variables, exceeding lp_max_variables={limit}")

    def si(b, c):
        return int(b) * nc + int(c)

    def ri(x, y, r):
        return nl + (int(x) * nb + int(y)) * nr + int(r)

    def di(b):
        return nl + nrvars + int(b)

    constraints = []
    rhs = []

    def add(atoms, head):
        # sum(body) - head <= body_size - 1; a missing head denotes false.
        row = {}
        for i in atoms:
            row[int(i)] = row.get(int(i), 0) + 1.0
        if head is not None:
            row[int(head)] = row.get(int(head), 0) - 1.0
        constraints.append(row)
        rhs.append(len(atoms) - 1 if atoms else -1)

    # Normalize folded bottom rows defensively.
    bounds = [(0.0, 1.0)] * nvar
    if bot_id is not None:
        for b in range(nb):
            bounds[si(b, bot_id)] = (0.0, 0.0)
    for c in range(nc):
        if c != bot_id:
            bounds[si(em[c], c)] = (1.0, 1.0)
    if top_id is not None:
        for b in range(nb):
            bounds[si(b, top_id)] = (1.0, 1.0)
    initial_dead = {int(em[bot_id])} if bot_id is not None else set()
    for a, c in tables["class_assertion"]:
        b = int(em[nc + int(a)])
        if c == bot_id:
            initial_dead.add(b)
        else:
            bounds[si(b, int(c))] = (1.0, 1.0)
    if extra_facts is not None:
        ef = _table(extra_facts, 2, "extra_facts")
        for b, c in ef:
            if c == bot_id:
                initial_dead.add(int(b))
            else:
                bounds[si(b, int(c))] = (1.0, 1.0)
    for b in initial_dead:
        bounds[di(b)] = (1.0, 1.0)
    for a, b in g0:
        for x in range(nb):
            add([si(x, a)], si(x, b))
    for a, b, c in g1:
        for x in range(nb):
            add([si(x, a), si(x, b)], si(x, c))
    for a, r, b in g2:
        for x in range(nb):
            add([si(x, a)], ri(x, em[b], r))
    for r, b, c in g3:
        for x in range(nb):
            for y in range(nb):
                add([ri(x, y, r), si(y, b)], si(x, c))
    for a, _ in g0b:
        for x in range(nb):
            add([si(x, a)], di(x))
    for a, b, _ in g1b:
        for x in range(nb):
            add([si(x, a), si(x, b)], di(x))
    for r, b, _ in g3b:
        for x in range(nb):
            for y in range(nb):
                add([ri(x, y, r), si(y, b)], di(x))
    for r, s in tables["role_inclusion"]:
        for x in range(nb):
            for y in range(nb):
                add([ri(x, y, r)], ri(x, y, s))
    for r1, r2, s in tables["role_chain"]:
        for x in range(nb):
            for m in range(nb):
                for y in range(nb):
                    add([ri(x, m, r1), ri(m, y, r2)], ri(x, y, s))
    if inv is not None:
        for r, s in enumerate(inv):
            if s >= 0:
                for x in range(nb):
                    for y in range(nb):
                        add([ri(x, y, r)], ri(y, x, s))
    for r in range(nr):
        for x in range(nb):
            for y in range(nb):
                add([ri(x, y, r), di(y)], di(x))
    for a, r, b in tables["object_property_assertion"]:
        bounds[ri(em[nc + int(a)], em[nc + int(b)], r)] = (1.0, 1.0)
    # LP weights are strictly positive; defaults to the all-ones objective.
    if objective_weights is None:
        weights = np.ones(nvar, dtype=float)
    elif callable(objective_weights):
        weights = np.asarray(objective_weights(np.random.default_rng()), dtype=float)
    else:
        weights = np.asarray(objective_weights, dtype=float)
    if (
        weights.shape != (nvar,)
        or not np.isfinite(weights).all()
        or np.any(weights <= 0)
    ):
        raise ValueError(
            "objective_weights must provide one finite, strictly positive weight per variable"
        )
    rr = []
    cc = []
    vv = []
    for i, row in enumerate(constraints):
        for j, v in row.items():
            rr.append(i)
            cc.append(j)
            vv.append(v)
    A = coo_matrix((vv, (rr, cc)), shape=(len(constraints), nvar)).tocsr()
    result = linprog(
        weights, A_ub=A, b_ub=np.asarray(rhs), bounds=bounds, method="highs"
    )
    if not result.success:
        raise RuntimeError(f"LP solver failed: {result.message}")
    x = result.x
    fraction = float(np.max(np.minimum(np.abs(x), np.abs(1 - x)))) if len(x) else 0.0
    rawL = sparse.csr_matrix((x[:nl] >= 0.5).reshape(nb, nc))
    if bot_id is not None:
        rawL[:, bot_id] = False
    raw_edges = np.asarray(
        [
            (b, y, r)
            for b in range(nb)
            for y in range(nb)
            for r in range(nr)
            if x[ri(b, y, r)] >= 0.5
        ],
        dtype=np.int64,
    ).reshape(-1, 3)
    dead = x[nl + nrvars:] >= 0.5
    labels = rawL.copy().tolil()
    labels[np.flatnonzero(dead), :] = False
    labels = labels.tocsr()
    edges = (
        raw_edges[~(dead[raw_edges[:, 0]] | dead[raw_edges[:, 1]])]
        if len(raw_edges)
        else raw_edges
    )
    lm = LeastModel(
        nc,
        nr,
        ni,
        em,
        nb,
        labels,
        edges,
        dead,
        rawL,
        raw_edges,
        1,
        "lp",
        top_id,
        bot_id,
    )
    lm.lp_max_fractionality = fraction
    lm.lp_objective = float(result.fun)
    return lm
