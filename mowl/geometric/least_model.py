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
    """Finite least model of a normalised EL++ ontology over blocks of syntactic elements.

    Elements ``0 .. nb_classes - 1`` are the class elements ``x_A`` (element id =
    class id); elements ``nb_classes ..`` are individuals. ``element_map`` sends
    every element to its block (the identity unless a quotient was requested).

    Guarantees
    ----------
    - ``raw_labels`` / ``raw_edges`` / ``dead`` are the fixpoint of the grounded
      completion rules (see :func:`least_model`).
    - ``labels`` / ``edges`` are the pruned view: rows of dead blocks are empty,
      no edge touches a dead block, and column ``bot_id`` is empty. The pruned
      view is a model of the tables it was computed from.

    Non-guarantees
    --------------
    - A class whose own block is dead is unsatisfiable and has an empty column.
    - With a non-identity ``element_map`` the model need not be the least model
      of the ontology; it is the least model in which merged elements coincide.

    Attributes
    ----------
    labels : scipy.sparse.csr_matrix
        Boolean ``(n_blocks, nb_classes)`` membership matrix (pruned).
    edges : numpy.ndarray
        ``(m, 3)`` int64 rows ``(block_x, block_y, role)``, sorted, unique (pruned).
    dead : numpy.ndarray
        Boolean ``(n_blocks,)``; a dead block derives bottom.
    """

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
        """``True`` iff no block of an individual (or of top) is dead."""
        return not (
            self.dead[self.element_map[self.nb_classes:]].any()
            or (self.top_id is not None and self.dead[self.element_map[self.top_id]])
        )

    def unsatisfiable_classes(self):
        """Class ids (``bot_id`` excluded) whose own block is dead."""
        ids = np.flatnonzero(self.dead[self.element_map[: self.nb_classes]])
        if self.bot_id is not None:
            ids = ids[ids != self.bot_id]
        return ids.astype(np.int64)

    def holds(self, sub, sup):
        """``True`` iff column ``sub`` is contained in column ``sup`` (pruned model)."""
        a = self.labels[:, int(sub)]
        b = self.labels[:, int(sup)]
        return (a - a.multiply(b)).nnz == 0

    def subsumption_matrix(self, class_ids=None, empty_subsumed_by_all=True, chunk=4096):
        """Sparse matrix of the subsumptions that hold in the pruned model.

        Parameters
        ----------
        class_ids : array-like of int, optional
            Classes to compare (default: all classes).
        empty_subsumed_by_all : bool, default True
            If ``True`` an empty column is contained in every column (one dense
            row per empty column). If ``False`` empty columns get no entries.
        chunk : int, default 4096
            Number of source classes per sparse product.

        Returns
        -------
        scipy.sparse.csr_matrix
            Boolean ``(len(ids), len(ids))``; entry ``(i, j)`` iff column
            ``ids[i]`` is a subset of column ``ids[j]`` (reflexive).

        Guarantees
        ----------
        Exact (not sampled); never materialises a dense ``n x n`` array except
        the rows of empty columns when ``empty_subsumed_by_all`` is ``True``.
        """
        ids = (
            np.arange(self.nb_classes, dtype=np.int64)
            if class_ids is None
            else np.asarray(class_ids, dtype=np.int64)
        )
        if ids.ndim != 1 or np.any(ids < 0) or np.any(ids >= self.nb_classes):
            raise ValueError(
                "class_ids must be a one-dimensional array of valid class ids"
            )
        n = len(ids)
        cols = self.labels[:, ids].astype(np.int32).tocsc()
        sizes = np.asarray(cols.getnnz(axis=0)).ravel()
        cols_r = cols.tocsr()
        out_r, out_c = [], []
        for start in range(0, n, int(chunk)):
            stop = min(start + int(chunk), n)
            m = (cols[:, start:stop].T @ cols_r).tocoo()
            keep = m.data == sizes[start + m.row]
            out_r.append(start + m.row[keep])
            out_c.append(m.col[keep])
        empty = np.flatnonzero(sizes == 0)
        if empty_subsumed_by_all and len(empty):
            out_r.append(np.repeat(empty, n))
            out_c.append(np.tile(np.arange(n), len(empty)))
        rows = np.concatenate(out_r) if out_r else np.empty(0, dtype=np.int64)
        cs = np.concatenate(out_c) if out_c else np.empty(0, dtype=np.int64)
        res = sparse.coo_matrix(
            (np.ones(len(rows), dtype=bool), (rows, cs)), shape=(n, n)
        ).tocsr()
        res.data[:] = True
        return res

    def instances(self, class_id):
        """Individual ids whose block belongs to ``class_id`` in the pruned model."""
        member = np.zeros(self.n_blocks, dtype=bool)
        member[self.labels[:, int(class_id)].nonzero()[0]] = True
        return np.flatnonzero(member[self.element_map[self.nb_classes:]]).astype(
            np.int64
        )

    def dense_labels(self):
        """Boolean ``(nb_elements, nb_classes)`` array: each element gets its block's row."""
        return self.labels[self.element_map].toarray().astype(bool, copy=False)

    def element_edges(self):
        """Edges as element ids; requires an identity ``element_map``."""
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
    col_keep = np.ones(nc, dtype=bool)
    if bot_id is not None:
        col_keep[bot_id] = False
    rawL = (rawL @ sparse.diags(col_keep.astype(np.int8))).astype(bool).tocsr()
    rawL.eliminate_zeros()
    pruned = (sparse.diags((~dead).astype(np.int8)) @ rawL).astype(bool).tocsr()
    pruned.eliminate_zeros()
    edges = (
        raw_edges[~(dead[raw_edges[:, 0]] | dead[raw_edges[:, 1]])]
        if len(raw_edges)
        else raw_edges
    )
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
    parts = []
    for r, m in enumerate(roles):
        coo = m.tocoo()
        if coo.nnz:
            parts.append(
                np.stack(
                    [coo.row.astype(np.int64), coo.col.astype(np.int64),
                     np.full(coo.nnz, r, dtype=np.int64)],
                    axis=1,
                )
            )
    if not parts:
        return np.empty((0, 3), dtype=np.int64)
    return np.unique(np.concatenate(parts), axis=0)


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


def _split_bot(t, bot_id):
    for key, botkey, hcol in (
        ("gci0", "gci0_bot", 1),
        ("gci1", "gci1_bot", 2),
        ("gci3", "gci3_bot", 2),
    ):
        if bot_id is not None and t[key].size:
            mask = t[key][:, hcol] == bot_id
            t[botkey] = np.vstack((t[botkey], t[key][mask]))
            t[key] = t[key][~mask]
    return t


def _binary(m):
    m = m.tocsr().astype(np.int32)
    m.eliminate_zeros()
    m.data[:] = 1
    return m


def _selector(ids, nc):
    m = len(ids)
    return sparse.csr_matrix(
        (np.ones(m, dtype=np.int32), (np.asarray(ids), np.arange(m))), shape=(nc, m)
    )


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
    """Count violated ground axiom instances in the finite model, independently of the solver.

    Table conventions are those of :func:`least_model`. Rows of ``gci0`` / ``gci1`` /
    ``gci3`` whose head is ``lm.bot_id`` are treated as bottom rules.

    Parameters
    ----------
    lm : LeastModel
        Model to check.
    pruned : bool, default True
        Check the pruned view (``labels`` / ``edges``) or the raw fixpoint.

    Returns
    -------
    Dict[str, int]
        Per table, the number of violated ground instances: ``gci0`` /
        ``gci1`` / ``gci2`` / ``gci3`` and the bottom tables count pairs
        (rule row, block) whose body holds and whose head fails (for
        ``gci3`` the body is "has an r-successor in B"); ``role_inclusion``
        and ``role_chain`` count pairs (rule row, (x, y)); assertions count
        rows. ``non_degenerate_violations`` counts live class elements that
        are not members of their own class.

    Guarantees
    ----------
    Evaluates set semantics directly with sparse products; shares no rule
    code with :func:`least_model`. All zeros iff the view is a model of the
    tables (and non-degenerate at the class elements).
    """
    vals = dict(
        gci0=gci0, gci1=gci1, gci2=gci2, gci3=gci3, gci0_bot=gci0_bot,
        gci1_bot=gci1_bot, gci3_bot=gci3_bot, role_inclusion=role_inclusion,
        role_chain=role_chain, class_assertion=class_assertion,
        object_property_assertion=object_property_assertion,
    )
    t = _split_bot({k: _table(vals[k], w, k) for k, w in _TABLE_WIDTHS.items()}, lm.bot_id)
    nc, nb, nr = lm.nb_classes, lm.n_blocks, lm.nb_rels
    L = _binary(lm.labels if pruned else lm.raw_labels)
    edges = lm.edges if pruned else lm.raw_edges
    R = []
    for r in range(nr):
        e = edges[edges[:, 2] == r] if len(edges) else edges
        R.append(
            sparse.csr_matrix(
                (np.ones(len(e), dtype=np.int32), (e[:, 0], e[:, 1])), shape=(nb, nb)
            )
        )
    v = {k: 0 for k in _TABLE_WIDTHS}

    def missing(body, head):
        return int(body.nnz - body.multiply(head).nnz)

    def conj(a, b):
        return _binary((L @ _selector(a, nc)).multiply(L @ _selector(b, nc)))

    g = t["gci0"]
    if g.size:
        v["gci0"] = missing(_binary(L @ _selector(g[:, 0], nc)), L @ _selector(g[:, 1], nc))
    g = t["gci1"]
    if g.size:
        v["gci1"] = missing(conj(g[:, 0], g[:, 1]), L @ _selector(g[:, 2], nc))
    g = t["gci0_bot"]
    if g.size:
        v["gci0_bot"] = int(_binary(L @ _selector(g[:, 0], nc)).nnz)
    g = t["gci1_bot"]
    if g.size:
        v["gci1_bot"] = int(conj(g[:, 0], g[:, 1]).nnz)
    need = set(t["gci2"][:, 1].tolist()) | set(t["gci3"][:, 0].tolist()) | set(
        t["gci3_bot"][:, 0].tolist()
    )
    for r in sorted(need):
        W = _binary(R[r] @ L)  # W[x, B] iff x has an r-successor in B
        g = t["gci2"][t["gci2"][:, 1] == r]
        if g.size:
            v["gci2"] += missing(_binary(L @ _selector(g[:, 0], nc)), W @ _selector(g[:, 2], nc))
        g = t["gci3"][t["gci3"][:, 0] == r]
        if g.size:
            v["gci3"] += missing(_binary(W @ _selector(g[:, 1], nc)), L @ _selector(g[:, 2], nc))
        g = t["gci3_bot"][t["gci3_bot"][:, 0] == r]
        if g.size:
            v["gci3_bot"] += int(_binary(W @ _selector(g[:, 1], nc)).nnz)
    for r, s in t["role_inclusion"]:
        v["role_inclusion"] += missing(R[int(r)], R[int(s)])
    for r1, r2, s in t["role_chain"]:
        v["role_chain"] += missing(_binary(R[int(r1)] @ R[int(r2)]), R[int(s)])
    em = lm.element_map
    g = t["class_assertion"]
    if g.size:
        v["class_assertion"] = int(len(g) - np.asarray(L[em[nc + g[:, 0]], g[:, 1]]).astype(bool).sum())
    g = t["object_property_assertion"]
    for r in np.unique(g[:, 1]) if g.size else []:
        rows = g[g[:, 1] == r]
        got = np.asarray(R[int(r)][em[nc + rows[:, 0]], em[nc + rows[:, 2]]]).astype(bool).sum()
        v["object_property_assertion"] += int(len(rows) - got)
    cls = np.arange(nc)
    if lm.bot_id is not None:
        cls = cls[cls != lm.bot_id]
    own = np.asarray(L[em[cls], cls]).ravel().astype(bool)
    v["non_degenerate_violations"] = int((~lm.dead[em[cls]] & ~own).sum())
    return v


def _check_model_reference(lm, pruned=True, **tables):
    """Set-based reference for :func:`check_model` (small inputs, tests only)."""
    t = _split_bot(
        {k: _table(tables.get(k), w, k) for k, w in _TABLE_WIDTHS.items()}, lm.bot_id
    )
    L = lm.labels if pruned else lm.raw_labels
    labels = [set(L[:, c].nonzero()[0].tolist()) for c in range(lm.nb_classes)]
    edges = lm.edges if pruned else lm.raw_edges
    es = set(map(tuple, edges.tolist()))
    v = {k: 0 for k in _TABLE_WIDTHS}

    def has(r, b):
        return {x for x, y, rr in es if rr == r and y in labels[b]}

    for a, b in t["gci0"]:
        v["gci0"] += len(labels[a] - labels[b])
    for a, b, c in t["gci1"]:
        v["gci1"] += len((labels[a] & labels[b]) - labels[c])
    for a, r, b in t["gci2"]:
        v["gci2"] += len(labels[a] - has(r, b))
    for r, b, c in t["gci3"]:
        v["gci3"] += len(has(r, b) - labels[c])
    for a, _ in t["gci0_bot"]:
        v["gci0_bot"] += len(labels[a])
    for a, b, _ in t["gci1_bot"]:
        v["gci1_bot"] += len(labels[a] & labels[b])
    for r, b, _ in t["gci3_bot"]:
        v["gci3_bot"] += len(has(r, b))
    for r, s in t["role_inclusion"]:
        v["role_inclusion"] += sum((x, y, s) not in es for x, y, rr in es if rr == r)
    for r1, r2, s in t["role_chain"]:
        pairs = {(x, y) for x, m, ra in es if ra == r1 for m2, y, rb in es if rb == r2 and m2 == m}
        v["role_chain"] += sum((x, y, s) not in es for x, y in pairs)
    em, nc = lm.element_map, lm.nb_classes
    for a, c in t["class_assertion"]:
        v["class_assertion"] += int(em[nc + a] not in labels[c])
    for a, r, b in t["object_property_assertion"]:
        v["object_property_assertion"] += int((em[nc + a], em[nc + b], r) not in es)
    v["non_degenerate_violations"] = sum(
        not lm.dead[em[c]] and c != lm.bot_id and em[c] not in labels[c] for c in range(nc)
    )
    return v


def _dfs_order(lm):
    """Depth-first order of live blocks along the most-specific-superclass forest."""
    nc = lm.nb_classes
    em = lm.element_map
    live = ~lm.dead
    L = lm.labels.tocsr()
    sizes = np.asarray(L.getnnz(axis=0)).ravel()
    parent = np.full(nc, -1, dtype=np.int64)
    is_node = np.zeros(nc, dtype=bool)
    for a in range(nc):
        b = em[a]
        if a == lm.bot_id or not live[b]:
            continue
        is_node[a] = True
        if a == lm.top_id:
            continue
        cand = L.indices[L.indptr[b]:L.indptr[b + 1]]
        cand = cand[(cand != a) & (cand != (-1 if lm.top_id is None else lm.top_id))]
        if len(cand) == 0:
            continue
        cand = cand[live[em[cand]] & (em[cand] != b)]
        # only classes that come after ``a`` in the (size, id) order: acyclic
        cand = cand[(sizes[cand] > sizes[a]) | ((sizes[cand] == sizes[a]) & (cand > a))]
        if len(cand):
            parent[a] = cand[np.lexsort((cand, sizes[cand]))[0]]
    nodes = np.flatnonzero(is_node)
    children = [[] for _ in range(nc)]
    for a in nodes:
        if parent[a] >= 0:
            children[parent[a]].append(int(a))
    subtree = np.ones(nc, dtype=np.int64)
    for a in nodes[np.lexsort((nodes, sizes[nodes]))]:  # children before parents
        if parent[a] >= 0:
            subtree[parent[a]] += subtree[a]
    seen = np.zeros(lm.n_blocks, dtype=bool)
    order = []
    roots = [int(a) for a in nodes if parent[a] < 0]
    stack = roots[::-1]
    while stack:
        a = stack.pop()
        b = int(em[a])
        if not seen[b]:
            seen[b] = True
            order.append(b)
        ch = sorted(children[a], key=lambda c: (-subtree[c], c))
        stack.extend(ch[::-1])
    rest = np.flatnonzero(live & ~seen)
    return np.concatenate([np.asarray(order, dtype=np.int64), rest.astype(np.int64)])


def interval_realisation(lm, order="dfs"):
    """Realise the pruned model in one dimension: every class is a union of intervals.

    Live blocks are placed at the integer positions ``0 .. n_live - 1`` given by
    ``order``; a class is the union of the maximal runs of consecutive positions
    of its members; a role is covered by products (run of sources) x {target}.

    Parameters
    ----------
    lm : LeastModel
    order : {"dfs", "identity"} or array-like of int
        ``"dfs"``: forest on the classes whose own block is live, where the
        parent of ``A`` is the class ``B`` in the label row of ``x_A`` (``B`` not
        ``A``, not top, own block live and different) that is smallest in the
        (column size, id) order among those after ``A`` in that order; roots in
        id order, children by decreasing subtree size then id, iterative
        pre-order; live blocks not reached (e.g. individuals) follow in id
        order. ``"identity"``: live blocks in id order. An array must be a
        permutation of the live block ids.

    Returns
    -------
    IntervalRealisation

    Guarantees
    ----------
    The total length of the intervals of each class equals the size of its
    column, so membership read from the intervals equals ``lm.labels``
    restricted to live blocks. A tree-shaped hierarchy (every class has at most
    one direct superclass, no other axioms) gets one interval per class under
    ``"dfs"``.

    Non-guarantees
    --------------
    The number of runs is not minimal; minimising it is NP-hard in general.
    """
    live_ids = np.flatnonzero(~lm.dead)
    if isinstance(order, str):
        if order == "identity":
            ordering = live_ids.astype(np.int64)
        elif order == "dfs":
            ordering = _dfs_order(lm)
        else:
            raise ValueError("order must be 'dfs', 'identity', or a permutation")
    else:
        ordering = np.asarray(order, dtype=np.int64)
        if ordering.ndim != 1 or not np.array_equal(np.sort(ordering), live_ids):
            raise ValueError("explicit order must permute live block ids")
    nc = lm.nb_classes
    rank = np.full(lm.n_blocks, -1, dtype=np.int64)
    rank[ordering] = np.arange(len(ordering))
    coo = lm.labels.tocoo()
    pos = rank[coo.row]
    keep = pos >= 0
    cls, pos = coo.col[keep].astype(np.int64), pos[keep]
    srt = np.lexsort((pos, cls))
    cls, pos = cls[srt], pos[srt]
    if len(cls):
        start = np.r_[True, (cls[1:] != cls[:-1]) | (pos[1:] != pos[:-1] + 1)]
        end = np.r_[start[1:], True]
        run_cls, run_lo, run_hi = cls[start], pos[start], pos[end]
    else:
        run_cls = run_lo = run_hi = np.empty(0, dtype=np.int64)
    runs = np.bincount(run_cls, minlength=nc).astype(np.int64)
    covered = np.bincount(run_cls, weights=run_hi - run_lo + 1, minlength=nc)
    if not np.array_equal(covered.astype(np.int64), np.bincount(cls, minlength=nc)):
        raise RuntimeError("interval membership reconstruction failed")
    bounds = np.stack([run_lo, run_hi], axis=1) if len(run_cls) else np.empty((0, 2), dtype=np.int64)
    cuts = np.cumsum(runs)[:-1]
    intervals = {c: part for c, part in enumerate(np.split(bounds, cuts))}
    products = 0
    if len(lm.edges):
        src = rank[lm.edges[:, 0]]
        tgt, rol = lm.edges[:, 1], lm.edges[:, 2]
        srt = np.lexsort((src, tgt, rol))
        src, tgt, rol = src[srt], tgt[srt], rol[srt]
        new = np.r_[True, (rol[1:] != rol[:-1]) | (tgt[1:] != tgt[:-1]) | (src[1:] != src[:-1] + 1)]
        products = int(new.sum())
    return IntervalRealisation(ordering, runs, intervals, products)


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
        weights = np.asarray(objective_weights(np.random.default_rng(0)), dtype=float)
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
