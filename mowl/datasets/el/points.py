"""Canonical-element point supervision (:class:`PointSupervision`).

Constructs positive membership pairs ``(x_D, C)`` from sparse ancestor sets,
hard negative pairs ``(x_D, C)`` from (a) the one-step disjointness closure
(:math:`D \\sqsubseteq A` and :math:`C \\sqsubseteq B` for some axiom
:math:`A \\sqcap B \\sqsubseteq \\bot` or :math:`A \\sqsubseteq \\bot`, represented as
sparse descendant-index product blocks :math:`\\text{desc}(A) \\times \\text{desc}(B)`
that are enumerated exhaustively below :data:`HARD_NEG_CAP` and sampled without
replacement proportional to :math:`|\\text{desc}(A)| \\cdot |\\text{desc}(B)|` when
above the cap, without ever materialising a full Cartesian product) plus
(b) an exact joint-satisfiability pass over candidate pairs that can trigger a
conjunction axiom (:math:`D \\sqsubseteq A` and :math:`C \\sqsubseteq B` in either
order for some :math:`A \\sqcap B \\sqsubseteq E` with :math:`E \\neq \\bot`,
represented with the same non-materialised descendant-index product blocks and
checked exactly; bounded by at most
``max_exact_checks = 4 * hard_neg_cap`` (:data:`MAX_EXACT_CHECKS` at default cap)
candidate checks in total and capped at :data:`HARD_NEG_CAP`, with
:attr:`PointSupervision.hard_neg_capped` recording whether either bound was hit),
random negative candidates by rejection sampling against each element's positive
set at :meth:`PointSupervision.sample` time, and canonical role edges
``(x_C, r, x_D)`` restricted to the element table from a
:class:`~mowl.geometric.completion.CanonicalModel`.

Why one exact check per candidate pair in (b) suffices without outer iteration:
In :class:`~mowl.geometric.completion.CanonicalModel`, every single-concept rule
(``NF1``, ``NF3``, ``NF4``, ``NF6``, ``NF7``, and backward ``BOTTOM`` propagation)
is already saturated to fixpoint on :math:`S(D)` and :math:`S(C)` individually,
and a fresh conjunction concept :math:`Z \\sqsubseteq D \\sqcap C` has no incoming
role edges. Therefore, :math:`S(D) \\cup S(C)` can derive concepts outside
:math:`S(D) \\cup S(C)` only if at least one binary conjunction axiom
(:math:`A \\sqcap B \\sqsubseteq E`) has one conjunct in :math:`S(D)` and the other
in :math:`S(C)`. Because joint-satisfiability checking saturates each candidate
pair :math:`\\{D, C\\}` completely to fixpoint across all chained axioms, a single
exact check per candidate pair is both sound and complete and requires no outer
fixpoint loop.

Guarantees:
- Element table contains one row per satisfiable named concept in
  ``class_index_dict`` (excluding ``TOP``, ``BOTTOM``, and normaliser auxiliary
  classes) followed by one row per satisfiable individual in
  ``individual_index_dict`` present in ``cm.individuals``.
- Any individual in ``individual_index_dict`` whose nominal concept is
  unsatisfiable in ``cm`` raises :class:`ValueError` (if an individual is
  unsatisfiable in the source ontology, :class:`~mowl.geometric.completion.CanonicalModel`
  itself raises :class:`~mowl.geometric.errors.InconsistentOntology`).
- Unsatisfiable concept names, ``TOP``, ``BOTTOM``, and auxiliary concept names
  are absent from :attr:`PointSupervision.element_names`,
  :attr:`PointSupervision.pos_pairs`, :attr:`PointSupervision.hard_neg_pairs`,
  and sampled random negatives.
- Every pair ``(x_D, C)`` in :attr:`PointSupervision.pos_pairs` is entailed by
  ``cm`` (:meth:`~mowl.geometric.completion.CanonicalModel.entails`), and no pair
  in :attr:`PointSupervision.hard_neg_pairs` or sampled ``"rand"`` is entailed.
- Every pair ``(x_D, C)`` in :attr:`PointSupervision.hard_neg_pairs` belongs to
  the disjointness closure and satisfies ``not cm.joint_satisfiable({D, C})``.
- Nothing per-element is stored on :class:`PointSupervision` or in the cache file;
  :meth:`PointSupervision.sample` returns at most ``n`` pairs in total across
  ``"pos"``, ``"hard"``, ``"rand"``, and ``"roles"``, drawing one hard negative
  per sampled positive when the positive's element has any and one random
  negative per sampled positive by rejection against ``pos_pairs``.
- :meth:`PointSupervision.save` and :meth:`PointSupervision.load` validate
  :data:`POINTS_CODE_TAG` and (when supplied) the ontology SHA-256 digest.

Non-guarantees:
- For file-backed datasets with an existing ``ontology_path``,
  :func:`ontology_sha256` hashes the ontology file on disk rather than iterating
  over in-memory ``OWLOntology`` axioms; in-memory edits to ``dataset.ontology``
  (e.g. ``add_axioms``) are therefore not detected by the cache key unless
  ``ontology_path`` is absent or ``use_cache=False``.
- Individuals present in ``individual_index_dict`` that do not occur in the
  training ontology (e.g. validation/test-only individuals absent from
  ``cm.individuals``) have no canonical element in ``cm`` and are omitted from
  the element table when ``allow_absent_individuals=True`` (the default for
  model training).
- When every candidate class in ``class_index_dict`` is entailed for some
  element ``x_D`` (e.g. a single-class ontology), no non-entailed class exists
  for ``x_D`` and ``sample()`` omits random negatives for that element rather
  than sampling a false negative.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import logging
import os
from pathlib import Path
import random
from typing import BinaryIO, Dict, FrozenSet, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.sparse as sp
import torch

from mowl.geometric.completion import CanonicalModel
from mowl.geometric.membership import _sparse_bool_coo
from mowl.geometric.normal_forms import AUX_NS, BOTTOM, NF1, NF2, TOP, is_nominal_name
from mowl.ontology.normalize import is_auxiliary_iri

logger = logging.getLogger(__name__)

POINTS_CODE_TAG: str = "points-v5"
HARD_NEG_CAP: int = 2_000_000
MAX_EXACT_CHECKS_FACTOR: int = 4
MAX_EXACT_CHECKS: int = MAX_EXACT_CHECKS_FACTOR * HARD_NEG_CAP


def _is_named_target_class(iri: str, cm: CanonicalModel) -> bool:
    """Return ``True`` iff ``iri`` is a satisfiable non-auxiliary named concept in ``cm``."""
    if iri in (TOP, BOTTOM):
        return False
    if is_nominal_name(iri) or is_auxiliary_iri(iri) or iri.startswith(AUX_NS):
        return False
    if iri not in cm.concept_names:
        return False
    return cm.is_satisfiable(iri)


def _randint_with_rng(
    high: int,
    size: int,
    rng: Optional[
        Union[
            int,
            torch.Generator,
            np.random.Generator,
            np.random.RandomState,
            random.Random,
        ]
    ],
) -> torch.Tensor:
    """Draw ``size`` uniform integers in ``0 .. high - 1`` using ``rng``."""
    if size <= 0 or high <= 0:
        return torch.empty((0,), dtype=torch.long)
    if rng is None:
        return torch.randint(0, high, (size,), dtype=torch.long)
    if isinstance(rng, int):
        gen = torch.Generator()
        gen.manual_seed(int(rng))
        return torch.randint(0, high, (size,), generator=gen, dtype=torch.long)
    if isinstance(rng, torch.Generator):
        return torch.randint(0, high, (size,), generator=rng, dtype=torch.long)
    if isinstance(rng, (np.random.Generator, np.random.RandomState)):
        arr = (
            rng.integers(0, high, size=size)
            if isinstance(rng, np.random.Generator)
            else rng.randint(0, high, size=size)
        )
        return torch.as_tensor(arr, dtype=torch.long)
    if isinstance(rng, random.Random):
        return torch.tensor([rng.randrange(high) for _ in range(size)], dtype=torch.long)
    raise TypeError(f"Unsupported rng type: {type(rng).__name__}")


def _sample_indices(
    total: int,
    cap: int,
    rng: Optional[
        Union[
            int,
            torch.Generator,
            np.random.Generator,
            np.random.RandomState,
            random.Random,
        ]
    ],
) -> torch.Tensor:
    """Permute ``0 .. total - 1`` with ``rng`` and return the first ``min(total, cap)`` indices."""
    if total <= 0 or cap <= 0:
        return torch.empty((0,), dtype=torch.long)
    k = min(int(total), int(cap))
    if rng is None:
        return torch.randperm(total, dtype=torch.long)[:k]
    if isinstance(rng, int):
        gen = torch.Generator()
        gen.manual_seed(int(rng))
        return torch.randperm(total, generator=gen, dtype=torch.long)[:k]
    if isinstance(rng, torch.Generator):
        return torch.randperm(total, generator=rng, dtype=torch.long)[:k]
    if isinstance(rng, np.random.Generator):
        chosen = rng.choice(total, size=k, replace=False)
        return torch.as_tensor(chosen, dtype=torch.long)
    if isinstance(rng, np.random.RandomState):
        chosen = rng.choice(total, size=k, replace=False)
        return torch.as_tensor(chosen, dtype=torch.long)
    if isinstance(rng, random.Random):
        chosen = rng.sample(range(total), k)
        return torch.tensor(chosen, dtype=torch.long)
    raise TypeError(f"Unsupported rng type: {type(rng).__name__}")


def _pair_joint_satisfiable(
    cm: CanonicalModel,
    d_name: str,
    c_iri: str,
    nf2_by_sub: Optional[Mapping[str, Sequence[NF2]]] = None,
    cache: Optional[Dict[FrozenSet[str], bool]] = None,
) -> bool:
    """Return ``True`` iff ``{d_name, c_iri}`` is jointly satisfiable in ``cm``.

    Starts from the pre-saturated ancestor sets ``S(d_name) | S(c_iri)`` in
    ``cm.completion.sub_super`` and closes under binary conjunction axioms
    (:class:`~mowl.geometric.normal_forms.NF2`). Because every single-concept
    rule (``NF1``, ``NF3``, ``NF4``, ``NF6``, ``NF7``, and backward ``BOTTOM``
    propagation) is already saturated on each concept name in ``cm.completion``
    and a fresh conjunction concept ``Z <= d_name & c_iri`` has no incoming role
    edges, ``S(Z)`` is identical to closing ``S(d_name) | S(c_iri)`` under
    ``NF2`` triggers ``(B1, B2) -> S(E)``.
    """
    key = frozenset((d_name, c_iri))
    if cache is not None and key in cache:
        return cache[key]
    if key in cm._joint_sat_cache:
        res = cm._joint_sat_cache[key]
        if cache is not None:
            cache[key] = res
        return res

    sub_super = cm.completion.sub_super
    s_d = sub_super.get(d_name, set())
    s_c = sub_super.get(c_iri, set())
    if d_name == BOTTOM or c_iri == BOTTOM or BOTTOM in s_d or BOTTOM in s_c:
        if cache is not None:
            cache[key] = False
        return False

    if nf2_by_sub is None:
        nf2_index: Dict[str, List[NF2]] = {}
        for ax in cm.axioms:
            if isinstance(ax, NF2):
                nf2_index.setdefault(ax.sub1, []).append(ax)
                if ax.sub2 != ax.sub1:
                    nf2_index.setdefault(ax.sub2, []).append(ax)
        nf2_by_sub = nf2_index

    closure: set[str] = set(s_d) | set(s_c)
    queue = list(s_c - s_d)
    head = 0
    while head < len(queue):
        x = queue[head]
        head += 1
        for ax in nf2_by_sub.get(x, ()):
            other = ax.sub2 if ax.sub1 == x else ax.sub1
            if other in closure:
                sup = ax.sup
                if cm.completion.is_unsatisfiable(sup):
                    if cache is not None:
                        cache[key] = False
                    return False
                if sup not in closure:
                    for u in sub_super.get(sup, (sup,)):
                        if u == BOTTOM:
                            if cache is not None:
                                cache[key] = False
                            return False
                        if u not in closure:
                            closure.add(u)
                            queue.append(u)

    if cache is not None:
        cache[key] = True
    return True


@dataclass
class _ProductBlocks:
    """Compact representation of rectangular product blocks ``desc_elem(A_k) x desc_cls(B_k)``.

    Never materialises any block product ``desc_elem(A_k) x desc_cls(B_k)``; stores
    only flat 1-D pools of element and target-class indices (`O(nnz(M))`) plus
    per-block ``(offset, length)`` descriptors and cumulative product sizes
    :math:`\\Sigma_k |\\text{desc}(A_k)| \\cdot |\\text{desc}(B_k)|`.
    """

    l_pool: np.ndarray
    r_pool: np.ndarray
    l_offsets: np.ndarray
    n_l: np.ndarray
    r_offsets: np.ndarray
    n_r: np.ndarray
    sizes: np.ndarray
    cum_sizes: np.ndarray
    total: int

    def decode_codes(self, p_arr: np.ndarray, stride: int) -> np.ndarray:
        """Decode flat product indices ``0 <= p < total`` into ``elem_idx * stride + class_idx``."""
        if p_arr.size == 0 or self.total == 0:
            return np.empty(0, dtype=np.int64)
        block_idx = np.searchsorted(self.cum_sizes, p_arr, side="right")
        prev_cum = np.where(
            block_idx > 0, self.cum_sizes[np.maximum(block_idx - 1, 0)], 0
        )
        rem = p_arr - prev_cum
        r_len = self.n_r[block_idx]
        i_idx = rem // r_len
        j_idx = rem % r_len
        e_arr = self.l_pool[self.l_offsets[block_idx] + i_idx]
        c_arr = self.r_pool[self.r_offsets[block_idx] + j_idx]
        return e_arr * stride + c_arr


def _filter_excluded_codes(
    codes: np.ndarray, exclude_sorted_list: Sequence[np.ndarray]
) -> np.ndarray:
    """Remove any code present in any sorted 1-D array in ``exclude_sorted_list``, preserving order."""
    if codes.size == 0:
        return codes
    for ex in exclude_sorted_list:
        if ex.size == 0:
            continue
        idx = np.searchsorted(ex, codes)
        idx_clamped = np.minimum(idx, ex.size - 1)
        keep = (idx >= ex.size) | (ex[idx_clamped] != codes)
        if not bool(np.all(keep)):
            codes = codes[keep]
            if codes.size == 0:
                return codes
    return codes


def _sample_or_enumerate_blocks(
    blocks: _ProductBlocks,
    stride: int,
    exclude_codes: Sequence[np.ndarray],
    cap: int,
    rng_seed: int,
) -> Tuple[np.ndarray, bool]:
    """Enumerate all valid pair codes when ``blocks.total <= cap``, else sample up to ``cap`` without replacement.

    Drawing a uniform index ``p`` in ``0 .. blocks.total - 1`` and decoding via
    ``blocks.cum_sizes`` is mathematically identical to sampling an axiom block
    proportional to its product size ``|desc(A_k)| * |desc(B_k)|`` and then a
    uniform ``(i, j)`` index pair in ``desc(A_k) x desc(B_k)``.
    """
    if blocks.total == 0:
        return np.empty(0, dtype=np.int64), False

    if cap <= 0 or blocks.total <= cap:
        p_arr = np.arange(blocks.total, dtype=np.int64)
        codes = blocks.decode_codes(p_arr, stride)
        codes = _filter_excluded_codes(codes, exclude_codes)
        if codes.size == 0:
            return np.empty(0, dtype=np.int64), False
        return np.unique(codes), False

    if blocks.total <= max(cap * 4, 4096):
        p_arr = _sample_indices(blocks.total, blocks.total, int(rng_seed)).numpy()
        codes = blocks.decode_codes(p_arr, stride)
        codes = _filter_excluded_codes(codes, exclude_codes)
        if codes.size == 0:
            return np.empty(0, dtype=np.int64), False
        u_codes, first_idx = np.unique(codes, return_index=True)
        if u_codes.size > cap:
            keep_order = np.argpartition(first_idx, cap)[:cap]
            return np.sort(u_codes[keep_order]), True
        return u_codes, False

    gen = torch.Generator()
    gen.manual_seed(int(rng_seed))
    collected = np.empty(0, dtype=np.int64)
    capped = False

    for _ in range(6):
        need = cap - int(collected.size)
        batch_draw = min(
            max(int((need + 1) * 1.15) + 64, 256),
            max(cap * 2, 1024),
        )
        p_arr = _randint_with_rng(blocks.total, batch_draw, gen).numpy()
        codes = blocks.decode_codes(p_arr, stride)
        ex_list = (
            list(exclude_codes) + [collected]
            if collected.size > 0
            else list(exclude_codes)
        )
        codes = _filter_excluded_codes(codes, ex_list)
        if codes.size == 0:
            continue
        u_codes, first_idx = np.unique(codes, return_index=True)
        if u_codes.size > need:
            capped = True
            if need > 0:
                keep_order = np.argpartition(first_idx, need)[:need]
                u_codes = np.sort(u_codes[keep_order])
                collected = (
                    np.union1d(collected, u_codes) if collected.size > 0 else u_codes
                )
            break
        collected = np.union1d(collected, u_codes) if collected.size > 0 else u_codes

    if collected.size >= cap:
        capped = True
    return collected, capped


def _build_product_blocks(
    l_base_pool: np.ndarray,
    r_base_pool: np.ndarray,
    block_specs: Sequence[Tuple[int, int, int, int]],
    extra_l: Sequence[np.ndarray] = (),
    extra_r: Sequence[np.ndarray] = (),
) -> _ProductBlocks:
    """Assemble a :class:`_ProductBlocks` instance from base pools, extra slices, and ``(l_off, n_l, r_off, n_r)`` tuples."""
    if not block_specs:
        empty = np.empty(0, dtype=np.int64)
        return _ProductBlocks(
            l_pool=l_base_pool,
            r_pool=r_base_pool,
            l_offsets=empty,
            n_l=empty,
            r_offsets=empty,
            n_r=empty,
            sizes=empty,
            cum_sizes=empty,
            total=0,
        )
    l_pool = (
        np.concatenate([l_base_pool] + list(extra_l)) if extra_l else l_base_pool
    )
    r_pool = (
        np.concatenate([r_base_pool] + list(extra_r)) if extra_r else r_base_pool
    )
    arr = np.asarray(block_specs, dtype=np.int64).reshape(-1, 4)
    l_offsets = arr[:, 0]
    n_l = arr[:, 1]
    r_offsets = arr[:, 2]
    n_r = arr[:, 3]
    sizes = n_l * n_r
    cum_sizes = np.cumsum(sizes)
    return _ProductBlocks(
        l_pool=l_pool,
        r_pool=r_pool,
        l_offsets=l_offsets,
        n_l=n_l,
        r_offsets=r_offsets,
        n_r=n_r,
        sizes=sizes,
        cum_sizes=cum_sizes,
        total=int(cum_sizes[-1]),
    )


def _sort_pairs_lexicographically(pairs: torch.Tensor) -> torch.Tensor:
    """Return a ``(N, 2)`` ``LongTensor`` sorted lexicographically by ``(col 0, col 1)``."""
    p = torch.as_tensor(pairs, dtype=torch.long).reshape(-1, 2)
    if p.shape[0] <= 1:
        return p.contiguous()
    max_col1 = int(p[:, 1].max().item())
    stride = max(max_col1 + 1, 1)
    keys = p[:, 0] * stride + p[:, 1]
    if bool((keys[1:] < keys[:-1]).any().item()):
        order = torch.argsort(keys)
        p = p[order]
    return p.contiguous()


def ontology_sha256(
    ontology_or_dataset: object,
    class_index_dict: Optional[Mapping[str, int]] = None,
    individual_index_dict: Optional[Mapping[str, int]] = None,
    object_property_index_dict: Optional[Mapping[str, int]] = None,
    *,
    rng_seed: int = 0,
    hard_neg_cap: int = HARD_NEG_CAP,
) -> str:
    """Compute a deterministic SHA-256 hex digest for an ontology/dataset, its index dicts, ``rng_seed``, and ``hard_neg_cap``.

    Guarantees:
    - Produces an identical 64-character hex string for identical ontology files
      (or axioms when no file path exists), entity-to-index mappings,
      ``rng_seed``, and ``hard_neg_cap``.
    - Works for both file-backed :class:`~mowl.datasets.PathDataset` instances
      and in-memory ``OWLOntology`` / :class:`~mowl.datasets.Dataset` objects.

    Non-guarantees:
    - When ``ontology_or_dataset`` has an existing ``ontology_path`` file on disk,
      hashes the raw file bytes directly for speed; in-memory modifications to
      ``dataset.ontology`` (such as ``add_axioms``) are not detected unless
      ``ontology_path`` is removed or the ontology object itself is passed.
    - Does not hash validation or testing split ontologies.
    """
    hasher = hashlib.sha256()
    hasher.update(f"tag:{POINTS_CODE_TAG}\n".encode("utf-8"))
    hasher.update(f"seed:{int(rng_seed)}\n".encode("utf-8"))
    hasher.update(f"hard_neg_cap:{int(hard_neg_cap)}\n".encode("utf-8"))

    ont = getattr(ontology_or_dataset, "ontology", ontology_or_dataset)
    ont_path = getattr(ontology_or_dataset, "ontology_path", None)
    if isinstance(ont_path, str) and os.path.exists(ont_path):
        with open(ont_path, "rb") as fh:
            while chunk := fh.read(65536):
                hasher.update(chunk)
    elif hasattr(ont, "getAxioms"):
        for ax_str in sorted(str(ax.toString()) for ax in ont.getAxioms()):
            hasher.update(ax_str.encode("utf-8"))
            hasher.update(b"\n")
    else:
        hasher.update(repr(ontology_or_dataset).encode("utf-8"))

    if class_index_dict:
        for k, v in sorted(class_index_dict.items()):
            if not is_auxiliary_iri(k) and not str(k).startswith(AUX_NS):
                hasher.update(f"C:{k}={int(v)}\n".encode("utf-8"))
    if individual_index_dict:
        for k, v in sorted(individual_index_dict.items()):
            hasher.update(f"I:{k}={int(v)}\n".encode("utf-8"))
    if object_property_index_dict:
        for k, v in sorted(object_property_index_dict.items()):
            if not is_auxiliary_iri(k) and not str(k).startswith(AUX_NS):
                hasher.update(f"R:{k}={int(v)}\n".encode("utf-8"))

    return hasher.hexdigest()


@dataclass
class PointSupervision:
    """Precomputed canonical-element supervision pairs for point-based training.

    Guarantees:
    - Stores only flat index tensors (:attr:`pos_pairs`, :attr:`hard_neg_pairs`,
      :attr:`role_pairs`, :attr:`element_class_index`, :attr:`element_is_individual`,
      :attr:`candidate_class_indices`) with no per-element Python sets or dense
      per-element negative pools in memory or on disk.
    - :attr:`pos_pairs` and :attr:`hard_neg_pairs` are maintained in lexicographical
      ``(elem_idx, class_idx)`` order so per-element positive and hard-negative
      slices can be queried in :math:`O(\\log P)` via :func:`torch.searchsorted`.

    Non-guarantees:
    - Does not store negative pools; random negatives are drawn on the fly in
      :meth:`sample` by rejection against :attr:`pos_pairs`.

    Attributes
    ----------
    n_elements : int
        Total number of canonical elements in the element table.
    element_names : List[str]
        Canonical concept or nominal names of the elements in deterministic order.
    element_class_index : torch.LongTensor
        1-D ``LongTensor`` of shape ``(n_elements,)`` storing the class index (when
        ``element_is_individual[i] == False``) or individual index (when
        ``element_is_individual[i] == True``) that element ``i`` belongs to.
    element_is_individual : torch.BoolTensor
        1-D ``BoolTensor`` of shape ``(n_elements,)`` flagging which elements
        correspond to ABox individuals.
    pos_pairs : torch.LongTensor
        2-D ``LongTensor`` of shape ``(P, 2)`` with columns ``[elem_idx, class_idx]``
        for every entailed pair ``(x_D, C)`` with ``C != TOP``.
    hard_neg_pairs : torch.LongTensor
        2-D ``LongTensor`` of shape ``(H, 2)`` with columns ``[elem_idx, class_idx]``
        from the disjointness closure :math:`M \\cdot \\text{Dis} \\cdot M^\\top`
        (capped at :data:`HARD_NEG_CAP`).
    role_pairs : torch.LongTensor
        2-D ``LongTensor`` of shape ``(R, 3)`` with columns
        ``[src_elem_idx, rel_idx, dst_elem_idx]`` for canonical role edges
        restricted to elements in the table.
    candidate_class_indices : torch.LongTensor
        1-D ``LongTensor`` of satisfiable named class indices eligible for
        random-negative sampling.
    rng_seed : int
        Default random seed used when ``sample(n, rng=None)`` is called.
    ontology_sha256 : Optional[str]
        Optional SHA-256 digest of the source ontology and index maps.
    hard_neg_capped : bool
        ``True`` iff the hard-negative candidate pool, the ``NF2`` exact-check
        bound (``max_exact_checks = 4 * hard_neg_cap``), or final
        ``hard_neg_pairs`` hit ``hard_neg_cap`` during construction.
    """

    n_elements: int
    element_names: List[str]
    element_class_index: torch.Tensor
    element_is_individual: torch.Tensor
    pos_pairs: torch.Tensor
    hard_neg_pairs: torch.Tensor
    role_pairs: torch.Tensor
    candidate_class_indices: torch.Tensor = field(
        default_factory=lambda: torch.empty((0,), dtype=torch.long)
    )
    rng_seed: int = 0
    ontology_sha256: Optional[str] = None
    hard_neg_capped: bool = False

    CODE_TAG: str = POINTS_CODE_TAG

    def __post_init__(self) -> None:
        self.n_elements = int(len(self.element_names))
        self.element_class_index = torch.as_tensor(
            self.element_class_index, dtype=torch.long
        ).reshape(self.n_elements)
        self.element_is_individual = torch.as_tensor(
            self.element_is_individual, dtype=torch.bool
        ).reshape(self.n_elements)
        self.pos_pairs = _sort_pairs_lexicographically(self.pos_pairs)
        self.hard_neg_pairs = _sort_pairs_lexicographically(self.hard_neg_pairs)
        self.role_pairs = torch.as_tensor(self.role_pairs, dtype=torch.long).reshape(
            -1, 3
        )
        self.candidate_class_indices = torch.as_tensor(
            self.candidate_class_indices, dtype=torch.long
        ).reshape(-1)
        if self.candidate_class_indices.numel() == 0 and self.n_elements > 0:
            cls_mask = ~self.element_is_individual
            if bool(cls_mask.any().item()):
                self.candidate_class_indices = torch.unique(
                    self.element_class_index[cls_mask], sorted=True
                )
        elif self.candidate_class_indices.numel() > 1:
            if bool(
                (
                    self.candidate_class_indices[1:]
                    < self.candidate_class_indices[:-1]
                ).any().item()
            ):
                self.candidate_class_indices = torch.sort(
                    self.candidate_class_indices
                ).values

        self.hard_neg_capped = bool(self.hard_neg_capped)
        self._default_gen = torch.Generator()
        self._default_gen.manual_seed(int(self.rng_seed))

    @classmethod
    def from_sparse_membership(
        cls,
        M: torch.Tensor,
        element_names: Sequence[str],
        element_class_index: torch.Tensor,
        element_is_individual: torch.Tensor,
        disjoint_pairs: Union[torch.Tensor, Sequence[Tuple[int, int]]] = (),
        *,
        bot_classes: Union[torch.Tensor, Sequence[int]] = (),
        conjunction_pairs: Union[torch.Tensor, Sequence[Tuple[int, int]]] = (),
        cm: Optional[CanonicalModel] = None,
        class_id_to_iri: Optional[Mapping[int, str]] = None,
        candidate_class_indices: Optional[torch.Tensor] = None,
        class_element_rows: Optional[torch.Tensor] = None,
        role_pairs: Optional[torch.Tensor] = None,
        hard_neg_cap: int = HARD_NEG_CAP,
        rng_seed: int = 0,
        ontology_sha256: Optional[str] = None,
    ) -> PointSupervision:
        """Build :class:`PointSupervision` from a sparse element × concept membership matrix ``M``.

        Computes positive pairs directly from the nonzero entries of ``M`` over
        ``candidate_class_indices``, and hard negatives from (a) the one-step
        disjointness closure :math:`M \\cdot \\text{Dis} \\cdot M_{\\text{cls}}^\\top`
        plus, when ``conjunction_pairs`` and ``cm`` are supplied, (b) an exact
        joint-satisfiability check over candidate pairs from
        :math:`M \\cdot \\text{Conj} \\cdot M_{\\text{cls}}^\\top` for binary
        conjunction axioms :math:`A \\sqcap B \\sqsubseteq E` (:math:`E \\neq \\bot`),
        bounded by at most ``max_exact_checks = 4 * hard_neg_cap`` candidate
        checks in total and capped at ``hard_neg_cap`` pairs sampled with
        ``rng_seed``, recording whether either bound was hit in
        :attr:`hard_neg_capped`.

        Guarantees:
        - Scales in time and memory with ``nnz(M)`` and the number of disjointness /
          conjunction pairs rather than :math:`|\\text{elements}| \\times |\\text{classes}|`.
        - Bounds step (b) exact checks at ``max_exact_checks = 4 * hard_neg_cap``
          and caps ``hard_neg_pairs`` at ``hard_neg_cap`` (default ``2,000,000``)
          using the dataset random generator seeded with ``rng_seed``, setting
          :attr:`hard_neg_capped` to ``True`` iff either bound was hit.

        Non-guarantees:
        - When ``cm`` is ``None``, only one-step pairs arising from ``disjoint_pairs``
          (:math:`A \\sqcap B \\sqsubseteq \\bot`) and ``bot_classes``
          (:math:`A \\sqsubseteq \\bot`) are included in ``hard_neg_pairs``.

        Parameters
        ----------
        M : torch.Tensor
            Sparse COO boolean (or numeric) tensor of shape ``(n_elements, n_cols)``
            where ``M[e, c]`` is nonzero iff element ``e`` is subsumed by concept ``c``.
        element_names : Sequence[str]
            Element names of length ``n_elements``.
        element_class_index : torch.Tensor
            1-D ``LongTensor`` of shape ``(n_elements,)`` with each element's own
            class or individual index.
        element_is_individual : torch.Tensor
            1-D ``BoolTensor`` of shape ``(n_elements,)`` flagging individual elements.
        disjoint_pairs : torch.Tensor or Sequence[Tuple[int, int]], optional
            Pairs ``(A, B)`` of column indices in ``M`` such that
            :math:`A \\sqcap B \\sqsubseteq \\bot`.
        bot_classes : torch.Tensor or Sequence[int], optional
            Column indices ``A`` in ``M`` such that :math:`A \\sqsubseteq \\bot`.
        conjunction_pairs : torch.Tensor or Sequence[Tuple[int, int]], optional
            Pairs ``(A, B)`` of column indices in ``M`` for binary conjunction axioms
            :math:`A \\sqcap B \\sqsubseteq E` with :math:`E \\neq \\bot`.
        cm : CanonicalModel, optional
            Canonical model used to check candidate pairs from ``conjunction_pairs``.
        class_id_to_iri : Mapping[int, str], optional
            Mapping from integer class index to class IRI for candidate pair checking.
        candidate_class_indices : torch.Tensor, optional
            Sorted 1-D ``LongTensor`` of target named class indices. Defaults to
            the unique class indices of non-individual elements.
        class_element_rows : torch.Tensor, optional
            1-D ``LongTensor`` of row indices in ``M`` corresponding to
            ``candidate_class_indices``.
        role_pairs : torch.Tensor, optional
            2-D ``LongTensor`` of shape ``(R, 3)`` with canonical role triples.
        hard_neg_cap : int, default 2_000_000
            Maximum number of hard negative pairs retained.
        rng_seed : int, default 0
            Random seed for capping ``hard_neg_pairs`` and default :meth:`sample` calls.
        ontology_sha256 : str, optional
            Optional SHA-256 digest stored for cache validation.

        Returns
        -------
        PointSupervision
            Constructed supervision structure.
        """
        elem_names_list = list(element_names)
        n_elements = len(elem_names_list)
        elem_cls_idx = torch.as_tensor(element_class_index, dtype=torch.long).reshape(
            n_elements
        )
        elem_is_ind = torch.as_tensor(element_is_individual, dtype=torch.bool).reshape(
            n_elements
        )

        if candidate_class_indices is None:
            cls_mask = ~elem_is_ind
            if bool(cls_mask.any().item()):
                cand_cls = torch.unique(elem_cls_idx[cls_mask], sorted=True)
            else:
                cand_cls = torch.empty((0,), dtype=torch.long)
        else:
            cand_cls = torch.unique(
                torch.as_tensor(candidate_class_indices, dtype=torch.long).reshape(-1),
                sorted=True,
            )

        if isinstance(M, torch.Tensor):
            n_cols = int(M.shape[1]) if M.dim() == 2 else 0
            m_coal = M.coalesce()
            m_indices = m_coal.indices()
            if m_indices.numel() > 0 and m_coal.values().dtype == torch.bool:
                vals = m_coal.values()
                if not bool(vals.all().item()):
                    m_indices = m_indices[:, vals]
        else:
            m_coo = sp.coo_matrix(M)
            n_cols = int(m_coo.shape[1])
            if m_coo.nnz > 0:
                mask = np.asarray(m_coo.data, dtype=bool)
                rows_np = np.asarray(m_coo.row[mask], dtype=np.int64)
                cols_np = np.asarray(m_coo.col[mask], dtype=np.int64)
                m_indices = torch.stack(
                    [torch.from_numpy(rows_np), torch.from_numpy(cols_np)], dim=0
                )
            else:
                m_indices = torch.empty((2, 0), dtype=torch.long)

        # 1. Positive pairs from sparse M restricted to candidate_class_indices
        if m_indices.shape[1] == 0 or cand_cls.numel() == 0 or n_cols == 0:
            pos_pairs = torch.empty((0, 2), dtype=torch.long)
        else:
            all_cols_are_cand = (
                cand_cls.numel() == n_cols
                and int(cand_cls[0].item()) == 0
                and int(cand_cls[-1].item()) == n_cols - 1
            )
            if all_cols_are_cand:
                pos_pairs = m_indices.t().contiguous().to(dtype=torch.long)
            else:
                is_cand_col = torch.zeros((n_cols,), dtype=torch.bool)
                valid_cand = cand_cls[(cand_cls >= 0) & (cand_cls < n_cols)]
                is_cand_col[valid_cand] = True
                keep_pos = is_cand_col[m_indices[1]]
                pos_pairs = (
                    m_indices[:, keep_pos].t().contiguous().to(dtype=torch.long)
                )

        # 2. Hard negative pairs:
        #    (a) one-step sparse product M * Dis * M_cls^T
        #    (b) exact check on conjunction-triggering candidate pairs M * Conj * M_cls^T
        dis_t = (
            torch.as_tensor(disjoint_pairs, dtype=torch.long).reshape(-1, 2)
            if isinstance(disjoint_pairs, torch.Tensor) or len(disjoint_pairs) > 0
            else torch.empty((0, 2), dtype=torch.long)
        )
        bot_t = (
            torch.as_tensor(bot_classes, dtype=torch.long).reshape(-1)
            if isinstance(bot_classes, torch.Tensor) or len(bot_classes) > 0
            else torch.empty((0,), dtype=torch.long)
        )
        conj_t = (
            torch.as_tensor(conjunction_pairs, dtype=torch.long).reshape(-1, 2)
            if isinstance(conjunction_pairs, torch.Tensor) or len(conjunction_pairs) > 0
            else torch.empty((0, 2), dtype=torch.long)
        )

        hard_neg_capped = False
        if (
            n_elements == 0
            or cand_cls.numel() == 0
            or n_cols == 0
            or m_indices.shape[1] == 0
            or (dis_t.shape[0] == 0 and bot_t.numel() == 0 and conj_t.shape[0] == 0)
        ):
            hard_neg_pairs = torch.empty((0, 2), dtype=torch.long)
        else:
            m_rows_np = m_indices[0].cpu().numpy()
            m_cols_np = m_indices[1].cpu().numpy()
            m_csr = sp.csr_matrix(
                (np.ones(m_rows_np.shape[0], dtype=bool), (m_rows_np, m_cols_np)),
                shape=(n_elements, n_cols),
                dtype=bool,
            )

            if class_element_rows is not None:
                cls_rows_np = (
                    torch.as_tensor(class_element_rows, dtype=torch.long)
                    .reshape(-1)
                    .cpu()
                    .numpy()
                )
                cls_target_ids_np = cand_cls.cpu().numpy()
            else:
                is_cand_col_np = np.zeros(n_cols, dtype=bool)
                cand_np = cand_cls.cpu().numpy()
                valid_cand_np = cand_np[(cand_np >= 0) & (cand_np < n_cols)]
                is_cand_col_np[valid_cand_np] = True
                cls_elem_mask_np = (~elem_is_ind.cpu().numpy()) & is_cand_col_np[
                    elem_cls_idx.cpu().numpy()
                ]
                cls_rows_np = np.flatnonzero(cls_elem_mask_np)
                cls_target_ids_np = elem_cls_idx.cpu().numpy()[cls_rows_np]

            m_cls_csr = m_csr[cls_rows_np]
            m_csc = m_csr.tocsc()
            m_csc.sort_indices()
            m_cls_csc = m_cls_csr.tocsc()
            m_cls_csc.sort_indices()

            l_base_pool = m_csc.indices.astype(np.int64, copy=False)
            l_indptr = m_csc.indptr.astype(np.int64, copy=False)
            r_base_pool = cls_target_ids_np[m_cls_csc.indices].astype(
                np.int64, copy=False
            )
            r_indptr = m_cls_csc.indptr.astype(np.int64, copy=False)

            pos_sorted = (
                _sort_pairs_lexicographically(pos_pairs)
                if pos_pairs.shape[0] > 0
                else torch.empty((0, 2), dtype=torch.long)
            )
            max_cls_id = max(
                int(cand_cls.max().item()) if cand_cls.numel() > 0 else 0,
                int(pos_sorted[:, 1].max().item()) if pos_sorted.shape[0] > 0 else 0,
            )
            stride = max(max_cls_id + 1, 1)
            pos_codes = (
                (
                    pos_sorted[:, 0].cpu().numpy().astype(np.int64) * stride
                    + pos_sorted[:, 1].cpu().numpy().astype(np.int64)
                )
                if pos_sorted.shape[0] > 0
                else np.empty(0, dtype=np.int64)
            )

            # Step (a): Product blocks for ⊥-sources (A & B <= bot and A <= bot)
            specs_a: List[Tuple[int, int, int, int]] = []
            extra_l_a: List[np.ndarray] = []
            extra_r_a: List[np.ndarray] = []

            if dis_t.shape[0] > 0:
                d_np = dis_t.cpu().numpy()
                valid_d = (
                    (d_np[:, 0] >= 0)
                    & (d_np[:, 0] < n_cols)
                    & (d_np[:, 1] >= 0)
                    & (d_np[:, 1] < n_cols)
                )
                d_np = d_np[valid_d]
                if d_np.shape[0] > 0:
                    seen_dir_a: set[Tuple[int, int]] = set()
                    for a_col, b_col in d_np.tolist():
                        for u_col, v_col in ((int(a_col), int(b_col)), (int(b_col), int(a_col))):
                            if (u_col, v_col) in seen_dir_a:
                                continue
                            seen_dir_a.add((u_col, v_col))
                            l_off = int(l_indptr[u_col])
                            n_l_u = int(l_indptr[u_col + 1] - l_off)
                            r_off = int(r_indptr[v_col])
                            n_r_v = int(r_indptr[v_col + 1] - r_off)
                            if n_l_u > 0 and n_r_v > 0:
                                specs_a.append((l_off, n_l_u, r_off, n_r_v))

            if bot_t.numel() > 0 and cls_target_ids_np.size > 0:
                b_np = np.unique(bot_t.cpu().numpy())
                b_np = b_np[(b_np >= 0) & (b_np < n_cols)]
                if b_np.size > 0:
                    all_elem_off: Optional[int] = None
                    all_cls_off: Optional[int] = None
                    l_cur = int(l_base_pool.size)
                    r_cur = int(r_base_pool.size)
                    for b_col in b_np.tolist():
                        u_col = int(b_col)
                        l_off = int(l_indptr[u_col])
                        n_l_u = int(l_indptr[u_col + 1] - l_off)
                        if n_l_u > 0:
                            if all_cls_off is None:
                                all_cls_off = r_cur
                                extra_r_a.append(
                                    cls_target_ids_np.astype(np.int64, copy=False)
                                )
                                r_cur += int(cls_target_ids_np.size)
                            specs_a.append(
                                (l_off, n_l_u, all_cls_off, int(cls_target_ids_np.size))
                            )
                        r_off = int(r_indptr[u_col])
                        n_r_u = int(r_indptr[u_col + 1] - r_off)
                        if n_r_u > 0:
                            if all_elem_off is None:
                                all_elem_off = l_cur
                                extra_l_a.append(np.arange(n_elements, dtype=np.int64))
                                l_cur += n_elements
                            specs_a.append((all_elem_off, n_elements, r_off, n_r_u))

            blocks_a = _build_product_blocks(
                l_base_pool, r_base_pool, specs_a, extra_l_a, extra_r_a
            )
            hard_a_codes, capped_a = _sample_or_enumerate_blocks(
                blocks_a,
                stride=stride,
                exclude_codes=[pos_codes],
                cap=int(hard_neg_cap),
                rng_seed=int(rng_seed),
            )

            # Step (b): Product blocks for NF2 conjunction axioms A & B <= E (E != bot)
            hard_b_codes = np.empty(0, dtype=np.int64)
            capped_b = False
            if conj_t.shape[0] > 0 and cm is not None and cls_rows_np.size > 0:
                c_np = conj_t.cpu().numpy()
                valid_c = (
                    (c_np[:, 0] >= 0)
                    & (c_np[:, 0] < n_cols)
                    & (c_np[:, 1] >= 0)
                    & (c_np[:, 1] < n_cols)
                    & (c_np[:, 0] != c_np[:, 1])
                )
                c_np = c_np[valid_c]
                if c_np.shape[0] > 0:
                    specs_b: List[Tuple[int, int, int, int]] = []
                    extra_l_b: List[np.ndarray] = []
                    extra_r_b: List[np.ndarray] = []
                    l_cur_b = int(l_base_pool.size)
                    r_cur_b = int(r_base_pool.size)
                    seen_undir_b: set[Tuple[int, int]] = set()

                    for a_col, b_col in c_np.tolist():
                        u_col, v_col = int(a_col), int(b_col)
                        pair_key = (min(u_col, v_col), max(u_col, v_col))
                        if pair_key in seen_undir_b:
                            continue
                        seen_undir_b.add(pair_key)

                        l_off_u = int(l_indptr[u_col])
                        n_l_u = int(l_indptr[u_col + 1] - l_off_u)
                        l_off_v = int(l_indptr[v_col])
                        n_l_v = int(l_indptr[v_col + 1] - l_off_v)
                        r_off_u = int(r_indptr[u_col])
                        n_r_u = int(r_indptr[u_col + 1] - r_off_u)
                        r_off_v = int(r_indptr[v_col])
                        n_r_v = int(r_indptr[v_col + 1] - r_off_v)

                        if max(n_l_u, n_l_v, n_r_u, n_r_v) <= 1024:
                            elem_u = l_base_pool[l_off_u : l_off_u + n_l_u]
                            elem_v = l_base_pool[l_off_v : l_off_v + n_l_v]
                            cls_u = r_base_pool[r_off_u : r_off_u + n_r_u]
                            cls_v = r_base_pool[r_off_v : r_off_v + n_r_v]

                            l_uv = np.setdiff1d(elem_u, elem_v, assume_unique=True)
                            r_uv = np.setdiff1d(cls_v, cls_u)
                            if l_uv.size > 0 and r_uv.size > 0:
                                extra_l_b.append(l_uv)
                                extra_r_b.append(r_uv)
                                specs_b.append(
                                    (l_cur_b, int(l_uv.size), r_cur_b, int(r_uv.size))
                                )
                                l_cur_b += int(l_uv.size)
                                r_cur_b += int(r_uv.size)

                            l_vu = np.setdiff1d(elem_v, elem_u, assume_unique=True)
                            r_vu = np.setdiff1d(cls_u, cls_v)
                            if l_vu.size > 0 and r_vu.size > 0:
                                extra_l_b.append(l_vu)
                                extra_r_b.append(r_vu)
                                specs_b.append(
                                    (l_cur_b, int(l_vu.size), r_cur_b, int(r_vu.size))
                                )
                                l_cur_b += int(l_vu.size)
                                r_cur_b += int(r_vu.size)
                        else:
                            if n_l_u > 0 and n_r_v > 0:
                                specs_b.append((l_off_u, n_l_u, r_off_v, n_r_v))
                            if n_l_v > 0 and n_r_u > 0:
                                specs_b.append((l_off_v, n_l_v, r_off_u, n_r_u))

                    blocks_b = _build_product_blocks(
                        l_base_pool, r_base_pool, specs_b, extra_l_b, extra_r_b
                    )
                    if blocks_b.total > 0:
                        if class_id_to_iri is not None:
                            id_to_iri: Dict[int, str] = dict(class_id_to_iri)
                        else:
                            id_to_iri = {
                                int(cls_target_ids_np[i]): elem_names_list[int(cls_rows_np[i])]
                                for i in range(cls_rows_np.size)
                            }

                        nf2_index: Dict[str, List[NF2]] = {}
                        for ax in cm.axioms:
                            if isinstance(ax, NF2):
                                nf2_index.setdefault(ax.sub1, []).append(ax)
                                if ax.sub2 != ax.sub1:
                                    nf2_index.setdefault(ax.sub2, []).append(ax)
                        pair_cache: Dict[FrozenSet[str], bool] = {}

                        max_exact_checks = (
                            MAX_EXACT_CHECKS_FACTOR * int(hard_neg_cap)
                            if hard_neg_cap > 0
                            else 0
                        )
                        cand_b_codes, capped_b_cand = _sample_or_enumerate_blocks(
                            blocks_b,
                            stride=stride,
                            exclude_codes=[pos_codes, hard_a_codes],
                            cap=max_exact_checks,
                            rng_seed=int(rng_seed),
                        )
                        if hard_neg_cap > 0 and cand_b_codes.size > 1:
                            perm = _sample_indices(
                                int(cand_b_codes.size),
                                int(cand_b_codes.size),
                                int(rng_seed),
                            ).numpy()
                            cand_b_codes = cand_b_codes[perm]

                        capped_b = bool(capped_b_cand)
                        unsat_b_list: List[int] = []
                        n_exact_checks = 0
                        for code in cand_b_codes.tolist():
                            e_idx = int(code // stride)
                            c_idx = int(code % stride)
                            d_name = elem_names_list[e_idx]
                            c_iri = id_to_iri.get(c_idx)
                            if c_iri is None:
                                continue
                            if (
                                max_exact_checks > 0
                                and n_exact_checks >= max_exact_checks
                            ):
                                capped_b = True
                                break
                            n_exact_checks += 1
                            if not _pair_joint_satisfiable(
                                cm, d_name, c_iri, nf2_index, pair_cache
                            ):
                                if (
                                    hard_neg_cap > 0
                                    and len(unsat_b_list) >= hard_neg_cap
                                ):
                                    capped_b = True
                                    break
                                unsat_b_list.append(int(code))
                        if (
                            max_exact_checks > 0
                            and n_exact_checks >= max_exact_checks
                        ):
                            capped_b = True
                        if unsat_b_list:
                            hard_b_codes = np.sort(
                                np.asarray(unsat_b_list, dtype=np.int64)
                            )

            hard_neg_capped = bool(capped_a or capped_b)
            if hard_a_codes.size == 0 and hard_b_codes.size == 0:
                hard_neg_pairs = torch.empty((0, 2), dtype=torch.long)
            else:
                combined_codes = (
                    hard_a_codes
                    if hard_b_codes.size == 0
                    else (
                        hard_b_codes
                        if hard_a_codes.size == 0
                        else np.union1d(hard_a_codes, hard_b_codes)
                    )
                )
                if hard_neg_cap > 0 and combined_codes.size > hard_neg_cap:
                    hard_neg_capped = True
                    keep_idx = np.sort(
                        _sample_indices(
                            int(combined_codes.size),
                            int(hard_neg_cap),
                            int(rng_seed),
                        ).numpy()
                    )
                    combined_codes = combined_codes[keep_idx]

                h_elem = torch.from_numpy(
                    (combined_codes // stride).astype(np.int64, copy=False)
                )
                h_cls = torch.from_numpy(
                    (combined_codes % stride).astype(np.int64, copy=False)
                )
                hard_neg_pairs = torch.stack([h_elem, h_cls], dim=1).contiguous()

        eff_roles = (
            torch.as_tensor(role_pairs, dtype=torch.long).reshape(-1, 3)
            if role_pairs is not None
            else torch.empty((0, 3), dtype=torch.long)
        )

        return cls(
            n_elements=n_elements,
            element_names=elem_names_list,
            element_class_index=elem_cls_idx,
            element_is_individual=elem_is_ind,
            pos_pairs=pos_pairs,
            hard_neg_pairs=hard_neg_pairs,
            role_pairs=eff_roles,
            candidate_class_indices=cand_cls,
            rng_seed=int(rng_seed),
            ontology_sha256=ontology_sha256,
            hard_neg_capped=hard_neg_capped,
        )

    @classmethod
    def from_canonical_model(
        cls,
        cm: CanonicalModel,
        class_index_dict: Mapping[str, int],
        individual_index_dict: Optional[Mapping[str, int]] = None,
        object_property_index_dict: Optional[Mapping[str, int]] = None,
        *,
        relation_index_dict: Optional[Mapping[str, int]] = None,
        rng_seed: int = 0,
        allow_absent_individuals: bool = True,
        ontology_sha256: Optional[str] = None,
        hard_neg_cap: int = HARD_NEG_CAP,
    ) -> PointSupervision:
        """Build :class:`PointSupervision` from a :class:`~mowl.geometric.completion.CanonicalModel`.

        Enumerates positive pairs from the sparse ancestor sets
        ``cm.completion.sub_super`` and hard negatives via (a) the one-step
        sparse disjointness closure :math:`M \\cdot \\text{Dis} \\cdot M^\\top`
        plus (b) an exact joint-satisfiability pass over candidate pairs from
        :math:`M \\cdot \\text{Conj} \\cdot M^\\top` for every ``NF2`` axiom
        :math:`A \\sqcap B \\sqsubseteq E` with :math:`E \\neq \\bot` (bounded by
        at most ``max_exact_checks = 4 * hard_neg_cap`` candidate checks in
        total, setting :attr:`hard_neg_capped` when the bound is hit).

        Why a single exact check per candidate pair in (b) suffices without
        iterating to a fixpoint:
        In ``cm.completion``, all single-concept rules (``NF1``, ``NF3``,
        ``NF4``, ``NF6``, ``NF7``, and backward ``BOTTOM`` propagation) are
        already saturated on :math:`S(D)` and :math:`S(C)` individually, and a
        fresh conjunction concept :math:`Z \\sqsubseteq D \\sqcap C` has no
        incoming role edges. Consequently, :math:`S(D) \\cup S(C)` can derive new
        concepts only if at least one binary conjunction axiom
        :math:`A \\sqcap B \\sqsubseteq E` has :math:`D \\sqsubseteq A` and
        :math:`C \\sqsubseteq B` (or reverse). Because each candidate pair
        :math:`(D, C)` in (b) is saturated completely to fixpoint across all
        chained axioms when checked for joint satisfiability, one exact check
        per candidate pair finds every multi-step unsatisfiable pair without
        outer iteration.

        Guarantees:
        - When ``hard_neg_capped`` is ``False``, :attr:`hard_neg_pairs` equals
          the exact set :math:`\\{(x_D, C) : \\neg \\text{cm.joint\\_satisfiable}(\\{D, C\\})\\}`.
        - Never calls ``cm.entails`` or ``cm.joint_satisfiable`` in an
          :math:`|\\text{elements}| \\times |\\text{classes}|` loop.
        - Raises :class:`ValueError` if any individual in ``individual_index_dict``
          is unsatisfiable in ``cm`` (or absent from ``cm`` when
          ``allow_absent_individuals=False``).

        Non-guarantees:
        - Individuals absent from ``cm.individuals`` are skipped when
          ``allow_absent_individuals=True``.

        Parameters
        ----------
        cm : CanonicalModel
            Canonical model of the training ontology.
        class_index_dict : Mapping[str, int]
            Mapping from class IRI to integer class index.
        individual_index_dict : Optional[Mapping[str, int]], optional
            Mapping from individual IRI to integer individual index.
        object_property_index_dict : Optional[Mapping[str, int]], optional
            Mapping from object property (role) IRI to integer relation index.
        relation_index_dict : Optional[Mapping[str, int]], optional
            Keyword alias for ``object_property_index_dict``.
        rng_seed : int, default 0
            Default seed for :meth:`sample` and hard-negative capping.
        allow_absent_individuals : bool, default True
            If ``False``, raises :class:`ValueError` when an individual in
            ``individual_index_dict`` is absent from ``cm.individuals``.
        ontology_sha256 : Optional[str], optional
            Optional SHA-256 digest stored with the supervision object for cache validation.
        hard_neg_cap : int, default 2_000_000
            Maximum number of hard negative pairs retained.
        """
        cls_dict: Dict[str, int] = {str(k): int(v) for k, v in class_index_dict.items()}
        ind_dict: Dict[str, int] = (
            {str(k): int(v) for k, v in individual_index_dict.items()}
            if individual_index_dict is not None
            else {}
        )
        rel_source = (
            object_property_index_dict
            if object_property_index_dict is not None
            else relation_index_dict
        )
        if rel_source is None:
            rel_dict: Dict[str, int] = {
                r_iri: idx for idx, r_iri in enumerate(sorted(cm.role_names))
            }
        else:
            rel_dict = {str(k): int(v) for k, v in rel_source.items()}

        # 1. Satisfiable named classes in class_index_dict (excluding TOP, BOTTOM, auxiliary)
        named_classes: List[Tuple[str, int]] = [
            (c_iri, c_idx)
            for c_iri, c_idx in sorted(cls_dict.items(), key=lambda kv: (kv[1], kv[0]))
            if _is_named_target_class(c_iri, cm)
        ]
        candidate_class_indices = torch.tensor(
            [c_idx for _, c_idx in named_classes], dtype=torch.long
        )
        class_id_to_iri: Dict[int, str] = {c_idx: c_iri for c_iri, c_idx in named_classes}

        # 2. Satisfiable individuals in individual_index_dict
        ind_elements: List[Tuple[str, int]] = []
        for ind_iri, ind_idx in sorted(ind_dict.items(), key=lambda kv: (kv[1], kv[0])):
            nom_name = cm.individuals.get(ind_iri)
            if nom_name is None:
                if not allow_absent_individuals:
                    raise ValueError(
                        f"Individual {ind_iri!r} is absent from the canonical model."
                    )
                continue
            if not cm.is_satisfiable(nom_name):
                raise ValueError(
                    f"Individual {ind_iri!r} is unsatisfiable in the canonical model."
                )
            ind_elements.append((nom_name, ind_idx))

        element_names: List[str] = [c_iri for c_iri, _ in named_classes] + [
            nom_name for nom_name, _ in ind_elements
        ]
        element_class_index = torch.tensor(
            [c_idx for _, c_idx in named_classes]
            + [ind_idx for _, ind_idx in ind_elements],
            dtype=torch.long,
        )
        element_is_individual = torch.tensor(
            [False] * len(named_classes) + [True] * len(ind_elements),
            dtype=torch.bool,
        )

        # 3. Build sparse membership matrix M from ancestor sets cm.completion.sub_super
        # and extract (a) disjointness axioms A \sqcap B \sqsubseteq \bot / A \sqsubseteq \bot
        # plus (b) conjunction axioms A \sqcap B \sqsubseteq E with E != \bot
        col_map: Dict[str, int] = {c_iri: c_idx for c_iri, c_idx in named_classes}
        next_col = (max(cls_dict.values()) + 1) if cls_dict else 0

        disjoint_pairs: List[Tuple[int, int]] = []
        bot_classes: List[int] = []
        conjunction_pairs: List[Tuple[int, int]] = []
        for ax in cm.axioms:
            if isinstance(ax, NF2):
                for cname in (ax.sub1, ax.sub2):
                    if (
                        cname not in (TOP, BOTTOM)
                        and cname not in col_map
                        and not cm.completion.is_unsatisfiable(cname)
                    ):
                        col_map[cname] = next_col
                        next_col += 1
                c1 = col_map.get(ax.sub1)
                c2 = col_map.get(ax.sub2)
                if c1 is not None and c2 is not None:
                    if cm.completion.is_unsatisfiable(ax.sup):
                        disjoint_pairs.append((c1, c2))
                    elif c1 != c2:
                        conjunction_pairs.append((c1, c2))
            elif isinstance(ax, NF1) and cm.completion.is_unsatisfiable(ax.sup):
                cname = ax.sub
                if cname not in (TOP, BOTTOM):
                    if cname not in col_map:
                        col_map[cname] = next_col
                        next_col += 1
                    bot_classes.append(col_map[cname])

        n_elements = len(element_names)
        m_rows: List[int] = []
        m_cols: List[int] = []
        sub_super = cm.completion.sub_super
        for elem_i, d_name in enumerate(element_names):
            for sup_iri in sub_super.get(d_name, ()):
                c_idx = col_map.get(sup_iri)
                if c_idx is not None:
                    m_rows.append(elem_i)
                    m_cols.append(c_idx)

        M = _sparse_bool_coo(m_rows, m_cols, size=(n_elements, max(next_col, 1)))
        class_element_rows = torch.arange(len(named_classes), dtype=torch.long)

        # 4. Canonical role edges (x_C, r, x_D) restricted to elements in the table
        cm_pos_to_elem: Dict[int, int] = {
            cm.element_of(name): elem_i for elem_i, name in enumerate(element_names)
        }
        role_list: List[Tuple[int, int, int]] = []
        for r_iri, r_idx in sorted(rel_dict.items(), key=lambda kv: (kv[1], kv[0])):
            if is_auxiliary_iri(r_iri) or r_iri.startswith(AUX_NS):
                continue
            for pos_src, pos_dst in sorted(cm.role(r_iri)):
                e_src = cm_pos_to_elem.get(pos_src)
                e_dst = cm_pos_to_elem.get(pos_dst)
                if e_src is not None and e_dst is not None:
                    role_list.append((e_src, r_idx, e_dst))

        role_pairs = (
            torch.tensor(role_list, dtype=torch.long)
            if role_list
            else torch.empty((0, 3), dtype=torch.long)
        )

        return cls.from_sparse_membership(
            M=M,
            element_names=element_names,
            element_class_index=element_class_index,
            element_is_individual=element_is_individual,
            disjoint_pairs=disjoint_pairs,
            bot_classes=bot_classes,
            conjunction_pairs=conjunction_pairs,
            cm=cm,
            class_id_to_iri=class_id_to_iri,
            candidate_class_indices=candidate_class_indices,
            class_element_rows=class_element_rows,
            role_pairs=role_pairs,
            hard_neg_cap=hard_neg_cap,
            rng_seed=int(rng_seed),
            ontology_sha256=ontology_sha256,
        )

    def _sample_hard_for_elements(
        self,
        elem_col: torch.Tensor,
        eff_rng: object,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample one hard negative per element in ``elem_col`` that has any in ``hard_neg_pairs``."""
        b_size = int(elem_col.numel())
        if b_size == 0 or self.hard_neg_pairs.shape[0] == 0:
            return (
                torch.zeros((b_size,), dtype=torch.bool),
                torch.empty((0, 2), dtype=torch.long),
            )
        elem_col = elem_col.contiguous()
        hard_elems = self.hard_neg_pairs[:, 0].contiguous()
        lo = torch.searchsorted(hard_elems, elem_col, side="left")
        hi = torch.searchsorted(hard_elems, elem_col, side="right")
        counts = hi - lo
        has_hard = counts > 0
        n_with_hard = int(has_hard.sum().item())
        if n_with_hard == 0:
            return has_hard, torch.empty((0, 2), dtype=torch.long)
        draws = _randint_with_rng(2**31 - 1, n_with_hard, eff_rng)  # type: ignore[arg-type]
        chosen_idx = lo[has_hard] + (draws % counts[has_hard])
        return has_hard, self.hard_neg_pairs[chosen_idx]

    def _sample_rand_for_elements(
        self,
        elem_col: torch.Tensor,
        eff_rng: object,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Rejection-sample one non-entailed class per element in ``elem_col`` against ``pos_pairs``."""
        b_size = int(elem_col.numel())
        n_cand = int(self.candidate_class_indices.numel())
        if b_size == 0 or n_cand == 0:
            return (
                torch.zeros((b_size,), dtype=torch.bool),
                torch.empty((0, 2), dtype=torch.long),
            )

        elem_col = elem_col.contiguous()
        if self.pos_pairs.shape[0] == 0:
            draws = _randint_with_rng(n_cand, b_size, eff_rng)  # type: ignore[arg-type]
            chosen_c = self.candidate_class_indices[draws]
            return (
                torch.ones((b_size,), dtype=torch.bool),
                torch.stack([elem_col, chosen_c], dim=1),
            )

        max_cls = max(
            int(self.candidate_class_indices.max().item()),
            int(self.pos_pairs[:, 1].max().item()),
        )
        stride = max_cls + 1
        pos_codes = (
            self.pos_pairs[:, 0] * stride + self.pos_pairs[:, 1]
        ).contiguous()
        n_pos_total = int(pos_codes.numel())

        chosen_c = torch.empty((b_size,), dtype=torch.long)
        pending = torch.arange(b_size, dtype=torch.long)

        for _ in range(16):
            if pending.numel() == 0:
                break
            draws = _randint_with_rng(n_cand, int(pending.numel()), eff_rng)  # type: ignore[arg-type]
            cand_c = self.candidate_class_indices[draws]
            cand_codes = elem_col[pending] * stride + cand_c
            idx = torch.searchsorted(pos_codes, cand_codes)
            idx_clamped = idx.clamp(max=n_pos_total - 1)
            is_entailed = (idx < n_pos_total) & (pos_codes[idx_clamped] == cand_codes)
            accepted = ~is_entailed
            if bool(accepted.any().item()):
                chosen_c[pending[accepted]] = cand_c[accepted]
            pending = pending[is_entailed]

        has_rand = torch.ones((b_size,), dtype=torch.bool)
        if pending.numel() > 0:
            pos_elems = self.pos_pairs[:, 0].contiguous()
            lo = torch.searchsorted(pos_elems, elem_col[pending], side="left")
            hi = torch.searchsorted(pos_elems, elem_col[pending], side="right")
            cand_list = self.candidate_class_indices.tolist()
            fallback_draws = _randint_with_rng(
                2**31 - 1, int(pending.numel()), eff_rng  # type: ignore[arg-type]
            ).tolist()
            for k, p_idx in enumerate(pending.tolist()):
                l_i = int(lo[k].item())
                h_i = int(hi[k].item())
                entailed_set = set(self.pos_pairs[l_i:h_i, 1].tolist())
                valid_c = [c for c in cand_list if c not in entailed_set]
                if not valid_c:
                    has_rand[p_idx] = False
                else:
                    chosen_c[p_idx] = valid_c[fallback_draws[k] % len(valid_c)]

        if not bool(has_rand.any().item()):
            return has_rand, torch.empty((0, 2), dtype=torch.long)
        valid_rows = torch.stack([elem_col[has_rand], chosen_c[has_rand]], dim=1)
        return has_rand, valid_rows

    def sample(
        self,
        n: int,
        rng: Optional[
            Union[
                int,
                torch.Generator,
                np.random.Generator,
                np.random.RandomState,
                random.Random,
            ]
        ] = None,
    ) -> Dict[str, torch.Tensor]:
        """Sample at most ``n`` pairs in total across positive, hard-negative, random-negative, and role pairs.

        Guarantees:
        - Total returned pairs satisfy
          ``pos.shape[0] + hard.shape[0] + rand.shape[0] + roles.shape[0] <= max(n, 0)``.
        - For each sampled positive ``(x_D, C)`` (when ``n >= 3``), draws one hard
          negative ``(x_D, C_hard)`` if element ``x_D`` has any hard negatives in
          :attr:`hard_neg_pairs`, and one random negative ``(x_D, C_rand)`` by
          rejection sampling against ``x_D``'s positive set in :attr:`pos_pairs`.

        Non-guarantees:
        - When ``n`` is smaller than the total available pairs, positives and
          roles are subsampled without replacement to stay within the total
          budget ``n``.

        Parameters
        ----------
        n : int
            Maximum total number of pairs across all four returned tensors
            (e.g. ``points_per_epoch``).
        rng : int, torch.Generator, np.random.Generator, random.Random, optional
            Random generator or integer seed. When ``None``, uses the internal
            generator seeded by :attr:`rng_seed`.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary with keys ``"pos"`` ``(n_pos, 2)``, ``"hard"`` ``(n_h, 2)``,
            ``"rand"`` ``(n_rand, 2)``, and ``"roles"`` ``(n_r, 3)``.
        """
        cap = max(int(n), 0)
        if cap == 0:
            return {
                "pos": torch.empty((0, 2), dtype=torch.long),
                "hard": torch.empty((0, 2), dtype=torch.long),
                "rand": torch.empty((0, 2), dtype=torch.long),
                "roles": torch.empty((0, 3), dtype=torch.long),
            }

        eff_rng = self._default_gen if rng is None else rng
        if isinstance(eff_rng, int):
            gen = torch.Generator()
            gen.manual_seed(int(eff_rng))
            eff_rng = gen

        n_pos_avail = int(self.pos_pairs.shape[0])
        n_role_avail = int(self.role_pairs.shape[0])

        if n_pos_avail == 0:
            role_idx = _sample_indices(n_role_avail, cap, eff_rng)
            return {
                "pos": torch.empty((0, 2), dtype=torch.long),
                "hard": torch.empty((0, 2), dtype=torch.long),
                "rand": torch.empty((0, 2), dtype=torch.long),
                "roles": self.role_pairs[role_idx],
            }

        role_reserve = min(n_role_avail, cap // 4)
        pos_budget = cap - role_reserve
        n_cand = int(self.candidate_class_indices.numel())
        max_pos_cand = (
            min(n_pos_avail, pos_budget)
            if n_cand <= 1
            else min(n_pos_avail, max(1, pos_budget // 2))
        )

        pos_idx = _sample_indices(n_pos_avail, max_pos_cand, eff_rng)
        pos_cand = self.pos_pairs[pos_idx]
        elem_col = pos_cand[:, 0]
        has_hard, hard_cand = self._sample_hard_for_elements(elem_col, eff_rng)
        has_rand, rand_cand = self._sample_rand_for_elements(elem_col, eff_rng)

        pair_weights = 1 + has_hard.to(dtype=torch.long) + has_rand.to(dtype=torch.long)
        cum_w = torch.cumsum(pair_weights, dim=0)

        if int(cum_w[-1].item()) + n_role_avail <= cap:
            pos = pos_cand
            hard = hard_cand
            rand = rand_cand
            role_idx = _sample_indices(n_role_avail, n_role_avail, eff_rng)
            roles = self.role_pairs[role_idx]
        else:
            k_keep = int((cum_w <= pos_budget).sum().item())
            if k_keep == 0:
                pos = pos_cand[:1]
                hard = torch.empty((0, 2), dtype=torch.long)
                if pos_budget >= 2 and rand_cand.shape[0] > 0 and bool(has_rand[0].item()):
                    rand = rand_cand[:1]
                else:
                    rand = torch.empty((0, 2), dtype=torch.long)
            else:
                pos = pos_cand[:k_keep]
                n_hard_keep = int(has_hard[:k_keep].sum().item())
                n_rand_keep = int(has_rand[:k_keep].sum().item())
                hard = hard_cand[:n_hard_keep]
                rand = rand_cand[:n_rand_keep]

            used = int(pos.shape[0] + hard.shape[0] + rand.shape[0])
            role_cap = min(n_role_avail, max(0, cap - used))
            role_idx = _sample_indices(n_role_avail, role_cap, eff_rng)
            roles = self.role_pairs[role_idx]

        return {
            "pos": pos,
            "hard": hard,
            "rand": rand,
            "roles": roles,
        }

    def to_dict(self) -> Dict[str, object]:
        """Serialise this :class:`PointSupervision` to a plain dictionary."""
        return {
            "code_tag": POINTS_CODE_TAG,
            "ontology_sha256": self.ontology_sha256,
            "n_elements": self.n_elements,
            "element_names": list(self.element_names),
            "element_class_index": self.element_class_index.cpu(),
            "element_is_individual": self.element_is_individual.cpu(),
            "pos_pairs": self.pos_pairs.cpu(),
            "hard_neg_pairs": self.hard_neg_pairs.cpu(),
            "role_pairs": self.role_pairs.cpu(),
            "candidate_class_indices": self.candidate_class_indices.cpu(),
            "rng_seed": int(self.rng_seed),
            "hard_neg_capped": bool(self.hard_neg_capped),
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, object],
        expected_sha256: Optional[str] = None,
    ) -> PointSupervision:
        """Reconstruct a :class:`PointSupervision` from a dictionary produced by :meth:`to_dict`."""
        tag = data.get("code_tag")
        if tag != POINTS_CODE_TAG:
            raise ValueError(
                f"Incompatible PointSupervision code_tag: expected {POINTS_CODE_TAG!r}, got {tag!r}."
            )
        saved_sha = data.get("ontology_sha256")
        if expected_sha256 is not None and saved_sha != expected_sha256:
            raise ValueError(
                f"PointSupervision ontology_sha256 mismatch: expected {expected_sha256!r}, got {saved_sha!r}."
            )
        element_names = list(data["element_names"])  # type: ignore[arg-type]
        return cls(
            n_elements=len(element_names),
            element_names=element_names,
            element_class_index=data["element_class_index"],  # type: ignore[arg-type]
            element_is_individual=data["element_is_individual"],  # type: ignore[arg-type]
            pos_pairs=data["pos_pairs"],  # type: ignore[arg-type]
            hard_neg_pairs=data["hard_neg_pairs"],  # type: ignore[arg-type]
            role_pairs=data["role_pairs"],  # type: ignore[arg-type]
            candidate_class_indices=data.get(  # type: ignore[arg-type]
                "candidate_class_indices", torch.empty((0,), dtype=torch.long)
            ),
            rng_seed=int(data.get("rng_seed", 0)),  # type: ignore[arg-type]
            ontology_sha256=str(saved_sha) if saved_sha is not None else None,
            hard_neg_capped=bool(data.get("hard_neg_capped", False)),
        )

    def save(self, path: Union[str, Path, BinaryIO]) -> None:
        """Save this :class:`PointSupervision` via :func:`torch.save`."""
        if isinstance(path, (str, Path)):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.to_dict(), path)

    @classmethod
    def load(
        cls,
        path: Union[str, Path, BinaryIO],
        expected_sha256: Optional[str] = None,
    ) -> PointSupervision:
        """Load a :class:`PointSupervision` previously saved with :meth:`save`."""
        data = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(data, Mapping):
            raise ValueError(
                f"Expected a dict in saved PointSupervision, got {type(data).__name__}."
            )
        return cls.from_dict(data, expected_sha256=expected_sha256)


def resolve_points_cache_dir(
    dataset: object, cache_dir: Optional[Union[str, Path]] = None
) -> Path:
    """Resolve the directory for caching :class:`PointSupervision` objects."""
    if cache_dir is not None:
        return Path(cache_dir)
    data_root = getattr(dataset, "data_root", None)
    if data_root is not None:
        return Path(data_root)
    return Path.home() / ".cache" / "mowl" / "points"


def build_or_load_point_supervision(
    dataset: object,
    class_index_dict: Mapping[str, int],
    individual_index_dict: Optional[Mapping[str, int]] = None,
    object_property_index_dict: Optional[Mapping[str, int]] = None,
    *,
    rng_seed: int = 0,
    hard_neg_cap: int = HARD_NEG_CAP,
    cache_dir: Optional[Union[str, Path]] = None,
    use_cache: bool = True,
) -> PointSupervision:
    """Build or load a cached :class:`PointSupervision` for ``dataset``.

    Keyed by ``ontology_sha256(dataset, ..., rng_seed=rng_seed, hard_neg_cap=hard_neg_cap)``,
    ``rng_seed``, ``hard_neg_cap``, and :data:`POINTS_CODE_TAG` under
    ``dataset.data_root`` (when present) or ``~/.cache/mowl/points/``.
    """
    sha = ontology_sha256(
        dataset,
        class_index_dict=class_index_dict,
        individual_index_dict=individual_index_dict,
        object_property_index_dict=object_property_index_dict,
        rng_seed=rng_seed,
        hard_neg_cap=hard_neg_cap,
    )
    resolved_dir = resolve_points_cache_dir(dataset, cache_dir=cache_dir)
    cache_path = (
        resolved_dir
        / f"points_{POINTS_CODE_TAG}_seed{int(rng_seed)}_cap{int(hard_neg_cap)}_{sha[:16]}.pt"
    )

    if use_cache and cache_path.exists():
        try:
            return PointSupervision.load(cache_path, expected_sha256=sha)
        except Exception as exc:
            logger.warning(
                "Failed to load cached PointSupervision from %s (%s); rebuilding.",
                cache_path,
                exc,
            )

    ont = getattr(dataset, "ontology", dataset)
    cm = CanonicalModel.from_ontology(ont)
    ps = PointSupervision.from_canonical_model(
        cm,
        class_index_dict=class_index_dict,
        individual_index_dict=individual_index_dict,
        object_property_index_dict=object_property_index_dict,
        rng_seed=rng_seed,
        allow_absent_individuals=True,
        ontology_sha256=sha,
        hard_neg_cap=hard_neg_cap,
    )
    if use_cache:
        try:
            ps.save(cache_path)
        except OSError as exc:
            logger.warning(
                "Could not write PointSupervision cache to %s: %s",
                cache_path,
                exc,
            )
    return ps
