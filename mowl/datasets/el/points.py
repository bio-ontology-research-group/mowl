"""Canonical-element point supervision for Objective O3 (:class:`PointSupervision`).

Constructs positive membership pairs ``(x_D, C)``, hard negative pairs ``(x_D, C)``
where :math:`D \\sqcap C \\sqsubseteq \\bot` is provable in the completion, random
negative candidates (classes not entailed for :math:`D`), and canonical role edges
``(x_C, r, x_D)`` restricted to the element table from a
:class:`~mowl.geometric.completion.CanonicalModel`.

Guarantees:
- Element table contains one row per satisfiable named concept in
  ``class_index_dict`` (excluding ``TOP``, ``BOTTOM``, and normaliser auxiliary
  classes) followed by one row per satisfiable individual in
  ``individual_index_dict`` present in ``cm.individuals``.
- Unsatisfiable concept names, ``TOP``, ``BOTTOM``, and auxiliary concept names
  are absent from :attr:`PointSupervision.element_names`,
  :attr:`PointSupervision.pos_pairs`, :attr:`PointSupervision.hard_neg_pairs`,
  and sampled random negatives.
- Every pair ``(x_D, C)`` in :attr:`PointSupervision.pos_pairs` is entailed by
  ``cm`` (:meth:`~mowl.geometric.completion.CanonicalModel.entails`), and no pair
  in :attr:`PointSupervision.hard_neg_pairs` or sampled ``"rand"`` is entailed.
- Every pair ``(x_D, C)`` in :attr:`PointSupervision.hard_neg_pairs` satisfies
  ``not cm.joint_satisfiable({D, C})``.
- :meth:`PointSupervision.save` and :meth:`PointSupervision.load` validate
  :data:`POINTS_CODE_TAG` and (when supplied) the ontology SHA-256 digest.

Non-guarantees:
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
from typing import BinaryIO, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from mowl.geometric.completion import CanonicalModel
from mowl.geometric.normal_forms import AUX_NS, BOTTOM, TOP, is_nominal_name
from mowl.ontology.normalize import is_auxiliary_iri

logger = logging.getLogger(__name__)

POINTS_CODE_TAG: str = "points-v1"


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
        arr = rng.integers(0, high, size=size) if isinstance(rng, np.random.Generator) else rng.randint(0, high, size=size)
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
    """Return all ``0 .. total - 1`` when ``total <= cap``, else sample ``cap`` without replacement."""
    if total <= 0 or cap <= 0:
        return torch.empty((0,), dtype=torch.long)
    if total <= cap:
        return torch.arange(total, dtype=torch.long)
    if rng is None:
        return torch.randperm(total, dtype=torch.long)[:cap]
    if isinstance(rng, int):
        gen = torch.Generator()
        gen.manual_seed(int(rng))
        return torch.randperm(total, generator=gen, dtype=torch.long)[:cap]
    if isinstance(rng, torch.Generator):
        return torch.randperm(total, generator=rng, dtype=torch.long)[:cap]
    if isinstance(rng, np.random.Generator):
        chosen = rng.choice(total, size=cap, replace=False)
        return torch.as_tensor(chosen, dtype=torch.long)
    if isinstance(rng, np.random.RandomState):
        chosen = rng.choice(total, size=cap, replace=False)
        return torch.as_tensor(chosen, dtype=torch.long)
    if isinstance(rng, random.Random):
        chosen = rng.sample(range(total), cap)
        return torch.tensor(chosen, dtype=torch.long)
    raise TypeError(f"Unsupported rng type: {type(rng).__name__}")


def ontology_sha256(
    ontology_or_dataset: object,
    class_index_dict: Optional[Mapping[str, int]] = None,
    individual_index_dict: Optional[Mapping[str, int]] = None,
    object_property_index_dict: Optional[Mapping[str, int]] = None,
) -> str:
    """Compute a deterministic SHA-256 hex digest for an ontology/dataset and its index dicts.

    Guarantees:
    - Produces an identical 64-character hex string for identical ontology axioms
      and entity-to-index mappings.
    - Works for both file-backed :class:`~mowl.datasets.PathDataset` instances
      and in-memory ``OWLOntology`` / :class:`~mowl.datasets.Dataset` objects.

    Non-guarantees:
    - Does not hash validation or testing split ontologies.
    """
    hasher = hashlib.sha256()
    hasher.update(f"tag:{POINTS_CODE_TAG}\n".encode("utf-8"))

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
    """Precomputed canonical-element supervision pairs for Objective O3.

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
        for every pair ``(x_D, C)`` with ``not cm.joint_satisfiable({D, C})``.
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

    CODE_TAG: str = POINTS_CODE_TAG

    def __post_init__(self) -> None:
        self.n_elements = int(len(self.element_names))
        self.element_class_index = torch.as_tensor(
            self.element_class_index, dtype=torch.long
        ).reshape(self.n_elements)
        self.element_is_individual = torch.as_tensor(
            self.element_is_individual, dtype=torch.bool
        ).reshape(self.n_elements)
        self.pos_pairs = torch.as_tensor(self.pos_pairs, dtype=torch.long).reshape(-1, 2)
        self.hard_neg_pairs = torch.as_tensor(
            self.hard_neg_pairs, dtype=torch.long
        ).reshape(-1, 2)
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

        # Precompute per-element non-entailed candidate class indices for fast O(1) random neg sampling
        pos_by_elem: Dict[int, set] = {i: set() for i in range(self.n_elements)}
        for e_idx, c_idx in self.pos_pairs.tolist():
            pos_by_elem[int(e_idx)].add(int(c_idx))
        cand_list = self.candidate_class_indices.tolist()
        self._non_entailed_by_elem: List[torch.Tensor] = [
            torch.tensor(
                [c for c in cand_list if c not in pos_by_elem[i]], dtype=torch.long
            )
            for i in range(self.n_elements)
        ]
        self._default_gen = torch.Generator()
        self._default_gen.manual_seed(int(self.rng_seed))

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
    ) -> PointSupervision:
        """Build :class:`PointSupervision` from a :class:`~mowl.geometric.completion.CanonicalModel`.

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
            Default seed for :meth:`sample`.
        allow_absent_individuals : bool, default True
            If ``False``, raises :class:`ValueError` when an individual in
            ``individual_index_dict`` is absent from ``cm.individuals``.
        ontology_sha256 : Optional[str], optional
            Optional SHA-256 digest stored with the supervision object for cache validation.
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

        # 3. Positive pairs (x_D, C) and hard negative pairs (x_D, C)
        pos_list: List[Tuple[int, int]] = []
        hard_neg_list: List[Tuple[int, int]] = []

        has_bottom_axiom = any(
            getattr(ax, "sup", None) == BOTTOM for ax in cm.axioms
        )

        for elem_i, d_name in enumerate(element_names):
            for c_iri, c_idx in named_classes:
                if cm.entails(d_name, c_iri):
                    pos_list.append((elem_i, c_idx))
                elif has_bottom_axiom and not cm.joint_satisfiable({d_name, c_iri}):
                    hard_neg_list.append((elem_i, c_idx))

        pos_pairs = (
            torch.tensor(pos_list, dtype=torch.long)
            if pos_list
            else torch.empty((0, 2), dtype=torch.long)
        )
        hard_neg_pairs = (
            torch.tensor(hard_neg_list, dtype=torch.long)
            if hard_neg_list
            else torch.empty((0, 2), dtype=torch.long)
        )

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

        return cls(
            n_elements=len(element_names),
            element_names=element_names,
            element_class_index=element_class_index,
            element_is_individual=element_is_individual,
            pos_pairs=pos_pairs,
            hard_neg_pairs=hard_neg_pairs,
            role_pairs=role_pairs,
            candidate_class_indices=candidate_class_indices,
            rng_seed=int(rng_seed),
            ontology_sha256=ontology_sha256,
        )

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
        """Sample up to ``n`` positive, hard-negative, random-negative, and role pairs.

        Parameters
        ----------
        n : int
            Maximum number of pairs per category (e.g. ``points_per_epoch``).
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
        eff_rng = self._default_gen if rng is None else rng

        if isinstance(eff_rng, int):
            gen = torch.Generator()
            gen.manual_seed(int(eff_rng))
            eff_rng = gen

        pos_idx = _sample_indices(self.pos_pairs.shape[0], cap, eff_rng)
        pos = self.pos_pairs[pos_idx]

        hard_cap = min(cap, pos.shape[0]) if pos.shape[0] > 0 else cap
        if self.hard_neg_pairs.shape[0] <= cap:
            hard_cap = self.hard_neg_pairs.shape[0]
        hard_idx = _sample_indices(self.hard_neg_pairs.shape[0], hard_cap, eff_rng)
        hard = self.hard_neg_pairs[hard_idx]

        role_idx = _sample_indices(self.role_pairs.shape[0], cap, eff_rng)
        roles = self.role_pairs[role_idx]

        # Sample one random negative (x_D, C_rand) per sampled positive (x_D, C_pos)
        rand_rows: List[Tuple[int, int]] = []
        if pos.shape[0] > 0:
            elem_col = pos[:, 0].tolist()
            # Draw uniform floats/ints in one batch via eff_rng
            raw_draws = _randint_with_rng(2**31 - 1, len(elem_col), eff_rng).tolist()
            for e_idx, draw in zip(elem_col, raw_draws):
                pool = self._non_entailed_by_elem[int(e_idx)]
                if pool.numel() > 0:
                    chosen_c = int(pool[draw % pool.numel()].item())
                    rand_rows.append((int(e_idx), chosen_c))

        rand = (
            torch.tensor(rand_rows, dtype=torch.long)
            if rand_rows
            else torch.empty((0, 2), dtype=torch.long)
        )

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
    cache_dir: Optional[Union[str, Path]] = None,
    use_cache: bool = True,
) -> PointSupervision:
    """Build or load a cached :class:`PointSupervision` for ``dataset``.

    Keyed by ``ontology_sha256(dataset, ...)`` and :data:`POINTS_CODE_TAG` under
    ``dataset.data_root`` (when present) or ``~/.cache/mowl/points/``.
    """
    sha = ontology_sha256(
        dataset,
        class_index_dict=class_index_dict,
        individual_index_dict=individual_index_dict,
        object_property_index_dict=object_property_index_dict,
    )
    resolved_dir = resolve_points_cache_dir(dataset, cache_dir=cache_dir)
    cache_path = resolved_dir / f"points_{POINTS_CODE_TAG}_{sha[:16]}.pt"

    if use_cache and cache_path.exists():
        try:
            loaded = PointSupervision.load(cache_path, expected_sha256=sha)
            loaded.rng_seed = int(rng_seed)
            loaded._default_gen.manual_seed(int(rng_seed))
            return loaded
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
