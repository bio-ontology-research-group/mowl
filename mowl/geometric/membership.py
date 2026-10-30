"""Precomputed sparse element × class membership and canonical role edges.

Constructs :class:`MembershipStructure` from a :class:`~mowl.geometric.completion.CanonicalModel`
for the canonical-element mixture model (R4).

Guarantees:
- Support :math:`S` consists of the canonical elements of satisfiable named
  concepts in ``class_index_dict`` (excluding ``TOP``, ``BOTTOM``, and auxiliary
  concept names not in ``class_index_dict``) together with the canonical elements
  of individuals.
- :attr:`MembershipStructure.M` is a coalesced ``torch.sparse_coo_tensor`` of
  shape ``(|S|, n_classes)`` and dtype ``torch.bool`` with ``M[x, C] == True``
  if and only if ``cm.entails(element_names[x], class_iri)``.
- :attr:`MembershipStructure.roles` maps each role index ``int`` to a coalesced
  ``torch.sparse_coo_tensor`` of shape ``(|S|, |S|)`` and dtype ``torch.bool``
  containing the canonical edges from :meth:`~mowl.geometric.completion.CanonicalModel.role`
  restricted to :math:`S \\times S` (never allocating a dense ``(|S|, |S|)`` matrix).
- :meth:`MembershipStructure.type_mask` computes the conjunction of columns of
  :attr:`MembershipStructure.M` for a 1-D ``LongTensor`` of class indices using
  sparse index operations only, returning all-``True`` over :math:`S` when the
  input tensor is empty.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Dict, List, Mapping, Optional, Tuple, Union

import torch

from mowl.geometric.completion import CanonicalModel
from mowl.geometric.normal_forms import BOTTOM, TOP

MEMBERSHIP_CODE_TAG: str = "membership-v1"


def _sparse_bool_coo(
    rows: List[int],
    cols: List[int],
    size: Tuple[int, int],
) -> torch.Tensor:
    """Create a coalesced 2-D boolean ``torch.sparse_coo_tensor`` from coordinate lists."""
    if len(rows) == 0:
        indices = torch.empty((2, 0), dtype=torch.long)
        values = torch.empty((0,), dtype=torch.bool)
    else:
        indices = torch.tensor([rows, cols], dtype=torch.long)
        values = torch.ones((len(rows),), dtype=torch.bool)
    return torch.sparse_coo_tensor(
        indices, values, size=size, dtype=torch.bool
    ).coalesce()


@dataclass
class MembershipStructure:
    """Sparse membership and role-edge structure over the canonical support :math:`S`.

    Attributes
    ----------
    element_names : List[str]
        Canonical concept/nominal names of elements in support :math:`S` in a
        fixed deterministic order.
    element_index : Dict[str, int]
        Mapping from each element name in :attr:`element_names` to its row index
        in ``0 .. |S| - 1``.
    M : torch.Tensor
        Coalesced ``torch.sparse_coo_tensor`` of shape ``(|S|, n_classes)`` and
        dtype ``torch.bool`` where ``M[x, c] == True`` iff
        ``cm.entails(element_names[x], class_iri_of_c)``.
    roles : Dict[int, torch.Tensor]
        Mapping from role integer index to a coalesced ``torch.sparse_coo_tensor``
        of shape ``(|S|, |S|)`` and dtype ``torch.bool`` with canonical role edges.
    individual_element : torch.LongTensor
        1-D ``LongTensor`` of shape ``(n_ind,)`` mapping each individual index
        (following ``individual_index_dict``) to its element index in :math:`S`.
    class_index_dict : Dict[str, int]
        Class IRI to column index mapping.
    individual_index_dict : Dict[str, int]
        Individual IRI to individual index mapping.
    role_index_dict : Dict[str, int]
        Role IRI to role key mapping.
    """

    element_names: List[str]
    element_index: Dict[str, int]
    M: torch.Tensor
    roles: Dict[int, torch.Tensor]
    individual_element: torch.Tensor
    class_index_dict: Dict[str, int] = field(default_factory=dict)
    individual_index_dict: Dict[str, int] = field(default_factory=dict)
    role_index_dict: Dict[str, int] = field(default_factory=dict)

    CODE_TAG: str = MEMBERSHIP_CODE_TAG

    def __post_init__(self) -> None:
        if not self.M.is_sparse:
            raise ValueError("M must be a torch.sparse_coo_tensor.")
        self.M = self.M.coalesce()
        self.roles = {int(k): v.coalesce() for k, v in self.roles.items()}
        self.individual_element = self.individual_element.to(dtype=torch.long)

    @classmethod
    def from_canonical_model(
        cls,
        cm: CanonicalModel,
        class_index_dict: Mapping[str, int],
        individual_index_dict: Mapping[str, int],
        relation_index_dict: Optional[Mapping[str, int]] = None,
    ) -> MembershipStructure:
        """Build a :class:`MembershipStructure` from a :class:`~mowl.geometric.completion.CanonicalModel`.

        Parameters
        ----------
        cm : CanonicalModel
            Canonical model of the ontology (including ABox individuals as nominals).
        class_index_dict : Mapping[str, int]
            Dataset mapping from class IRI to column index ``0 .. n_classes - 1``.
        individual_index_dict : Mapping[str, int]
            Dataset mapping from individual IRI to index ``0 .. n_ind - 1``.
        relation_index_dict : Optional[Mapping[str, int]], optional
            Dataset mapping from role IRI to role index. When ``None``, roles in
            ``sorted(cm.role_names)`` are indexed ``0 .. len(cm.role_names) - 1``.
        """
        cls_dict: Dict[str, int] = {str(k): int(v) for k, v in class_index_dict.items()}
        ind_dict: Dict[str, int] = {
            str(k): int(v) for k, v in individual_index_dict.items()
        }
        if relation_index_dict is None:
            rel_dict: Dict[str, int] = {
                r_iri: idx for idx, r_iri in enumerate(sorted(cm.role_names))
            }
        else:
            rel_dict = {str(k): int(v) for k, v in relation_index_dict.items()}

        n_classes = (max(cls_dict.values()) + 1) if cls_dict else 0
        n_ind = (max(ind_dict.values()) + 1) if ind_dict else 0

        # Validate all individuals in individual_index_dict exist and are satisfiable in cm
        sorted_ind_items = sorted(ind_dict.items(), key=lambda kv: kv[1])
        ind_nominals_in_order: List[Tuple[int, str]] = []
        seen_nominals = set()
        for ind_iri, ind_idx in sorted_ind_items:
            nom_name = cm.individuals.get(ind_iri)
            if nom_name is None:
                raise ValueError(
                    f"Individual {ind_iri!r} is absent from the canonical model."
                )
            if not cm.is_satisfiable(nom_name):
                raise ValueError(
                    f"Individual {ind_iri!r} is unsatisfiable in the canonical model."
                )
            ind_nominals_in_order.append((ind_idx, nom_name))
            seen_nominals.add(nom_name)

        # Support S = satisfiable named concepts in class_index_dict (excluding TOP/BOTTOM
        # and auxiliary concepts not in class_index_dict) + individuals' elements
        cm_concepts = cm.concept_names
        named_concept_elements: List[str] = [
            c_iri
            for c_iri, _ in sorted(cls_dict.items(), key=lambda kv: (kv[1], kv[0]))
            if c_iri in cm_concepts
            and c_iri not in (TOP, BOTTOM)
            and cm.is_satisfiable(c_iri)
        ]

        individual_elements: List[str] = [nom for _, nom in ind_nominals_in_order]
        # Include any additional satisfiable individuals from cm.individuals if not in ind_dict
        for ind_iri in sorted(cm.individuals.keys()):
            nom = cm.individuals[ind_iri]
            if nom not in seen_nominals and cm.is_satisfiable(nom):
                individual_elements.append(nom)
                seen_nominals.add(nom)

        element_names: List[str] = named_concept_elements + individual_elements
        element_index: Dict[str, int] = {
            name: idx for idx, name in enumerate(element_names)
        }
        n_elements = len(element_names)

        # Build individual_element: LongTensor (n_ind,)
        ind_elem_tensor = torch.empty((n_ind,), dtype=torch.long)
        for ind_idx, nom_name in ind_nominals_in_order:
            ind_elem_tensor[ind_idx] = element_index[nom_name]

        # Build M: torch.sparse_coo (|S| × n_classes) boolean
        # M[x, C] == 1 iff cm.entails(name_of_x, C)
        m_rows: List[int] = []
        m_cols: List[int] = []
        sub_super = cm.completion.sub_super
        for row_idx, elem_name in enumerate(element_names):
            sups = sub_super.get(elem_name, ())
            for sup_iri in sups:
                col_idx = cls_dict.get(sup_iri)
                if col_idx is not None and cm.entails(elem_name, sup_iri):
                    m_rows.append(row_idx)
                    m_cols.append(col_idx)

        M = _sparse_bool_coo(m_rows, m_cols, size=(n_elements, n_classes))

        # Map canonical domain positions cm.element_of(name) -> index in S
        cm_pos_to_s: Dict[int, int] = {
            cm.element_of(name): s_idx for s_idx, name in enumerate(element_names)
        }

        # Build roles: Dict[int, torch.sparse_coo (|S| × |S|)] from cm.role(role_iri)
        roles: Dict[int, torch.Tensor] = {}
        for r_iri, r_idx in sorted(rel_dict.items(), key=lambda kv: kv[1]):
            r_rows: List[int] = []
            r_cols: List[int] = []
            for pos_i, pos_j in cm.role(r_iri):
                s_i = cm_pos_to_s.get(pos_i)
                s_j = cm_pos_to_s.get(pos_j)
                if s_i is not None and s_j is not None:
                    r_rows.append(s_i)
                    r_cols.append(s_j)
            roles[r_idx] = _sparse_bool_coo(
                r_rows, r_cols, size=(n_elements, n_elements)
            )

        return cls(
            element_names=element_names,
            element_index=element_index,
            M=M,
            roles=roles,
            individual_element=ind_elem_tensor,
            class_index_dict=cls_dict,
            individual_index_dict=ind_dict,
            role_index_dict=rel_dict,
        )

    def type_mask(self, ind_types: torch.Tensor) -> torch.Tensor:
        """Return a boolean mask over :math:`S` for elements belonging to *all* classes in ``ind_types``.

        Parameters
        ----------
        ind_types : torch.LongTensor
            1-D tensor of class indices in ``0 .. n_classes - 1``. When empty,
            returns all-``True`` over :math:`S`.

        Returns
        -------
        torch.BoolTensor
            1-D boolean tensor of shape ``(|S|,)``.
        """
        n_elements, n_classes = self.M.shape
        device = self.M.device
        types_tensor = torch.as_tensor(ind_types, dtype=torch.long, device=device).reshape(-1)
        if types_tensor.numel() == 0:
            return torch.ones((n_elements,), dtype=torch.bool, device=device)

        unique_types = torch.unique(types_tensor)
        if bool((unique_types < 0).any().item()) or bool(
            (unique_types >= n_classes).any().item()
        ):
            raise IndexError(
                f"Class index out of bounds for M with {n_classes} classes."
            )

        m_coal = self.M.coalesce()
        indices = m_coal.indices()
        values = m_coal.values().bool()
        if values.numel() == 0:
            return torch.zeros((n_elements,), dtype=torch.bool, device=device)

        rows = indices[0][values]
        cols = indices[1][values]
        matched = torch.isin(cols, unique_types)
        counts = torch.bincount(rows[matched], minlength=n_elements)
        return counts == unique_types.numel()

    def to_dict(self) -> Dict[str, object]:
        """Serialise this :class:`MembershipStructure` to a plain dictionary."""
        return {
            "code_tag": MEMBERSHIP_CODE_TAG,
            "element_names": list(self.element_names),
            "element_index": dict(self.element_index),
            "M": self.M.coalesce(),
            "roles": {int(k): v.coalesce() for k, v in self.roles.items()},
            "individual_element": self.individual_element,
            "class_index_dict": dict(self.class_index_dict),
            "individual_index_dict": dict(self.individual_index_dict),
            "role_index_dict": dict(self.role_index_dict),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> MembershipStructure:
        """Reconstruct a :class:`MembershipStructure` from a dictionary produced by :meth:`to_dict`."""
        tag = data.get("code_tag")
        if tag != MEMBERSHIP_CODE_TAG:
            raise ValueError(
                f"Incompatible MembershipStructure code_tag: expected {MEMBERSHIP_CODE_TAG!r}, got {tag!r}."
            )
        return cls(
            element_names=list(data["element_names"]),  # type: ignore[arg-type]
            element_index=dict(data["element_index"]),  # type: ignore[arg-type]
            M=data["M"],  # type: ignore[arg-type]
            roles={int(k): v for k, v in dict(data["roles"]).items()},  # type: ignore[arg-type]
            individual_element=data["individual_element"],  # type: ignore[arg-type]
            class_index_dict=dict(data.get("class_index_dict", {})),  # type: ignore[arg-type]
            individual_index_dict=dict(data.get("individual_index_dict", {})),  # type: ignore[arg-type]
            role_index_dict=dict(data.get("role_index_dict", {})),  # type: ignore[arg-type]
        )

    def save(self, path: Union[str, Path, BinaryIO]) -> None:
        """Save this :class:`MembershipStructure` as a plain dictionary via :func:`torch.save`."""
        torch.save(self.to_dict(), path)

    @classmethod
    def load(cls, path: Union[str, Path, BinaryIO]) -> MembershipStructure:
        """Load a :class:`MembershipStructure` previously saved with :meth:`save`."""
        data = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(data, Mapping):
            raise ValueError(
                f"Expected a dict in saved MembershipStructure, got {type(data).__name__}."
            )
        return cls.from_dict(data)
