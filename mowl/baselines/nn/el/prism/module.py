"""PrismModule neural module for :math:`\\mathcal{EL}` prism embeddings."""
from __future__ import annotations

from typing import Optional, Tuple

import torch as th
import torch.nn as nn

from mowl.nn import ELModule
import mowl.baselines.nn.el.prism.losses as L
from mowl.nn.el.structural import DefinitionTable, ExistenceTargets, existence_loss

_PRISM_OBJECTIVES = ("axiom", "structural")
_EMPTY_BOX_READINGS = ("signed", "absolute")


class PrismModule(ELModule):
    """Prism (:math:`\\text{triangle} \\times \\text{box}`) neural module following Federico, Leemhuis & Kutz (DL 2026).

    Every ontology class :math:`C` is represented as a prism
    :math:`P_C = T_C \\times B_C \\subset \\mathbb{R}^2 \\times \\mathbb{R}^{d-2}`
    (§4.1 Definition 4 and Proposition 5 of Federico et al., DL 2026), with
    :math:`2d - 1` scalar parameters per class:

    - ``tri_center`` of shape ``(nb_ont_classes, 2)`` and ``tri_inradius`` of
      shape ``(nb_ont_classes, 1)`` (used as :math:`|\\rho| \\ge 0`), defining the
      three triangle support values :math:`h_k = n_k \\cdot c + |\\rho|`.
    - ``box_center`` and ``box_offset`` of shape ``(nb_ont_classes, d - 2)``
      (offset used as :math:`|o| \\ge 0`).
    - ``rel_embed`` of shape ``(nb_rels, d)``, where the first two coordinates
      translate the triangle and the remaining :math:`d - 2` coordinates
      translate the box.
    - Optional ``ind_embed`` of shape ``(nb_inds, d)`` representing individuals
      as points (degenerate prisms with zero inradius and zero box offsets).

    Guarantees:

    - Raises :class:`ValueError` if ``embed_dim < 3``.
    - :meth:`prism_params` returns ``(h, box_center, |box_offset|)`` of shapes
      ``(B, 3)``, ``(B, d - 2)``, ``(B, d - 2)`` with non-negative ``box_offset``.
    - Every GCI loss returns a column of shape ``(B, 1)`` like every
      :class:`~mowl.nn.ELModule` (``per_axiom_column = False`` gives the flat ``(B,)``).
    - :meth:`gci1_bot_loss` aggregates box coordinates with ``min`` (semantically
      disjointness-if-any-separates; ELBE uses the Euclidean norm).

    Non-guarantees:

    - :meth:`gci0_bot_loss` (under ``objective="axiom"``) and
      :meth:`gci3_bot_loss` have a floor of ``margin`` at ``margin > 0`` since
      :math:`\\sum_k h_k = 3 |\\rho| \\ge 0`; moreover, :meth:`gci3_bot_loss`
      shrinks :math:`C` itself (:math:`\\text{relu}(\\sum_k h_{C,k} + \\text{margin})`),
      which conflicts with :func:`~mowl.nn.el.structural.existence_loss` whenever
      :math:`C` is satisfiable.
    - Under ``objective="structural"``, only the box GCI losses (:meth:`gci0_loss`,
      :meth:`gci0_bot_loss`, :meth:`gci1_loss`, :meth:`gci1_bot_loss`) are squared
      hinges; the role and assertion scores (:meth:`gci2_loss`, :meth:`gci3_loss`,
      :meth:`gci3_bot_loss`, :meth:`class_assertion_loss`,
      :meth:`object_property_assertion_loss`) are unsquared norms/hinges.
    - Only with ``empty_box_reading="absolute"`` (the earlier reading, not the default):
      those role and assertion scores pass ``torch.abs(w)`` on materialised defined classes,
      so a defined class that is empty only in its box part (:math:`w_m < 0` while
      :math:`\\sum_k h_k > 0`) becomes a phantom non-empty box of half-width :math:`|w_m| > 0`.
      The default ``"signed"`` reading does not have this.
    - Touching triangles (:math:`\\sum_k \\min(h_{C,k}, h_{D,k}) = 0`) count as
      disjoint at ``margin = 0``.
    - Role axiom losses (``role_inclusion``, ``role_chain``) are not implemented
      (``role_axiom_capable = False``).
    - Negative sampling is only supported for ``gci2`` and
      ``object_property_assertion`` (``neg_capable_gcis``).

    Parameters
    ----------
    nb_ont_classes : int
        Number of ontology classes.
    nb_rels : int
        Number of object properties (roles).
    nb_inds : Optional[int], default None
        Number of individuals (ABox).
    embed_dim : int, default 50
        Ambient Euclidean dimension :math:`d \\ge 3`.
    margin : float, default 0.1
        Inclusion and separation margin.
    objective : {"axiom", "structural"}, default "axiom"
        Training objective mode: ``"axiom"`` or ``"structural"``.
    max_depth : int, default 8
        Maximum conjunction definition depth :math:`\\text{max\\_level}` under
        ``objective="structural"``.
    structural_definitions : bool, default True
        Whether ``objective="structural"`` compiles acyclic conjunction
        definitions :math:`A \\equiv X \\sqcap Y` via
        :class:`~mowl.nn.el.structural.DefinitionTable`.
    existence_weight : float, default 1.0
        Non-negative weight multiplying :func:`~mowl.nn.el.structural.existence_loss`
        in :meth:`structural_extra_loss`.
    size_reg : float, default 1e-3
        Non-negative weight multiplying the primitive prism size regulariser
        ``mean([|r|, 2|o|])`` in :meth:`structural_extra_loss`.
    gci2_negative_triangle : {"overlap", "inradius"}, default "overlap"
        Triangle part of the negative loss of :math:`C \\sqsubseteq \\exists r.D` and of
        negative role assertions: ``"overlap"`` is the exact overlap test of the translated
        triangle and the triangle of :math:`D` (zero iff they overlap); ``"inradius"`` is the
        earlier per-direction test on the inscribed discs, see
        :func:`~mowl.baselines.nn.el.prism.losses.gci2_score_neg`.
    empty_box_reading : {"signed", "absolute"}, default "signed"
        How the role and assertion scores of ``objective="structural"`` read a DEFINED class
        whose box part is empty (half-width :math:`w_m < 0`). ``"signed"`` gates by
        non-emptiness as :class:`~mowl.nn.el.multiboxel.MultiBoxELModule` does: an empty
        source (``gci2``) or filler (``gci3``, ``gci3_bot``) makes the row vacuously
        satisfied, and an empty target or class is read with its negative half-width, so it
        contains nothing. ``"absolute"`` is the earlier reading that takes ``|w|`` and so
        sees a phantom non-empty box.
    size_reg_without_tables : bool, default False
        The size term has to know which class is ``owl:Thing`` (it must not shrink it) and
        which classes are defined, so by default it is ``0`` until :meth:`set_structural_tables`
        has been called. ``True`` applies it to every class before that too, which shrinks
        ``owl:Thing`` (the earlier behaviour of this module).
    """

    neg_capable_gcis = frozenset({"gci2", "object_property_assertion"})
    role_axiom_capable = False

    #: A positive margin requires the inclusion with a gap (ELBE's convention).
    margin_semantics = "strict"

    def __init__(
        self,
        nb_ont_classes: int,
        nb_rels: int,
        nb_inds: Optional[int] = None,
        embed_dim: int = 50,
        margin: float = 0.1,
        objective: str = "axiom",
        max_depth: int = 8,
        structural_definitions: bool = True,
        existence_weight: float = 1.0,
        size_reg: float = 1e-3,
        gci2_negative_triangle: str = "overlap",
        empty_box_reading: str = "signed",
        size_reg_without_tables: bool = False,
    ) -> None:
        if empty_box_reading not in _EMPTY_BOX_READINGS:
            raise ValueError(
                f"empty_box_reading must be one of {_EMPTY_BOX_READINGS}, "
                f"got {empty_box_reading!r}"
            )
        if gci2_negative_triangle not in ("overlap", "inradius"):
            raise ValueError("gci2_negative_triangle must be 'overlap' or 'inradius'.")
        if isinstance(embed_dim, bool) or not isinstance(embed_dim, int) or embed_dim < 3:
            raise ValueError(f"embed_dim must be an int >= 3, got {embed_dim!r}")
        if not isinstance(objective, str) or objective not in _PRISM_OBJECTIVES:
            raise ValueError(
                f"objective must be one of {_PRISM_OBJECTIVES}, got {objective!r}"
            )
        if isinstance(max_depth, bool) or not isinstance(max_depth, int) or max_depth < 0:
            raise ValueError(f"max_depth must be an int >= 0, got {max_depth!r}")
        if not isinstance(structural_definitions, bool):
            raise ValueError(
                f"structural_definitions must be a bool, got {structural_definitions!r}"
            )
        if (
            isinstance(existence_weight, bool)
            or not isinstance(existence_weight, (int, float))
            or float(existence_weight) < 0.0
        ):
            raise ValueError(
                f"existence_weight must be >= 0, got {existence_weight!r}"
            )
        if (
            isinstance(size_reg, bool)
            or not isinstance(size_reg, (int, float))
            or float(size_reg) < 0.0
        ):
            raise ValueError(f"size_reg must be >= 0, got {size_reg!r}")
        super().__init__()

        self.nb_ont_classes = nb_ont_classes
        self.nb_rels = nb_rels
        self.nb_inds = nb_inds
        self.embed_dim = embed_dim
        self.margin = margin
        self.objective = objective
        self.max_depth = max_depth
        self.structural_definitions = bool(structural_definitions)
        self.existence_weight = float(existence_weight)
        self.size_reg = float(size_reg)
        self.gci2_negative_triangle = gci2_negative_triangle
        self.empty_box_reading = empty_box_reading
        self.size_reg_without_tables = bool(size_reg_without_tables)
        self._tables_installed = False
        self.definition_table: Optional[DefinitionTable] = None
        self.existence_targets: Optional[ExistenceTargets] = None
        self.top_idx: Optional[int] = None
        self._structural_device: Optional[th.device] = None
        self._def_mask_dev: Optional[th.Tensor] = None
        self._def_x_dev: Optional[th.Tensor] = None
        self._def_y_dev: Optional[th.Tensor] = None
        self._target_classes_dev: Optional[th.Tensor] = None
        self._target_pairs_dev: Optional[th.Tensor] = None
        self._prim_mask_dev: Optional[th.Tensor] = None
        self._has_prim: bool = False

        box_dim = embed_dim - 2
        self.tri_center = self._init_normalized_embedding(nb_ont_classes, 2)
        self.tri_inradius = nn.Embedding(nb_ont_classes, 1)
        nn.init.uniform_(self.tri_inradius.weight, a=0.1, b=1.0)

        self.box_center = self._init_normalized_embedding(nb_ont_classes, box_dim)
        self.box_offset = self._init_normalized_embedding(nb_ont_classes, box_dim)
        self.rel_embed = self._init_normalized_embedding(nb_rels, embed_dim)

        if self.nb_inds is not None and self.nb_inds > 0:
            self.ind_embed = self._init_normalized_embedding(self.nb_inds, embed_dim)
        else:
            self.ind_embed = None

    @staticmethod
    def _init_normalized_embedding(num_entities: int, dim: int) -> nn.Embedding:
        emb = nn.Embedding(num_entities, dim)
        nn.init.uniform_(emb.weight, a=-1.0, b=1.0)
        if num_entities > 0:
            norms = th.linalg.norm(emb.weight.data, dim=1, keepdim=True).clamp_min(1e-12)
            emb.weight.data.div_(norms)
        return emb

    def prism_params(self, idx: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """Return ``(h, box_center, box_offset)`` for class indices ``idx``.

        Guarantees:

        - ``h`` has shape ``(B, 3)`` with ``h.sum(dim=-1) == 3 * |tri_inradius(idx)|``.
        - ``box_center`` and ``box_offset`` have shape ``(B, d - 2)``, with
          ``box_offset >= 0``.

        Non-guarantees:

        - Does not detach the returned tensors from the autograd graph.
        """
        tc = self.tri_center(idx)
        tr = th.abs(self.tri_inradius(idx))
        h = L.supports(tc, tr)
        bc = self.box_center(idx)
        bo = th.abs(self.box_offset(idx))
        return h, bc, bo

    def _sync_structural_device(self, device: th.device) -> None:
        """Stage structural table tensors onto ``device`` once when attached or moved."""
        if self._structural_device == device:
            return
        self._structural_device = device
        table = self.definition_table
        if table is not None:
            self._def_mask_dev = table.def_mask.to(device=device)
            self._def_x_dev = table.def_x.to(device=device)
            self._def_y_dev = table.def_y.to(device=device)
            if table._masked_gci0_keys.device != device:
                table._masked_gci0_keys = table._masked_gci0_keys.to(device=device)
            if table._masked_gci1_keys.device != device:
                table._masked_gci1_keys = table._masked_gci1_keys.to(device=device)
        else:
            self._def_mask_dev = None
            self._def_x_dev = None
            self._def_y_dev = None

        targets = self.existence_targets
        if targets is not None:
            self._target_classes_dev = targets.classes.to(device=device, dtype=th.long)
            self._target_pairs_dev = targets.pairs.to(device=device, dtype=th.long)
        else:
            self._target_classes_dev = None
            self._target_pairs_dev = None

        if self.nb_ont_classes > 0:
            prim_mask = th.ones(self.nb_ont_classes, dtype=th.bool, device=device)
            if self._def_mask_dev is not None:
                prim_mask = prim_mask & (~self._def_mask_dev)
            if self.top_idx is not None and 0 <= self.top_idx < self.nb_ont_classes:
                prim_mask[self.top_idx] = False
            self._prim_mask_dev = prim_mask
            self._has_prim = bool(prim_mask.any().item())
        else:
            self._prim_mask_dev = None
            self._has_prim = False

    def set_structural_tables(
        self,
        table: DefinitionTable,
        targets: ExistenceTargets,
        top_idx: Optional[int] = None,
    ) -> None:
        """Install the compiled :class:`DefinitionTable` and :class:`ExistenceTargets`.

        Guarantees:

        - Stores ``table`` on :attr:`definition_table`, ``targets`` on
          :attr:`existence_targets`, and ``top_idx`` on :attr:`top_idx`, and
          stages device copies of the table and target index tensors on
          ``tri_center.weight.device``.
        - Subsequent calls to :meth:`class_regions`, :meth:`structural_extra_loss`,
          and the GCI loss methods use these tables when ``objective="structural"``.

        Non-guarantees:

        - Does not re-validate table dimensions against ``nb_ont_classes``.

        Parameters
        ----------
        table : DefinitionTable
            Compiled acyclic conjunction definition table.
        targets : ExistenceTargets
            Satisfiable classes and conjunction pairs for existence regularisation.
        top_idx : Optional[int], default None
            Class index of ``owl:Thing`` to exclude from primitive size regularisation.

        Returns
        -------
        None
        """
        self.definition_table = table
        self.existence_targets = targets
        self.top_idx = top_idx
        self._tables_installed = True
        self._structural_device = None
        self._sync_structural_device(self.tri_center.weight.device)

    def class_regions(self, idx: th.Tensor) -> L.PrismRegions:
        """Return materialised :class:`~mowl.baselines.nn.el.prism.losses.PrismRegions` for class indices ``idx``.

        Primitive classes are read from :meth:`prism_params` (``w = |box_offset|``).
        Defined classes in :attr:`definition_table` are evaluated recursively via
        :func:`~mowl.baselines.nn.el.prism.losses.prism_regions_intersection` with signed box
        half-widths ``w = (upper - lower) / 2`` and triangle supports
        ``h = min(h_X, h_Y)``. When no :attr:`definition_table` is set, every class
        is treated as primitive.

        Guarantees:

        - Returns a :class:`~mowl.baselines.nn.el.prism.losses.PrismRegions` on the module
          device with ``h`` of shape ``(B, 3)`` and ``cen, w`` of shape ``(B, d - 2)``.
        - Preserves signed negative half-widths ``w_m < 0`` on defined classes
          whose box intersection is empty along coordinate ``m``.

        Non-guarantees:

        - Assumes :attr:`definition_table` is acyclic (as guaranteed by
          :meth:`DefinitionTable.from_normal_forms`).

        Parameters
        ----------
        idx : torch.Tensor
            1-D integer tensor of class indices of shape ``(B,)``.

        Returns
        -------
        mowl.baselines.nn.el.prism.losses.PrismRegions
            Materialised prism regions for ``idx`` on the module device.
        """
        device = self.tri_center.weight.device
        idx_t = (
            idx
            if (idx.device == device and idx.dtype == th.long)
            else idx.to(device=device, dtype=th.long)
        )
        h, bc, bo = self.prism_params(idx_t)

        table = self.definition_table
        if table is None or table.n_defined == 0 or idx_t.numel() == 0:
            return L.PrismRegions(h=h, cen=bc, w=bo)

        self._sync_structural_device(device)
        assert self._def_mask_dev is not None
        assert self._def_x_dev is not None
        assert self._def_y_dev is not None
        def_mask = self._def_mask_dev[idx_t]
        if not bool(def_mask.any().item()):
            return L.PrismRegions(h=h, cen=bc, w=bo)

        def_pos = th.nonzero(def_mask, as_tuple=False).squeeze(1)
        def_ids = idx_t[def_pos]
        dx = self._def_x_dev[def_ids]
        dy = self._def_y_dev[def_ids]

        rx = self.class_regions(dx)
        ry = self.class_regions(dy)
        inter_def = L.prism_regions_intersection(rx, ry)

        h_out = h.clone()
        cen_out = bc.clone()
        w_out = bo.clone()
        h_out[def_pos] = inter_def.h
        cen_out[def_pos] = inter_def.cen
        w_out[def_pos] = inter_def.w
        return L.PrismRegions(h=h_out, cen=cen_out, w=w_out)

    def all_class_regions(self) -> L.PrismRegions:
        """Return :class:`~mowl.baselines.nn.el.prism.losses.PrismRegions` for all classes on CPU without grad.

        Guarantees:

        - Returns a :class:`~mowl.baselines.nn.el.prism.losses.PrismRegions` of shape
          ``(nb_ont_classes, ...)`` on the ``cpu`` device with ``requires_grad == False``.

        Non-guarantees:

        - Detached from the autograd graph and unsuitable for backpropagating loss
          gradients.

        Parameters
        ----------
        None

        Returns
        -------
        mowl.baselines.nn.el.prism.losses.PrismRegions
            Detached CPU prism regions for all ``0 .. nb_ont_classes - 1``.
        """
        device = self.tri_center.weight.device
        with th.no_grad():
            idx = th.arange(self.nb_ont_classes, dtype=th.long, device=device)
            reg = self.class_regions(idx)
            return L.PrismRegions(
                h=reg.h.detach().cpu(),
                cen=reg.cen.detach().cpu(),
                w=reg.w.detach().cpu(),
            )

    def structural_extra_loss(self) -> th.Tensor:
        """Compute existence loss plus primitive size regularisation for ``objective="structural"``.

        Guarantees:

        - Returns a scalar tensor of shape ``()`` on the module device.
        - Excludes defined classes (``definition_table.def_mask``) and ``top_idx``
          (``owl:Thing``) from the primitive size regularisation term
          ``size_reg * mean([|r|, 2|o|])``.

        Non-guarantees:

        - Returns ``0.0`` for the size term before :meth:`set_structural_tables` (``owl:Thing``
          and the defined classes are unknown), unless ``size_reg_without_tables=True``.
        - Returns ``0.0`` for the existence term when :attr:`existence_targets` is
          ``None`` or ``existence_weight == 0``.

        Parameters
        ----------
        None

        Returns
        -------
        torch.Tensor
            Scalar extra training loss combining existence and primitive size terms.
        """
        device = self.tri_center.weight.device
        dtype = self.tri_center.weight.dtype
        self._sync_structural_device(device)
        total = th.zeros((), device=device, dtype=dtype)

        if self.existence_targets is not None and self.existence_weight > 0.0:
            assert self._target_classes_dev is not None
            assert self._target_pairs_dev is not None
            if self._target_classes_dev.numel() > 0:
                reg_c = self.class_regions(self._target_classes_dev)
                class_sizes = L.prism_nonempty_measure(reg_c)
            else:
                class_sizes = th.empty(0, device=device, dtype=dtype)

            if self._target_pairs_dev.shape[0] > 0:
                rc = self.class_regions(self._target_pairs_dev[:, 0])
                rd = self.class_regions(self._target_pairs_dev[:, 1])
                inter = L.prism_regions_intersection(rc, rd)
                pair_sizes = L.prism_nonempty_measure(inter)
            else:
                pair_sizes = th.empty(0, device=device, dtype=dtype)

            total = total + self.existence_weight * existence_loss(
                class_sizes, pair_sizes, self.margin
            )

        size_known = self._tables_installed or self.size_reg_without_tables
        if self.size_reg > 0.0 and self._has_prim and size_known:
            assert self._prim_mask_dev is not None
            r_abs = th.abs(self.tri_inradius.weight[self._prim_mask_dev])
            w_abs = 2.0 * th.abs(self.box_offset.weight[self._prim_mask_dev])
            size_term = th.cat([r_abs, w_abs], dim=-1).mean()
            total = total + self.size_reg * size_term

        return total

    def regularization_loss(self) -> th.Tensor:
        """Return zero regularisation loss (no bump or component regulariser)."""
        return th.tensor(0.0, device=self.tri_center.weight.device)

    def gci0_loss(self, data: th.Tensor, neg: bool = False) -> th.Tensor:
        if self.objective == "structural":
            c = self.class_regions(data[:, 0])
            d = self.class_regions(data[:, 1])
            out = L.gci0_structural(c, d, self.margin)
            if self.definition_table is not None:
                masked = self.definition_table.masked_gci0(data)
                out = th.where(masked, th.zeros_like(out), out)
            return out
        return L.gci0_loss(
            data,
            self.tri_center,
            self.tri_inradius,
            self.box_center,
            self.box_offset,
            self.margin,
            neg=neg,
        )

    def gci0_bot_loss(self, data: th.Tensor, neg: bool = False) -> th.Tensor:
        if self.objective == "structural":
            c = self.class_regions(data[:, 0])
            return L.gci0_bot_structural(c)
        return L.gci0_bot_loss(
            data,
            self.tri_center,
            self.tri_inradius,
            self.margin,
            neg=neg,
        )

    def gci1_loss(self, data: th.Tensor, neg: bool = False) -> th.Tensor:
        if self.objective == "structural":
            c = self.class_regions(data[:, 0])
            d = self.class_regions(data[:, 1])
            e = self.class_regions(data[:, 2])
            masked = (
                self.definition_table.masked_gci1(data)
                if self.definition_table is not None
                else None
            )
            return L.gci1_structural(c, d, e, self.margin, masked=masked)
        return L.gci1_loss(
            data,
            self.tri_center,
            self.tri_inradius,
            self.box_center,
            self.box_offset,
            self.margin,
            neg=neg,
        )

    def gci1_bot_loss(self, data: th.Tensor, neg: bool = False) -> th.Tensor:
        if self.objective == "structural":
            c = self.class_regions(data[:, 0])
            d = self.class_regions(data[:, 1])
            return L.gci1_bot_structural(c, d, self.margin)
        return L.gci1_bot_loss(
            data,
            self.tri_center,
            self.tri_inradius,
            self.box_center,
            self.box_offset,
            self.margin,
            neg=neg,
        )

    def _empty_gate(self, r: L.PrismRegions) -> th.Tensor:
        """Boolean ``(B,)``: the prism is non-empty (detached), so its row counts."""
        return L.prism_nonempty_measure(
            L.PrismRegions(h=r.h.detach(), cen=r.cen.detach(), w=r.w.detach())
        ) > 0.0

    def gci2_loss(self, data: th.Tensor, neg: bool = False) -> th.Tensor:
        if self.objective == "structural":
            c = self.class_regions(data[:, 0])
            d = self.class_regions(data[:, 2])
            rel = self.rel_embed(data[:, 1])
            if self.empty_box_reading == "absolute":
                c_abs = L.PrismRegions(h=c.h, cen=c.cen, w=th.abs(c.w))
                d_abs = L.PrismRegions(h=d.h, cen=d.cen, w=th.abs(d.w))
                if neg:
                    return L.gci2_score_neg_prisms(
                        c_abs, d_abs, rel, self.margin,
                        triangle_negative=self.gci2_negative_triangle)
                return L.gci2_score_prisms(c_abs, d_abs, rel, self.margin)
            if neg:
                # a negative needs a non-empty source and target to be meaningful
                c_abs = L.PrismRegions(h=c.h, cen=c.cen, w=th.abs(c.w))
                d_abs = L.PrismRegions(h=d.h, cen=d.cen, w=th.abs(d.w))
                out = L.gci2_score_neg_prisms(
                    c_abs, d_abs, rel, self.margin,
                    triangle_negative=self.gci2_negative_triangle)
                keep = self._empty_gate(c) & self._empty_gate(d)
                return th.where(keep, out, th.zeros_like(out))
            out = L.gci2_score(c.h, c.cen, c.w, d.h, d.cen, d.w, rel, self.margin, signed=True)
            return th.where(self._empty_gate(c), out, th.zeros_like(out))
        return L.gci2_loss(
            data,
            self.tri_center,
            self.tri_inradius,
            self.box_center,
            self.box_offset,
            self.rel_embed,
            self.margin,
            neg=neg,
            triangle_negative=self.gci2_negative_triangle,
        )

    def gci3_loss(self, data: th.Tensor, neg: bool = False) -> th.Tensor:
        if self.objective == "structural":
            rel = self.rel_embed(data[:, 0])
            c = self.class_regions(data[:, 1])
            d = self.class_regions(data[:, 2])
            if self.empty_box_reading == "absolute":
                c_abs = L.PrismRegions(h=c.h, cen=c.cen, w=th.abs(c.w))
                d_abs = L.PrismRegions(h=d.h, cen=d.cen, w=th.abs(d.w))
                return L.gci3_score_prisms(c_abs, d_abs, rel, self.margin)
            out = L.gci3_score_prisms(c, d, rel, self.margin, signed=True)
            return th.where(self._empty_gate(c), out, th.zeros_like(out))
        return L.gci3_loss(
            data,
            self.tri_center,
            self.tri_inradius,
            self.box_center,
            self.box_offset,
            self.rel_embed,
            self.margin,
            neg=neg,
        )

    def gci3_bot_loss(self, data: th.Tensor, neg: bool = False) -> th.Tensor:
        if self.objective == "structural":
            c = self.class_regions(L.gci3_bot_filler(data))
            if self.empty_box_reading == "absolute":
                c_abs = L.PrismRegions(h=c.h, cen=c.cen, w=th.abs(c.w))
                return L.gci3_bot_score_prisms(c_abs, self.margin)
            out = L.gci3_bot_score_prisms(c, self.margin)
            box_empty = th.amin(c.w.detach(), dim=-1) <= 0.0
            return th.where(box_empty, th.zeros_like(out), out)
        return L.gci3_bot_loss(
            data,
            self.tri_center,
            self.tri_inradius,
            self.margin,
            neg=neg,
        )

    def class_assertion_loss(self, data: th.Tensor, neg: bool = False) -> th.Tensor:
        if self.ind_embed is None:
            raise ValueError(
                "The number of individuals must be specified to use this loss function."
            )
        if self.objective == "structural":
            ind_pts = self.ind_embed(data[:, 0])
            c = self.class_regions(data[:, 1])
            if self.empty_box_reading == "absolute":
                c = L.PrismRegions(h=c.h, cen=c.cen, w=th.abs(c.w))
                return L.class_assertion_score_prisms(ind_pts, c, self.margin)
            return L.class_assertion_score_prisms(ind_pts, c, self.margin, signed=True)
        return L.class_assertion_loss(
            data,
            self.ind_embed,
            self.tri_center,
            self.tri_inradius,
            self.box_center,
            self.box_offset,
            self.margin,
            neg=neg,
        )

    def object_property_assertion_loss(
        self, data: th.Tensor, neg: bool = False
    ) -> th.Tensor:
        if self.ind_embed is None:
            raise ValueError(
                "The number of individuals must be specified to use this loss function."
            )
        return L.object_property_assertion_loss(
            data,
            self.ind_embed,
            self.rel_embed,
            self.margin,
            neg=neg,
            triangle_negative=self.gci2_negative_triangle,
        )
