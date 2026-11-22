"""The evaluation regime of GroundedEL: crisp violation rates and the exact-model check.

Mixed into :class:`mowl.models.GroundedEL`; reads the module and the axioms of the model."""
from __future__ import annotations

import copy
from typing import Any, Dict
import torch as th
from mowl.nn.el.grounded import crisp_membership, sample_class_region
from mowl.nn.el.grounded.module import _validate_int_ge
from mowl.nn.el.structural import Regions
from mowl.representation.library.grounded_evaluation import CrispViolations, all_box_corners
from mowl.representation.library.grounded_logic import validate_nonneg_weight as _validate_nonneg_weight
from mowl.owlapi.defaults import BOT as BOTTOM, TOP
from mowl.models.grounded.common import _ToleranceCountDict, _short_iri_map

class GroundedEvaluation:
    """Crisp element and region violation rates (:meth:`crisp_report`, the evaluation regime
    :class:`~mowl.representation.library.grounded_evaluation.CrispViolations` over the
    module) and the exact-model check of the finite domain (:meth:`evaluate_model`)."""

    #: The evaluation regime of the crisp report (a part; ``None``: ``CrispViolations``).
    crisp_regime = None

    def _box_corner_samples(self, cen, w, active, gen, max_corners_per_box=32):
        """The corners (just inside) of the active non-empty boxes ``(1, K, d)``."""
        return all_box_corners(Regions(cen=cen, w=w, active=active), gen,
                               cap=max_corners_per_box)

    def _eval_elem_memberships_chunked(self, eval_pos, mentioned_classes, chunk_size=512):
        """Crisp membership ``(L, M)`` of ``eval_pos`` in ``mentioned_classes``."""
        return crisp_membership(self.module.class_model.depth_table(
            eval_pos, mentioned_classes, chunk_size=chunk_size))

    def crisp_report(self, n_eval: int = 10000, tol: float = 0.0) -> Dict[str, Any]:
        """Reasoner-free crisp violation rates of the axioms (§7): ``{"elem_viol",
        "region_viol", "worst_depth", "n_axioms_violated_by_depth", "n_empty_classes",
        "n_dead_elements"}`` (:class:`~mowl.representation.library.grounded_evaluation.CrispViolations`).

        A sampled point is a counterexample of a region axiom at tolerance ``tol`` if its
        violating depth exceeds ``tol``; ``n_axioms_violated_by_depth`` counts violated
        axioms at ``tol`` in ``{0, 1e-5, 1e-4, 1e-3, 1e-2}``. Draws from a generator seeded
        with ``seed + 9999``, so the report is a function of the parameters."""
        _validate_int_ge("n_eval", n_eval, 1)
        tol_f = _validate_nonneg_weight("tol", tol)
        gen = th.Generator(device="cpu")
        gen.manual_seed(self.seed + 9999)
        regime = copy.copy(self.crisp_regime) if self.crisp_regime is not None else \
            CrispViolations()
        regime.n_eval, regime.tol = int(n_eval), tol_f
        out = regime.report(self.module, self.training_datasets, self.grounding(), gen,
                            _short_iri_map(self.class_index_dict),
                            _short_iri_map(self.object_property_index_dict))
        out["n_axioms_violated_by_depth"] = _ToleranceCountDict(
            out["n_axioms_violated_by_depth"])
        return out

    def evaluate_model(
        self,
        use_reasoner: bool = True,
        n_eval: int = 10000,
        tol: float = 0.0,
    ) -> Dict[str, Any]:
        """Compute §7 evaluation metrics dictionary.

        Lazily imports :class:`~mowl.geometric.completion.CanonicalModel` inside
        this method ONLY when ``use_reasoner=True``.

        Guarantees:

        - Returns a dict with keys ``"elem_viol"``, ``"region_viol"``,
          ``"region_viol_tol1e-4"``, ``"exist"``, ``"dead_ok"``,
          ``"n_empty_classes"``, ``"n_dead_elements"``, ``"n_false_dead"``,
          ``"faithful_named_precision"``, ``"faithful_named_recall"``,
          ``"faithful_named"``, ``"exact"``, and ``"exact_tol1e-4"``.
        - ``exact`` is ``True`` if and only if ``elem_viol == 0.0``,
          ``region_viol == 0.0``, ``exist == 1.0``, and ``dead_ok is True``.
        - ``region_viol_tol1e-4`` and ``exact_tol1e-4`` are always computed at
          tolerance ``1e-4``, independent of ``tol``.

        Non-guarantees:

        - When ``use_reasoner=False``, ``exist`` is evaluated over live classes
          (``dead <= 0.5``) and ``n_false_dead`` and ``faithful_named_*`` are ``None``.

        Parameters
        ----------
        use_reasoner : bool, default True
            Whether to compute ground-truth satisfiability and subsumptions via
            :class:`~mowl.geometric.completion.CanonicalModel`.
        n_eval : int, default 10000
            Sample budget for ``region_viol`` and region inclusion checks.
        tol : float, default 0.0
            Non-negative boundary tolerance :math:`\\ge 0` passed to :meth:`crisp_report`.

        Returns
        -------
        Dict[str, Any]
            §7 evaluation metrics dictionary.
        """
        base = self.crisp_report(n_eval=n_eval, tol=tol)
        elem_viol = base["elem_viol"]
        region_viol = base["region_viol"]
        n_empty_classes = int(base["n_empty_classes"])
        n_dead_elements = int(base["n_dead_elements"])
        td = self.training_datasets
        total_region_axioms = sum(
            len(td[k])
            for k in (
                "gci0",
                "gci0_bot",
                "gci1",
                "gci1_bot",
                "gci2",
                "gci3",
                "gci3_bot",
                "role_inclusion",
                "role_chain",
            )
            if k in td and len(td[k]) > 0
        )
        region_viol_tol1e4 = (
            float(base["n_axioms_violated_by_depth"]["1e-4"] / total_region_axioms)
            if total_region_axioms > 0
            else 0.0
        )

        mod = self.module
        device = self.device
        gen = th.Generator(device="cpu")
        gen.manual_seed(self.seed + 2026)

        with th.no_grad():
            pos_all = mod.positions()
            dead_cls = (mod.dead()[: self.nb_classes] > 0.5).cpu()
            if self.bot_id is not None:
                dead_cls[self.bot_id] = True

            self_inside = th.zeros((self.nb_classes,), dtype=th.bool)
            for c_start in range(0, self.nb_classes, 8192):
                c_end = min(c_start + 8192, self.nb_classes)
                c_ids = th.arange(c_start, c_end, dtype=th.long, device=device)
                dep_c = mod.depth(pos_all[c_ids], c_ids, crisp=True).cpu()
                self_inside[c_start:c_end] = (dep_c > 0.0) & (~dead_cls[c_start:c_end])

            if not use_reasoner:
                live_cls = ~dead_cls
                exist_val = (
                    float(self_inside[live_cls].to(th.float32).mean().item())
                    if bool(live_cls.any().item())
                    else 1.0
                )
                dead_ok = True
                exact = bool(
                    elem_viol == 0.0
                    and region_viol == 0.0
                    and exist_val == 1.0
                    and dead_ok
                )
                exact_tol1e4 = bool(
                    elem_viol == 0.0
                    and region_viol_tol1e4 == 0.0
                    and exist_val == 1.0
                    and dead_ok
                )
                return {
                    "elem_viol": elem_viol,
                    "region_viol": region_viol,
                    "region_viol_tol1e-4": region_viol_tol1e4,
                    "exist": exist_val,
                    "dead_ok": dead_ok,
                    "n_empty_classes": n_empty_classes,
                    "n_dead_elements": n_dead_elements,
                    "n_false_dead": None,
                    "faithful_named_precision": None,
                    "faithful_named_recall": None,
                    "faithful_named": (None, None),
                    "exact": exact,
                    "exact_tol1e-4": exact_tol1e4,
                }

            if self.dataset is None or getattr(self.dataset, "ontology", None) is None:
                raise ValueError(
                    "use_reasoner=True requires an OWL dataset; pass use_reasoner=False for table-constructed GroundedEL models"
                )

            from mowl.geometric.completion import CanonicalModel

            cm = CanonicalModel.from_ontology(self.dataset.ontology)

            sat_mask = th.zeros((self.nb_classes,), dtype=th.bool)
            for c_iri, c_idx in self.class_index_dict.items():
                if c_iri == BOTTOM:
                    sat_mask[c_idx] = False
                elif c_iri == TOP:
                    sat_mask[c_idx] = bool(cm.is_satisfiable(TOP))
                else:
                    sat_mask[c_idx] = bool(cm.is_satisfiable(c_iri))

            unsat_mask = ~sat_mask
            dead_ok = bool(th.equal(dead_cls, unsat_mask))
            n_false_dead = int((dead_cls & sat_mask).sum().item())

            if bool(sat_mask.any().item()):
                exist_val = float(
                    self_inside[sat_mask].to(dtype=th.float32).mean().item()
                )
            else:
                exist_val = 1.0

            # Named class region inclusion faithfulness (M2)
            named_items = [
                (iri, idx)
                for iri, idx in self.named_class_index_dict.items()
                if iri not in (TOP, BOTTOM) and bool(sat_mask[idx].item())
            ]
            if len(named_items) > 128:
                named_items = named_items[:128]

            n_named = len(named_items)
            if n_named >= 2:
                n_pts_per_cls = max(16, min(256, n_eval // max(n_named, 1)))
                all_regs = mod.class_regions()
                cls_points: Dict[int, th.Tensor] = {}
                for _, c_idx in named_items:
                    s_reg = sample_class_region(
                        mod, c_idx, generator=gen, n_samples=n_pts_per_cls
                    )
                    corners = self._box_corner_samples(
                        all_regs.cen[c_idx : c_idx + 1],
                        all_regs.w[c_idx : c_idx + 1],
                        all_regs.active[c_idx : c_idx + 1],
                        gen,
                    )
                    pts_parts = [pos_all[c_idx : c_idx + 1]]
                    if bool(s_reg.valid.any().item()):
                        pts_parts.append(s_reg.points[s_reg.valid])
                    if corners.numel() > 0:
                        pts_parts.append(corners)
                    cand_pts = th.cat(pts_parts, dim=0)
                    in_c = mod.depth(cand_pts, c_idx, crisp=True) > 0.0
                    cls_points[c_idx] = cand_pts[in_c]

                tp = 0
                fp = 0
                tn = 0
                fn = 0
                for a_iri, a_idx in named_items:
                    pts_a = cls_points[a_idx]
                    for b_iri, b_idx in named_items:
                        if a_idx == b_idx:
                            continue
                        gt_entailed = bool(
                            cm.extension(a_iri).issubset(cm.extension(b_iri))
                        )
                        if pts_a.shape[0] == 0:
                            pred_inc = True
                        else:
                            pred_inc = bool(
                                (mod.depth(pts_a, b_idx, crisp=True) > 0.0)
                                .all()
                                .item()
                            )
                        if gt_entailed and pred_inc:
                            tp += 1
                        elif not gt_entailed and pred_inc:
                            fp += 1
                        elif not gt_entailed and not pred_inc:
                            tn += 1
                        else:
                            fn += 1

                # Precision/recall of non-entailed pairs being separated & entailed pairs included
                precision = (
                    float(tn / (tn + fn))
                    if (tn + fn) > 0
                    else (1.0 if fp == 0 else 0.0)
                )
                recall = (
                    float(tn / (tn + fp))
                    if (tn + fp) > 0
                    else (1.0 if fn == 0 else 0.0)
                )
            else:
                precision = 1.0
                recall = 1.0

            exact = bool(
                elem_viol == 0.0
                and region_viol == 0.0
                and exist_val == 1.0
                and dead_ok
            )
            exact_tol1e4 = bool(
                elem_viol == 0.0
                and region_viol_tol1e4 == 0.0
                and exist_val == 1.0
                and dead_ok
            )
            return {
                "elem_viol": elem_viol,
                "region_viol": region_viol,
                "region_viol_tol1e-4": region_viol_tol1e4,
                "exist": exist_val,
                "dead_ok": dead_ok,
                "n_empty_classes": n_empty_classes,
                "n_dead_elements": n_dead_elements,
                "n_false_dead": n_false_dead,
                "faithful_named_precision": precision,
                "faithful_named_recall": recall,
                "faithful_named": (precision, recall),
                "exact": exact,
                "exact_tol1e-4": exact_tol1e4,
            }
