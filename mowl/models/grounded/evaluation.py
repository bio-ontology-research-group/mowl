"""The evaluation regime of GroundedEL: crisp violation rates and the exact-model check.

Mixed into :class:`mowl.models.GroundedEL`; reads the module and the axioms of the model."""
from __future__ import annotations

import copy
from typing import Any, Dict
from mowl.training.sampled_objective import seeded_generator
from mowl.nn.el.grounded import crisp_membership
from mowl.nn.el.grounded.module import _validate_int_ge
from mowl.nn.el.structural import Regions
from mowl.representation.library.grounded_evaluation import (CrispViolations, ExactDomainCheck,
                                                             all_box_corners)
from mowl.representation.library.grounded_logic import validate_nonneg_weight as _validate_nonneg_weight
from mowl.owlapi.defaults import BOT as BOTTOM, TOP
from mowl.models.grounded.common import _ToleranceCountDict, _short_iri_map

class GroundedEvaluation:
    """Crisp element and region violation rates (:meth:`crisp_report`, the evaluation regime
    :class:`~mowl.representation.library.grounded_evaluation.CrispViolations` over the
    module) and the exact-model check of the finite domain (:meth:`evaluate_model`)."""

    #: The evaluation regime of the crisp report (a part; ``None``: ``CrispViolations``).
    crisp_regime = None
    #: The evaluation regime of the exact-model check (a part; ``None``: ``ExactDomainCheck``).
    exact_regime = None

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
        gen = seeded_generator(self.seed, 9999)
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
        reasoner = None
        if use_reasoner:
            if self.dataset is None or getattr(self.dataset, "ontology", None) is None:
                raise ValueError(
                    "use_reasoner=True requires an OWL dataset; pass use_reasoner=False for "
                    "table-constructed GroundedEL models")
            from mowl.geometric.completion import CanonicalModel
            reasoner = CanonicalModel.from_ontology(self.dataset.ontology)
        gen = seeded_generator(self.seed, 2026)
        regime = self.exact_regime if self.exact_regime is not None else ExactDomainCheck()
        return regime.report(self.module, base, self.training_datasets, gen, n_eval=n_eval,
                             bot_id=self.bot_id, reasoner=reasoner,
                             class_index=self.class_index_dict,
                             named_index=self.named_class_index_dict, top=TOP, bottom=BOTTOM)
