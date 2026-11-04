"""GroundedEL model: element-grounded EL++ embedding model, training loop, and evaluation.

Implements §3, §4, §5, §7, and §8 of the GroundedEL design:

1. **Syntactic domain construction (:math:`\\Delta`):** Builds one element per
   class in :attr:`class_index_dict` (including normaliser auxiliary classes and
   :math:`\\top, \\bot` if present) followed by one element per individual in
   :attr:`individual_index_dict`.
2. **Reasoner-free training loop (:meth:`train`):** Each epoch anneals
   :math:`\\tau` geometrically from ``tau0`` to ``tau1``, samples mini-batches
   per non-empty axiom type together with candidate elements (§3) and region
   points (§4), sums :math:`L_{\\text{fact}} + L_{\\text{axioms}} + \\lambda_{\\text{reg}} L_{\\text{region}} + L_{\\text{minimality}}`,
   updates parameters with Adam, and optionally performs reasoner-free idle
   component re-seeding (§5).
3. **Evaluation (:meth:`evaluate_model`, :meth:`crisp_report`):** Computes crisp
   element and region violation rates, and (when ``use_reasoner=True`` inside
   :meth:`evaluate_model` only) existence, dead-element accuracy, named-class
   faithfulness precision/recall, and exact-model status.

Guarantees:
- **Reasoner-free training path (R1):** Neither :class:`GroundedEL` construction
  nor :meth:`train` nor :meth:`crisp_report` imports or calls any reasoner or
  semantic completion routine. :class:`~mowl.geometric.completion.CanonicalModel`
  is imported lazily exclusively inside :meth:`evaluate_model` when
  ``use_reasoner=True``.
- **Exact-limit initialisation (R2):** With ``init="elements"`` and caller-supplied
  least-model ``init_labels``, :meth:`evaluate_model` reports ``exact=True``
  (``elem_viol == 0``, ``region_viol == 0``, ``exist == 1``, ``dead_ok == True``)
  at zero training epochs.
- **Monotone capacity (R3):** Expanding module components via
  :meth:`~mowl.nn.GroundedELModule.expand_components` preserves all element
  losses and deterministic evaluation scores.

Non-guarantees:
- Gradient-based optimisation from random initialisation is non-convex and is
  not guaranteed to reach an exact zero-violation model on every random seed or
  at insufficient component capacity :math:`K`.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from mowl.base_models.elmodel import EmbeddingELModel
from mowl.nn.el.grounded import (
    GroundedELModule,
    class_assertion,
    crisp_membership,
    fact,
    gci0,
    gci0_bot,
    gci1,
    gci1_bot,
    gci2,
    gci3,
    gci3_bot,
    minimality,
    role_assertion,
    role_chain,
    role_inclusion,
    sample_class_region,
    sample_elements,
    sample_filler_elements,
    sample_pair_region,
    sample_role_region,
)
from mowl.nn.el.grounded.module import (
    _DEAD_LOGIT,
    _INACTIVE_RAW_WIDTH,
    _LIVE_LOGIT,
    _inv_softplus,
    _validate_int_ge,
    _validate_positive_float,
)
from mowl.nn.el.grounded.losses import _validate_nonneg_weight
from mowl.nn.el.structural import Regions, inside_measure
from mowl.owlapi.defaults import BOT as BOTTOM, TOP

logger = logging.getLogger(__name__)

__all__ = ["GroundedEL"]


class GroundedEL(EmbeddingELModel):
    """Element-grounded multi-box and product-role :math:`\\mathcal{EL}^{++}` embedding model.

    Guarantees:
    - **Reasoner-free training (R1):** :meth:`__init__`, :meth:`init_module`,
      :meth:`train`, :meth:`compute_batch_losses`, :meth:`reseed_idle`, and
      :meth:`crisp_report` never import or invoke any reasoner or semantic
      completion algorithm.
    - **Exact-limit realisation (R2):** Constructed with ``init="elements"`` and
      valid least-model ``init_labels``, achieves ``elem_viol == 0.0`` and
      ``region_viol == 0.0`` with zero training epochs.
    - **Monotone capacity (R3):** Calling ``model.module.expand_components(2 * K)``
      leaves element hinge losses bitwise unchanged.

    Non-guarantees:
    - Does not guarantee global convergence to ``exact=True`` from arbitrary
      random initialisations at finite epoch budgets.
    - When ``use_reasoner=False`` in :meth:`evaluate_model`, reasoner-dependent
      metrics (``exist``, ``dead_ok``, ``faithful_named_precision``,
      ``faithful_named_recall``, ``exact``) are computed syntactically or omitted
      in :meth:`crisp_report`.

    Parameters
    ----------
    dataset : mowl.datasets.Dataset
        mOWL dataset containing the ontology.
    embed_dim : int, default 50
        Euclidean embedding dimension :math:`d \\ge 1`.
    n_components : int, default 2
        Number of box components :math:`K \\ge 1` per class.
    n_role_components : int, default 2
        Number of product-box components :math:`K_r \\ge 1` per role.
    epochs : int, default 1000
        Default number of training epochs when ``epochs`` is not passed to
        :meth:`train`.
    learning_rate : float, default 0.01
        Adam learning rate :math:`> 0`.
    batch_size : int, default 4096
        Mini-batch size :math:`\\ge 1` per axiom type.
    tau0 : float, default 1.0
        Initial sigmoid temperature :math:`\\tau_0 > 0`.
    tau1 : float, default 0.01
        Final sigmoid temperature :math:`\\tau_1 > 0`.
    eps_min : float, default 1e-2
        Minimality weight :math:`\\varepsilon \\ge 0` on random element-class
        memberships.
    eps_role : float, default 1e-2
        Minimality weight :math:`\\varepsilon_r \\ge 0` on random role triples.
    eps_dead : float, default 1e-2
        Minimality weight :math:`\\varepsilon_\\delta \\ge 0` on element dead scores.
    lambda_reg : float, default 1.0
        Weight :math:`\\lambda_{\\text{reg}} \\ge 0` on region soundness hinges (§4).
    n_region_samples : int, default 4
        Number of region points sampled per axiom in each step :math:`\\ge 0`.
    n_adv : int, default 0
        Number of signed-gradient ascent steps :math:`\\ge 0` for adversarial
        region sampling.
    n_random_elements : int, default 4
        Number of uniform random elements :math:`\\ge 0` sampled per axiom row.
    agg : {"max", "lse"}, default "max"
        Component depth aggregation mode.
    reseed_every : int, default 0
        Epoch interval for reasoner-free idle component re-seeding (``0`` disables
        periodic re-seeding).
    device : str, default "cpu"
        PyTorch device string.
    seed : int, default 0
        Random seed for parameter initialisation, mini-batching, and sampling.
    init : {"random", "elements"}, default "random"
        Initialisation mode: ``"random"`` or ``"elements"`` (which applies
        :meth:`~mowl.nn.GroundedELModule.exact_limit_init` using ``init_labels``).
    init_labels : Optional[Any], default None
        Caller-supplied least-model labels required when ``init="elements"``.
        May be a 2-D tensor/array of shape ``(nb_elements, nb_classes)``, a
        ``(labels, edges)`` tuple, or a dict ``{"labels": ..., "edges": ...}``.
    init_edges : Optional[Any], default None
        Optional caller-supplied role edges for ``init="elements"``. When omitted,
        syntactic edges are derived from ``init_labels`` and the normalised role
        axioms (`gci2`, `object_property_assertion`, `role_inclusion`, `role_chain`).
    init_delta : float, default 0.05
        Positive box half-width :math:`\\delta > 0` for ``init="elements"``.
    log_every : int, default 100
        Epoch interval for logging per-term losses in :meth:`train` (``0`` disables).
    agg_tau : float, default 0.1
        Temperature :math:`\\tau_{\\text{agg}} > 0` when ``agg="lse"``.
    dead_init_logit : float, default -4.0
        Initial logit for element deadness :math:`\\ell_x`.
    element_features : Optional[torch.Tensor], default None
        Optional feature matrix for data elements (§6 DeepGO-SE hook).
    encoder : Optional[torch.nn.Module], default None
        Optional encoder network for data elements (§6 DeepGO-SE hook).
    model_filepath : Optional[str], default None
        Optional path for saving model checkpoints.
    load_normalized : bool, default False
        Whether the dataset ontology is already normalised.
    load_role_axioms : bool, default True
        Whether to load ``role_inclusion`` and ``role_chain`` datasets.
    """

    def __init__(
        self,
        dataset,
        embed_dim: int = 50,
        n_components: int = 2,
        n_role_components: int = 2,
        epochs: int = 1000,
        learning_rate: float = 0.01,
        batch_size: int = 4096,
        tau0: float = 1.0,
        tau1: float = 0.01,
        eps_min: float = 1e-2,
        eps_role: float = 1e-2,
        eps_dead: float = 1e-2,
        lambda_reg: float = 1.0,
        n_region_samples: int = 4,
        n_adv: int = 0,
        n_random_elements: int = 4,
        agg: str = "max",
        reseed_every: int = 0,
        device: str = "cpu",
        seed: int = 0,
        init: str = "random",
        init_labels: Optional[Any] = None,
        init_edges: Optional[Any] = None,
        init_delta: float = 0.05,
        log_every: int = 100,
        agg_tau: float = 0.1,
        dead_init_logit: float = -4.0,
        element_features: Optional[th.Tensor] = None,
        encoder: Optional[nn.Module] = None,
        model_filepath: Optional[str] = None,
        load_normalized: bool = False,
        load_role_axioms: bool = True,
    ) -> None:
        _validate_int_ge("embed_dim", embed_dim, 1)
        _validate_int_ge("n_components", n_components, 1)
        _validate_int_ge("n_role_components", n_role_components, 1)
        _validate_int_ge("epochs", epochs, 0)
        _validate_positive_float("learning_rate", learning_rate)
        _validate_int_ge("batch_size", batch_size, 1)
        tau0_f = _validate_positive_float("tau0", tau0)
        tau1_f = _validate_positive_float("tau1", tau1)
        eps_min_f = _validate_nonneg_weight("eps_min", eps_min)
        eps_role_f = _validate_nonneg_weight("eps_role", eps_role)
        eps_dead_f = _validate_nonneg_weight("eps_dead", eps_dead)
        lambda_reg_f = _validate_nonneg_weight("lambda_reg", lambda_reg)
        _validate_int_ge("n_region_samples", n_region_samples, 0)
        _validate_int_ge("n_adv", n_adv, 0)
        _validate_int_ge("n_random_elements", n_random_elements, 0)
        if not isinstance(agg, str) or agg not in ("max", "lse"):
            raise ValueError(f"agg must be 'max' or 'lse', got {agg!r}")
        _validate_int_ge("reseed_every", reseed_every, 0)
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f"seed must be an int, got {seed!r}")
        if not isinstance(init, str) or init not in ("random", "elements"):
            raise ValueError(f"init must be 'random' or 'elements', got {init!r}")
        if init == "elements" and init_labels is None:
            raise ValueError("init='elements' requires init_labels to be provided")
        init_delta_f = _validate_positive_float("init_delta", init_delta)
        _validate_int_ge("log_every", log_every, 0)
        agg_tau_f = _validate_positive_float("agg_tau", agg_tau)

        super().__init__(
            dataset,
            embed_dim=embed_dim,
            batch_size=batch_size,
            extended=True,
            model_filepath=model_filepath,
            load_normalized=load_normalized,
            device=device,
            learning_rate=learning_rate,
            neg_sampling_gcis=[],
            load_role_axioms=load_role_axioms,
        )

        self.n_components = int(n_components)
        self.n_role_components = int(n_role_components)
        self.epochs = int(epochs)
        self.tau0 = tau0_f
        self.tau1 = tau1_f
        self.eps_min = eps_min_f
        self.eps_role = eps_role_f
        self.eps_dead = eps_dead_f
        self.lambda_reg = lambda_reg_f
        self.n_region_samples = int(n_region_samples)
        self.n_adv = int(n_adv)
        self.n_random_elements = int(n_random_elements)
        self.agg = agg
        self.agg_tau = agg_tau_f
        self.reseed_every = int(reseed_every)
        self.seed = int(seed)
        self.init = init
        self.init_labels = init_labels
        self.init_edges = init_edges
        self.init_delta = init_delta_f
        self.log_every = int(log_every)
        self.dead_init_logit = float(dead_init_logit)
        self.element_features = element_features
        self.encoder = encoder

        self.nb_classes = len(self.class_index_dict)
        self.nb_inds = len(self.individual_index_dict)
        self.nb_elements = self.nb_classes + self.nb_inds
        self.nb_rels = len(self.object_property_index_dict)
        self.top_id: Optional[int] = self.class_index_dict.get(TOP, None)
        self.bot_id: Optional[int] = self.class_index_dict.get(BOTTOM, None)

        self.loss_history: List[Dict[str, float]] = []
        self._filler_by_role: Optional[Dict[int, th.Tensor]] = None
        self._lhs_by_role: Optional[Dict[int, th.Tensor]] = None

        self.init_module()

    def init_module(self) -> None:
        """Initialise :class:`~mowl.nn.GroundedELModule` and apply ``init="elements"`` when configured.

        Guarantees:
        - Seeds PyTorch RNG with ``self.seed`` prior to module construction.
        - Fixes ``dead_logits[bot_id] = +100.0`` and marks ``bot_id`` components
          inactive when :math:`\\bot` is present in :attr:`class_index_dict`.
        - When ``init == "elements"``, expands ``n_components`` to at least
          ``nb_elements`` and calls :meth:`~mowl.nn.GroundedELModule.exact_limit_init`.

        Non-guarantees:
        - Mutates global ``torch.manual_seed(self.seed)`` during module
          construction for deterministic parameter initialisation.
        """
        th.manual_seed(self.seed)
        eff_k = (
            max(self.n_components, self.nb_elements)
            if self.init == "elements"
            else self.n_components
        )
        self.n_components = eff_k
        self.module = GroundedELModule(
            nb_elements=self.nb_elements,
            nb_classes=self.nb_classes,
            nb_rels=self.nb_rels,
            embed_dim=self.embed_dim,
            n_components=self.n_components,
            n_role_components=self.n_role_components,
            tau=self.tau0,
            agg=self.agg,
            agg_tau=self.agg_tau,
            dead_init_logit=self.dead_init_logit,
            element_features=self.element_features,
            encoder=self.encoder,
            top_id=self.top_id,
            bot_id=self.bot_id,
        ).to(self.device)

        with th.no_grad():
            if self.bot_id is not None:
                self.module.dead_logits[self.bot_id] = _DEAD_LOGIT
                self.module.class_active[self.bot_id, :] = False
                self.module.class_raw_widths[self.bot_id, :] = _INACTIVE_RAW_WIDTH

        if self.init == "elements":
            labels_t, edges_list = self._resolve_init_labels_and_edges()
            self.module.exact_limit_init(
                labels_t,
                edges=edges_list,
                delta=self.init_delta,
                seed=self.seed,
            )
            self.n_components = self.module.n_components
            self.n_role_components = self.module.n_role_components

    def _resolve_init_labels_and_edges(
        self,
    ) -> Tuple[th.Tensor, List[Tuple[int, int, int]]]:
        """Convert caller-supplied ``init_labels`` (and optional ``init_edges``) into ``(labels_tensor, edges_list)``."""
        raw_labels = self.init_labels
        raw_edges = self.init_edges

        if isinstance(raw_labels, tuple) and len(raw_labels) == 2:
            raw_labels, raw_edges = raw_labels[0], raw_labels[1]
        elif isinstance(raw_labels, Mapping) and "labels" in raw_labels:
            if raw_edges is None and "edges" in raw_labels:
                raw_edges = raw_labels["edges"]
            raw_labels = raw_labels["labels"]

        if isinstance(raw_labels, (th.Tensor, np.ndarray)):
            labels_t = th.as_tensor(raw_labels, dtype=th.bool, device=self.device).clone()
            if labels_t.shape != (self.nb_elements, self.nb_classes):
                raise ValueError(
                    f"init_labels tensor must have shape ({self.nb_elements}, {self.nb_classes}), "
                    f"got {tuple(labels_t.shape)}"
                )
        elif isinstance(raw_labels, Mapping):
            # Support IRI-keyed or index-keyed dictionary mapping class -> elements or element -> classes
            labels_t = th.zeros(
                (self.nb_elements, self.nb_classes), dtype=th.bool, device=self.device
            )
            elem_iri_to_idx: Dict[str, int] = dict(self.class_index_dict)
            for ind_iri, ind_idx in self.individual_index_dict.items():
                elem_iri_to_idx[ind_iri] = self.nb_classes + int(ind_idx)

            for k_key, vals in raw_labels.items():
                if isinstance(k_key, str) and k_key in self.class_index_dict:
                    c_idx = self.class_index_dict[k_key]
                    for v in vals:
                        e_idx = elem_iri_to_idx[v] if isinstance(v, str) else int(v)
                        labels_t[e_idx, c_idx] = True
                else:
                    e_idx = elem_iri_to_idx[k_key] if isinstance(k_key, str) else int(k_key)
                    for v in vals:
                        c_idx = self.class_index_dict[v] if isinstance(v, str) else int(v)
                        labels_t[e_idx, c_idx] = True
        else:
            raise TypeError(
                f"Unsupported init_labels type {type(raw_labels).__name__}; expected Tensor, ndarray, tuple, or Mapping"
            )

        if self.bot_id is not None:
            labels_t[:, self.bot_id] = False
            labels_t[self.bot_id, :] = False
        if self.top_id is not None:
            # Ensure live elements belong to TOP, and x_TOP belongs to TOP when consistent
            live_mask = th.zeros((self.nb_elements,), dtype=th.bool, device=self.device)
            cls_arange = th.arange(self.nb_classes, device=self.device)
            live_mask[: self.nb_classes] = labels_t[cls_arange, cls_arange]
            if self.nb_inds > 0:
                live_mask[self.nb_classes :] = labels_t[self.nb_classes :].any(dim=1)
            if bool(live_mask.any().item()) and not bool(labels_t[self.top_id, self.top_id].item()):
                labels_t[self.top_id, self.top_id] = True
                live_mask[self.top_id] = True
            labels_t[live_mask, self.top_id] = True

        if raw_edges is not None:
            edges_list: List[Tuple[int, int, int]] = []
            if isinstance(raw_edges, Mapping):
                for r_key, pairs in raw_edges.items():
                    r_idx = (
                        self.object_property_index_dict[r_key]
                        if isinstance(r_key, str)
                        else int(r_key)
                    )
                    for u, v in pairs:
                        edges_list.append((int(u), int(v), int(r_idx)))
            elif isinstance(raw_edges, (th.Tensor, np.ndarray)):
                arr = th.as_tensor(raw_edges, dtype=th.long).cpu()
                if arr.numel() > 0:
                    for row in arr.tolist():
                        edges_list.append((int(row[0]), int(row[1]), int(row[2])))
            else:
                for u, v, r in raw_edges:
                    r_idx = (
                        self.object_property_index_dict[r]
                        if isinstance(r, str)
                        else int(r)
                    )
                    edges_list.append((int(u), int(v), int(r_idx)))
            return labels_t, edges_list

        # Derive syntactic edges from labels_t and normalised training datasets (CR3, CR5, CR6, ABox)
        edges_set = set()
        td = self.training_datasets
        cls_arange = th.arange(self.nb_classes, device=self.device)
        live_elem = th.zeros((self.nb_elements,), dtype=th.bool, device=self.device)
        live_elem[: self.nb_classes] = labels_t[cls_arange, cls_arange]
        if self.nb_inds > 0:
            live_elem[self.nb_classes :] = labels_t[self.nb_classes :].any(dim=1)

        if "object_property_assertion" in td and len(td["object_property_assertion"]) > 0:
            for row in td["object_property_assertion"][:].detach().cpu().tolist():
                ind_a, r_id, ind_b = int(row[0]), int(row[1]), int(row[2])
                edges_set.add((self.nb_classes + ind_a, self.nb_classes + ind_b, r_id))

        if "gci2" in td and len(td["gci2"]) > 0:
            for row in td["gci2"][:].detach().cpu().tolist():
                a_id, r_id, b_id = int(row[0]), int(row[1]), int(row[2])
                if not bool(live_elem[b_id].item()):
                    continue
                xs = th.nonzero(labels_t[:, a_id] & live_elem, as_tuple=False).view(-1).cpu().tolist()
                for x_id in xs:
                    edges_set.add((int(x_id), b_id, r_id))

        r_inc: List[Tuple[int, int]] = []
        if "role_inclusion" in td and len(td["role_inclusion"]) > 0:
            r_inc = [
                (int(row[0]), int(row[1]))
                for row in td["role_inclusion"][:].detach().cpu().tolist()
            ]
        r_chain: List[Tuple[int, int, int]] = []
        if "role_chain" in td and len(td["role_chain"]) > 0:
            r_chain = [
                (int(row[0]), int(row[1]), int(row[2]))
                for row in td["role_chain"][:].detach().cpu().tolist()
            ]

        if r_inc or r_chain:
            changed = True
            while changed:
                changed = False
                current = list(edges_set)
                for u, v, r_sub in current:
                    for sub_id, sup_id in r_inc:
                        if r_sub == sub_id and (u, v, sup_id) not in edges_set:
                            edges_set.add((u, v, sup_id))
                            changed = True
                if r_chain:
                    by_role: Dict[int, List[Tuple[int, int]]] = {}
                    for u, v, r_id in edges_set:
                        by_role.setdefault(r_id, []).append((u, v))
                    for r1, r2, sup_id in r_chain:
                        for u, mid1 in by_role.get(r1, ()):
                            for mid2, w_el in by_role.get(r2, ()):
                                if mid1 == mid2 and (u, w_el, sup_id) not in edges_set:
                                    edges_set.add((u, w_el, sup_id))
                                    changed = True

        return labels_t, sorted(edges_set)

    def _ensure_role_tables(self) -> None:
        """Build cached per-role filler and LHS element index tables from ``training_datasets``."""
        if self._filler_by_role is not None and self._lhs_by_role is not None:
            return
        td = self.training_datasets
        fillers: Dict[int, List[int]] = {r: [] for r in range(self.nb_rels)}
        lhs_map: Dict[int, List[int]] = {r: [] for r in range(self.nb_rels)}

        if "gci2" in td and len(td["gci2"]) > 0:
            for row in td["gci2"][:].detach().cpu().tolist():
                a_id, r_id, b_id = int(row[0]), int(row[1]), int(row[2])
                if b_id not in fillers[r_id]:
                    fillers[r_id].append(b_id)
                if a_id not in lhs_map[r_id]:
                    lhs_map[r_id].append(a_id)

        if "object_property_assertion" in td and len(td["object_property_assertion"]) > 0:
            for row in td["object_property_assertion"][:].detach().cpu().tolist():
                ind_a, r_id, ind_b = int(row[0]), int(row[1]), int(row[2])
                el_a = self.nb_classes + ind_a
                el_b = self.nb_classes + ind_b
                if el_b not in fillers[r_id]:
                    fillers[r_id].append(el_b)
                if el_a not in lhs_map[r_id]:
                    lhs_map[r_id].append(el_a)

        self._filler_by_role = {
            r: th.tensor(vals, dtype=th.long, device=self.device)
            for r, vals in fillers.items()
        }
        self._lhs_by_role = {
            r: th.tensor(vals, dtype=th.long, device=self.device)
            for r, vals in lhs_map.items()
        }

    def _sample_rows(self, data: th.Tensor, gen: th.Generator) -> th.Tensor:
        """Sample up to ``self.batch_size`` rows from ``data`` using ``gen``."""
        n = int(data.shape[0])
        if n <= self.batch_size:
            return data.to(device=self.device, dtype=th.long)
        idx = th.randint(
            0, n, (self.batch_size,), dtype=th.long, device=gen.device, generator=gen
        ).to(device=data.device)
        return data[idx].to(device=self.device, dtype=th.long)

    def _batch_candidate_elements(
        self,
        batch_elems: th.Tensor,
        gen: th.Generator,
    ) -> th.Tensor:
        """Return candidate batch elements (all elements when ``nb_elements <= 32``, else subsampled)."""
        if self.nb_elements <= 32:
            return th.arange(self.nb_elements, dtype=th.long, device=self.device)
        uniq = th.unique(batch_elems)
        max_b = max(self.n_random_elements, 8)
        if uniq.numel() <= max_b:
            return uniq
        perm = th.randperm(uniq.numel(), device=gen.device, generator=gen)[
            :max_b
        ].to(device=uniq.device)
        return uniq[perm]

    def compute_batch_losses(
        self,
        generator: Optional[th.Generator] = None,
    ) -> Dict[str, th.Tensor]:
        """Compute one step of per-term GroundedEL losses (§3 and §4).

        Guarantees:
        - Returns a dictionary containing ``"fact"``, ``"axioms"``, ``"region"``,
          ``"minimality"``, and ``"total"`` scalar tensors on ``self.device``.
        - Uses only ``generator`` (or a freshly seeded generator from ``self.seed``
          when ``None``) for all random element and region draws.

        Non-guarantees:
        - When ``n_region_samples == 0`` or ``lambda_reg == 0``, ``"region"`` is
          exact ``0.0``.

        Parameters
        ----------
        generator : Optional[torch.Generator], default None
            Explicit PyTorch RNG on CPU.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary of scalar loss tensors.
        """
        if generator is None:
            generator = th.Generator(device="cpu")
            generator.manual_seed(self.seed)
        self._ensure_role_tables()

        mod = self.module
        device = self.device
        dtype = mod.class_centers.dtype
        td = self.training_datasets

        pos_all = mod.positions()  # (nb_elements, d)
        dead_raw = mod.dead()
        if self.bot_id is not None:
            dead_all = dead_raw.clone()
            dead_all[self.bot_id] = 1.0
        else:
            dead_all = dead_raw
        # Silencer gate for non-bottom rules so satisfiable elements are not killed to evade box motion
        dead_gate = dead_all.detach()

        # 1. Fact (existence) loss on class elements
        live_cls_list = [
            c for c in range(self.nb_classes) if c != self.bot_id
        ]
        if live_cls_list:
            cls_ids = th.tensor(live_cls_list, dtype=th.long, device=device)
            if cls_ids.numel() > self.batch_size:
                cls_ids = self._sample_rows(cls_ids.unsqueeze(1), generator).squeeze(1)
            m_self = mod.membership(pos_all[cls_ids], cls_ids)
            loss_fact = fact(m_self, dead=dead_gate[cls_ids]).mean()
        else:
            loss_fact = th.zeros((), dtype=dtype, device=device)

        axiom_terms: List[th.Tensor] = []
        region_terms: List[th.Tensor] = []
        do_region = self.lambda_reg > 0.0 and self.n_region_samples > 0

        # 2. GCI0: A <= B
        if "gci0" in td and len(td["gci0"]) > 0:
            rows = self._sample_rows(td["gci0"][:], generator)
            bsz = int(rows.shape[0])
            cand = self._batch_candidate_elements(rows.reshape(-1), generator)
            x_mat = sample_elements(
                rows[:, 0], self.nb_elements, self.n_random_elements, generator, batch_elements=cand
            )
            s_cnt = int(x_mat.shape[1])
            x_flat = x_mat.reshape(-1)
            a_flat = rows[:, 0].unsqueeze(1).expand(bsz, s_cnt).reshape(-1)
            b_flat = rows[:, 1].unsqueeze(1).expand(bsz, s_cnt).reshape(-1)
            px = pos_all[x_flat]
            axiom_terms.append(
                gci0(
                    mod.membership(px, a_flat),
                    mod.membership(px, b_flat),
                    dead=dead_gate[x_flat],
                ).mean()
            )
            if do_region:
                a_reg = rows[:, 0].repeat(self.n_region_samples)
                b_reg = rows[:, 1].repeat(self.n_region_samples)
                h_fn = (
                    (lambda z: gci0(mod.membership(z, a_reg), mod.membership(z, b_reg), dead=None))
                    if self.n_adv > 0
                    else None
                )
                reg = sample_class_region(
                    mod, a_reg, generator=generator, n_adv=self.n_adv, hinge_fn=h_fn
                )
                if bool(reg.valid.any().item()):
                    pts = reg.points[reg.valid].detach()
                    av = a_reg[reg.valid]
                    bv = b_reg[reg.valid]
                    region_terms.append(
                        gci0(mod.membership(pts, av), mod.membership(pts, bv), dead=None).mean()
                    )

        # 3. GCI0_bot: A <= bot
        if "gci0_bot" in td and len(td["gci0_bot"]) > 0:
            rows = self._sample_rows(td["gci0_bot"][:], generator)
            a_col = rows[:, 0]
            bsz = int(a_col.shape[0])
            cand = self._batch_candidate_elements(a_col, generator)
            x_mat = sample_elements(
                a_col, self.nb_elements, self.n_random_elements, generator, batch_elements=cand
            )
            s_cnt = int(x_mat.shape[1])
            x_flat = x_mat.reshape(-1)
            a_flat = a_col.unsqueeze(1).expand(bsz, s_cnt).reshape(-1)
            px = pos_all[x_flat]
            ma = mod.membership(px, a_flat)
            # At x == A, fact + gci0_bot directly forces dead[A] >= 0.5 -> 1.0
            is_self = x_flat == a_flat
            ma_eff = th.where(is_self, th.maximum(ma, (1.0 - dead_all[x_flat]).detach()), ma)
            axiom_terms.append(gci0_bot(ma_eff, dead=dead_all[x_flat]).mean())
            if do_region:
                a_reg = a_col.repeat(self.n_region_samples)
                h_fn = (
                    (lambda z: gci0_bot(mod.membership(z, a_reg), dead=None))
                    if self.n_adv > 0
                    else None
                )
                reg = sample_class_region(
                    mod, a_reg, generator=generator, n_adv=self.n_adv, hinge_fn=h_fn
                )
                if bool(reg.valid.any().item()):
                    pts = reg.points[reg.valid].detach()
                    av = a_reg[reg.valid]
                    # Region of A <= bot must be empty: penalise membership without detaching points from box center
                    region_terms.append(gci0_bot(mod.membership(pts, av), dead=None).mean())

        # 4. GCI1: A cap B <= C
        if "gci1" in td and len(td["gci1"]) > 0:
            rows = self._sample_rows(td["gci1"][:], generator)
            bsz = int(rows.shape[0])
            cand = self._batch_candidate_elements(rows.reshape(-1), generator)
            x_mat = sample_elements(
                rows[:, :2], self.nb_elements, self.n_random_elements, generator, batch_elements=cand
            )
            s_cnt = int(x_mat.shape[1])
            x_flat = x_mat.reshape(-1)
            a_flat = rows[:, 0].unsqueeze(1).expand(bsz, s_cnt).reshape(-1)
            b_flat = rows[:, 1].unsqueeze(1).expand(bsz, s_cnt).reshape(-1)
            c_flat = rows[:, 2].unsqueeze(1).expand(bsz, s_cnt).reshape(-1)
            px = pos_all[x_flat]
            axiom_terms.append(
                gci1(
                    mod.membership(px, a_flat),
                    mod.membership(px, b_flat),
                    mod.membership(px, c_flat),
                    dead=dead_gate[x_flat],
                ).mean()
            )
            if do_region:
                a_reg = rows[:, 0].repeat(self.n_region_samples)
                b_reg = rows[:, 1].repeat(self.n_region_samples)
                c_reg = rows[:, 2].repeat(self.n_region_samples)
                h_fn = (
                    (
                        lambda z: gci1(
                            mod.membership(z, a_reg),
                            mod.membership(z, b_reg),
                            mod.membership(z, c_reg),
                            dead=None,
                        )
                    )
                    if self.n_adv > 0
                    else None
                )
                reg = sample_pair_region(
                    mod, a_reg, b_reg, generator=generator, n_adv=self.n_adv, hinge_fn=h_fn
                )
                if bool(reg.valid.any().item()):
                    pts = reg.points[reg.valid].detach()
                    av = a_reg[reg.valid]
                    bv = b_reg[reg.valid]
                    cv = c_reg[reg.valid]
                    region_terms.append(
                        gci1(
                            mod.membership(pts, av),
                            mod.membership(pts, bv),
                            mod.membership(pts, cv),
                            dead=None,
                        ).mean()
                    )

        # 5. GCI1_bot: A cap B <= bot
        if "gci1_bot" in td and len(td["gci1_bot"]) > 0:
            rows = self._sample_rows(td["gci1_bot"][:], generator)
            bsz = int(rows.shape[0])
            cand = self._batch_candidate_elements(rows[:, :2].reshape(-1), generator)
            x_mat = sample_elements(
                rows[:, :2], self.nb_elements, self.n_random_elements, generator, batch_elements=cand
            )
            s_cnt = int(x_mat.shape[1])
            x_flat = x_mat.reshape(-1)
            a_flat = rows[:, 0].unsqueeze(1).expand(bsz, s_cnt).reshape(-1)
            b_flat = rows[:, 1].unsqueeze(1).expand(bsz, s_cnt).reshape(-1)
            px = pos_all[x_flat]
            ma = mod.membership(px, a_flat)
            mb = mod.membership(px, b_flat)
            # Only allow dead_all[x] to rise when x is genuinely crisp-inside both A and B
            both_crisp = (mod.depth(px, a_flat, crisp=True) > 0) & (
                mod.depth(px, b_flat, crisp=True) > 0
            )
            dead_eff = th.where(both_crisp, dead_all[x_flat], dead_gate[x_flat])
            axiom_terms.append(gci1_bot(ma, mb, dead=dead_eff).mean())
            if do_region:
                a_reg = rows[:, 0].repeat(self.n_region_samples)
                b_reg = rows[:, 1].repeat(self.n_region_samples)
                h_fn = (
                    (
                        lambda z: gci1_bot(
                            mod.membership(z, a_reg), mod.membership(z, b_reg), dead=None
                        )
                    )
                    if self.n_adv > 0
                    else None
                )
                reg = sample_pair_region(
                    mod, a_reg, b_reg, generator=generator, n_adv=self.n_adv, hinge_fn=h_fn
                )
                if bool(reg.valid.any().item()):
                    pts = reg.points[reg.valid].detach()
                    av = a_reg[reg.valid]
                    bv = b_reg[reg.valid]
                    region_terms.append(
                        gci1_bot(mod.membership(pts, av), mod.membership(pts, bv), dead=None).mean()
                    )

        # 6. GCI2: A <= exists r.B
        if "gci2" in td and len(td["gci2"]) > 0:
            rows = self._sample_rows(td["gci2"][:], generator)
            bsz = int(rows.shape[0])
            cand = self._batch_candidate_elements(rows[:, [0, 2]].reshape(-1), generator)
            x_mat = sample_elements(
                rows[:, 0], self.nb_elements, self.n_random_elements, generator, batch_elements=cand
            )
            s_cnt = int(x_mat.shape[1])
            x_flat = x_mat.reshape(-1)
            a_flat = rows[:, 0].unsqueeze(1).expand(bsz, s_cnt).reshape(-1)
            r_flat = rows[:, 1].unsqueeze(1).expand(bsz, s_cnt).reshape(-1)
            b_flat = rows[:, 2].unsqueeze(1).expand(bsz, s_cnt).reshape(-1)
            px = pos_all[x_flat]
            pb = pos_all[b_flat]
            ma = mod.membership(px, a_flat)
            rho_xb = mod.role_membership(px, pb, r_flat)
            df_b = dead_all[b_flat].detach()
            # Edge term uses dead_gate; dead propagation term uses differentiable dead_all[x] when filler is dead
            edge_h = gci2(ma, rho_xb, dead=dead_gate[x_flat], dead_filler=None)
            dead_prop_h = gci2(
                ma,
                th.ones_like(rho_xb),
                dead=th.where(df_b > 0.5, dead_all[x_flat], dead_gate[x_flat]),
                dead_filler=df_b,
            )
            axiom_terms.append((edge_h + dead_prop_h).mean())
            if do_region:
                a_reg = rows[:, 0].repeat(self.n_region_samples)
                r_reg = rows[:, 1].repeat(self.n_region_samples)
                b_reg = rows[:, 2].repeat(self.n_region_samples)
                pb_reg = pos_all[b_reg].detach()
                df_reg = dead_all[b_reg].detach()
                h_fn = (
                    (
                        lambda z: gci2(
                            mod.membership(z, a_reg),
                            mod.role_membership(z, pb_reg, r_reg),
                            dead=None,
                            dead_filler=df_reg,
                        )
                    )
                    if self.n_adv > 0
                    else None
                )
                reg = sample_class_region(
                    mod, a_reg, generator=generator, n_adv=self.n_adv, hinge_fn=h_fn
                )
                if bool(reg.valid.any().item()):
                    pts = reg.points[reg.valid].detach()
                    av = a_reg[reg.valid]
                    rv = r_reg[reg.valid]
                    bv = b_reg[reg.valid]
                    region_terms.append(
                        gci2(
                            mod.membership(pts, av),
                            mod.role_membership(pts, pos_all[bv], rv),
                            dead=None,
                            dead_filler=dead_all[bv].detach(),
                        ).mean()
                    )

        # 7. GCI3: exists r.B <= C
        if "gci3" in td and len(td["gci3"]) > 0:
            rows = self._sample_rows(td["gci3"][:], generator)
            bsz = int(rows.shape[0])
            cand = self._batch_candidate_elements(rows[:, 1:].reshape(-1), generator)
            x_mat = sample_elements(
                rows[:, 2], self.nb_elements, self.n_random_elements, generator, batch_elements=cand
            )
            # Build y candidates from role fillers + row's B + random
            assert self._filler_by_role is not None
            all_fillers = th.unique(
                th.cat(
                    [rows[:, 1]]
                    + [
                        self._filler_by_role[int(r_i)]
                        for r_i in th.unique(rows[:, 0]).cpu().tolist()
                        if int(r_i) in self._filler_by_role
                    ]
                )
            )
            y_mat = sample_filler_elements(
                all_fillers,
                self.nb_elements,
                self.n_random_elements,
                generator=generator,
                batch_size=bsz,
            )
            sx = int(x_mat.shape[1])
            sy = int(y_mat.shape[1])
            x_grid = x_mat.unsqueeze(2).expand(bsz, sx, sy).reshape(-1)
            y_grid = y_mat.unsqueeze(1).expand(bsz, sx, sy).reshape(-1)
            r_grid = rows[:, 0].view(bsz, 1, 1).expand(bsz, sx, sy).reshape(-1)
            b_grid = rows[:, 1].view(bsz, 1, 1).expand(bsz, sx, sy).reshape(-1)
            c_grid = rows[:, 2].view(bsz, 1, 1).expand(bsz, sx, sy).reshape(-1)
            px = pos_all[x_grid]
            py = pos_all[y_grid]
            axiom_terms.append(
                gci3(
                    mod.role_membership(px, py, r_grid),
                    mod.membership(py, b_grid),
                    mod.membership(px, c_grid),
                    dead=dead_gate[x_grid],
                ).mean()
            )
            if do_region:
                r_reg = rows[:, 0].repeat(self.n_region_samples)
                b_reg = rows[:, 1].repeat(self.n_region_samples)
                c_reg = rows[:, 2].repeat(self.n_region_samples)
                h_samp, t_samp = sample_role_region(mod, r_reg, generator=generator)
                if bool(h_samp.valid.any().item()):
                    zx = h_samp.points[h_samp.valid].detach()
                    zy = t_samp.points[t_samp.valid].detach()
                    rv = r_reg[h_samp.valid]
                    bv = b_reg[h_samp.valid]
                    cv = c_reg[h_samp.valid]
                    region_terms.append(
                        gci3(
                            mod.role_membership(zx, zy, rv),
                            mod.membership(zy, bv),
                            mod.membership(zx, cv),
                            dead=None,
                        ).mean()
                    )
                    # Also evaluate at z = zx with y = filler element x_B (§4)
                    py_fill = pos_all[bv]
                    region_terms.append(
                        gci3(
                            mod.role_membership(zx, py_fill, rv),
                            mod.membership(py_fill, bv),
                            mod.membership(zx, cv),
                            dead=None,
                        ).mean()
                    )

        # 8. GCI3_bot: exists r.B <= bot
        if "gci3_bot" in td and len(td["gci3_bot"]) > 0:
            rows = self._sample_rows(td["gci3_bot"][:], generator)
            bsz = int(rows.shape[0])
            cand = self._batch_candidate_elements(rows[:, 1], generator)
            x_mat = sample_elements(
                rows[:, 1], self.nb_elements, self.n_random_elements, generator, batch_elements=cand
            )
            assert self._filler_by_role is not None
            all_fillers = th.unique(
                th.cat(
                    [rows[:, 1]]
                    + [
                        self._filler_by_role[int(r_i)]
                        for r_i in th.unique(rows[:, 0]).cpu().tolist()
                        if int(r_i) in self._filler_by_role
                    ]
                )
            )
            y_mat = sample_filler_elements(
                all_fillers,
                self.nb_elements,
                self.n_random_elements,
                generator=generator,
                batch_size=bsz,
            )
            sx = int(x_mat.shape[1])
            sy = int(y_mat.shape[1])
            x_grid = x_mat.unsqueeze(2).expand(bsz, sx, sy).reshape(-1)
            y_grid = y_mat.unsqueeze(1).expand(bsz, sx, sy).reshape(-1)
            r_grid = rows[:, 0].view(bsz, 1, 1).expand(bsz, sx, sy).reshape(-1)
            b_grid = rows[:, 1].view(bsz, 1, 1).expand(bsz, sx, sy).reshape(-1)
            px = pos_all[x_grid]
            py = pos_all[y_grid]
            rho_xy = mod.role_membership(px, py, r_grid)
            mb_y = mod.membership(py, b_grid)
            both_crisp = (mod.role_depth(px, py, r_grid, crisp=True) > 0) & (
                mod.depth(py, b_grid, crisp=True) > 0
            )
            dead_eff = th.where(both_crisp, dead_all[x_grid], dead_gate[x_grid])
            axiom_terms.append(gci3_bot(rho_xy, mb_y, dead=dead_eff).mean())
            if do_region:
                r_reg = rows[:, 0].repeat(self.n_region_samples)
                b_reg = rows[:, 1].repeat(self.n_region_samples)
                h_samp, t_samp = sample_role_region(mod, r_reg, generator=generator)
                if bool(h_samp.valid.any().item()):
                    zx = h_samp.points[h_samp.valid].detach()
                    zy = t_samp.points[t_samp.valid].detach()
                    rv = r_reg[h_samp.valid]
                    bv = b_reg[h_samp.valid]
                    region_terms.append(
                        gci3_bot(
                            mod.role_membership(zx, zy, rv),
                            mod.membership(zy, bv),
                            dead=None,
                        ).mean()
                    )

        # 9. Role inclusion: r <= s
        if "role_inclusion" in td and len(td["role_inclusion"]) > 0:
            rows = self._sample_rows(td["role_inclusion"][:], generator)
            bsz = int(rows.shape[0])
            all_el = self._batch_candidate_elements(
                th.arange(min(self.nb_elements, 16), device=device), generator
            )
            sx = int(all_el.numel())
            x_grid = all_el.view(1, sx, 1).expand(bsz, sx, sx).reshape(-1)
            y_grid = all_el.view(1, 1, sx).expand(bsz, sx, sx).reshape(-1)
            r_grid = rows[:, 0].view(bsz, 1, 1).expand(bsz, sx, sx).reshape(-1)
            s_grid = rows[:, 1].view(bsz, 1, 1).expand(bsz, sx, sx).reshape(-1)
            px = pos_all[x_grid]
            py = pos_all[y_grid]
            axiom_terms.append(
                role_inclusion(
                    mod.role_membership(px, py, r_grid),
                    mod.role_membership(px, py, s_grid),
                ).mean()
            )
            if do_region:
                r_reg = rows[:, 0].repeat(self.n_region_samples)
                s_reg = rows[:, 1].repeat(self.n_region_samples)
                h_samp, t_samp = sample_role_region(mod, r_reg, generator=generator)
                if bool(h_samp.valid.any().item()):
                    zx = h_samp.points[h_samp.valid].detach()
                    zy = t_samp.points[t_samp.valid].detach()
                    rv = r_reg[h_samp.valid]
                    sv = s_reg[h_samp.valid]
                    region_terms.append(
                        role_inclusion(
                            mod.role_membership(zx, zy, rv),
                            mod.role_membership(zx, zy, sv),
                        ).mean()
                    )

        # 10. Role chain: r o s <= t
        if "role_chain" in td and len(td["role_chain"]) > 0:
            rows = self._sample_rows(td["role_chain"][:], generator)
            bsz = int(rows.shape[0])
            all_el = self._batch_candidate_elements(
                th.arange(min(self.nb_elements, 12), device=device), generator
            )
            if all_el.numel() > 10:
                all_el = all_el[:10]
            sx = int(all_el.numel())
            x_g = all_el.view(1, sx, 1, 1).expand(bsz, sx, sx, sx).reshape(-1)
            y_g = all_el.view(1, 1, sx, 1).expand(bsz, sx, sx, sx).reshape(-1)
            z_g = all_el.view(1, 1, 1, sx).expand(bsz, sx, sx, sx).reshape(-1)
            r_g = rows[:, 0].view(bsz, 1, 1, 1).expand(bsz, sx, sx, sx).reshape(-1)
            s_g = rows[:, 1].view(bsz, 1, 1, 1).expand(bsz, sx, sx, sx).reshape(-1)
            t_g = rows[:, 2].view(bsz, 1, 1, 1).expand(bsz, sx, sx, sx).reshape(-1)
            px = pos_all[x_g]
            py = pos_all[y_g]
            pz = pos_all[z_g]
            axiom_terms.append(
                role_chain(
                    mod.role_membership(px, py, r_g),
                    mod.role_membership(py, pz, s_g),
                    mod.role_membership(px, pz, t_g),
                ).mean()
            )
            if do_region:
                r_reg = rows[:, 0].repeat(self.n_region_samples)
                s_reg = rows[:, 1].repeat(self.n_region_samples)
                t_reg = rows[:, 2].repeat(self.n_region_samples)
                hx, hy = sample_role_region(mod, r_reg, generator=generator)
                _, tz = sample_role_region(mod, s_reg, generator=generator)
                v_chain = hx.valid & tz.valid
                if bool(v_chain.any().item()):
                    zx = hx.points[v_chain].detach()
                    zy = hy.points[v_chain].detach()
                    zz = tz.points[v_chain].detach()
                    rv = r_reg[v_chain]
                    sv = s_reg[v_chain]
                    tv = t_reg[v_chain]
                    region_terms.append(
                        role_chain(
                            mod.role_membership(zx, zy, rv),
                            mod.role_membership(zy, zz, sv),
                            mod.role_membership(zx, zz, tv),
                        ).mean()
                    )

        # 11. Class assertion: A(a)
        if "class_assertion" in td and len(td["class_assertion"]) > 0:
            rows = self._sample_rows(td["class_assertion"][:], generator)
            el_a = self.nb_classes + rows[:, 0]
            c_id = rows[:, 1]
            axiom_terms.append(
                class_assertion(mod.membership(pos_all[el_a], c_id)).mean()
            )

        # 12. Object property assertion: r(a, b)
        if (
            "object_property_assertion" in td
            and len(td["object_property_assertion"]) > 0
        ):
            rows = self._sample_rows(td["object_property_assertion"][:], generator)
            el_a = self.nb_classes + rows[:, 0]
            r_id = rows[:, 1]
            el_b = self.nb_classes + rows[:, 2]
            axiom_terms.append(
                role_assertion(
                    mod.role_membership(pos_all[el_a], pos_all[el_b], r_id)
                ).mean()
            )

        loss_axioms = (
            th.stack(axiom_terms).sum()
            if axiom_terms
            else th.zeros((), dtype=dtype, device=device)
        )
        loss_region = (
            th.stack(region_terms).sum()
            if region_terms
            else th.zeros((), dtype=dtype, device=device)
        )

        # 13. Minimality (principled negatives)
        ord_cls_list = [
            c
            for c in range(self.nb_classes)
            if c != self.top_id and c != self.bot_id
        ]
        if ord_cls_list and self.eps_min > 0.0:
            ord_cls = th.tensor(ord_cls_list, dtype=th.long, device=device)
            if self.nb_elements * ord_cls.numel() <= 1024:
                all_el = th.arange(self.nb_elements, dtype=th.long, device=device)
                gx = all_el.unsqueeze(1).expand(self.nb_elements, ord_cls.numel()).reshape(-1)
                ga = ord_cls.unsqueeze(0).expand(self.nb_elements, ord_cls.numel()).reshape(-1)
            else:
                n_min = min(self.batch_size, max(256, self.nb_elements * self.n_random_elements))
                gx = th.randint(
                    0, self.nb_elements, (n_min,), dtype=th.long, device=generator.device, generator=generator
                ).to(device=device)
                ga_idx = th.randint(
                    0, ord_cls.numel(), (n_min,), dtype=th.long, device=generator.device, generator=generator
                ).to(device=device)
                ga = ord_cls[ga_idx]
            m_rand = mod.membership(pos_all[gx], ga)
        else:
            m_rand = None

        if self.nb_rels > 0 and self.eps_role > 0.0:
            if (self.nb_elements ** 2) * self.nb_rels <= 1024:
                all_el = th.arange(self.nb_elements, dtype=th.long, device=device)
                all_r = th.arange(self.nb_rels, dtype=th.long, device=device)
                rx = all_el.view(self.nb_elements, 1, 1).expand(self.nb_elements, self.nb_elements, self.nb_rels).reshape(-1)
                ry = all_el.view(1, self.nb_elements, 1).expand(self.nb_elements, self.nb_elements, self.nb_rels).reshape(-1)
                rr = all_r.view(1, 1, self.nb_rels).expand(self.nb_elements, self.nb_elements, self.nb_rels).reshape(-1)
            else:
                n_rmin = min(self.batch_size, max(256, self.nb_elements * self.n_random_elements))
                rx = th.randint(
                    0, self.nb_elements, (n_rmin,), dtype=th.long, device=generator.device, generator=generator
                ).to(device=device)
                ry = th.randint(
                    0, self.nb_elements, (n_rmin,), dtype=th.long, device=generator.device, generator=generator
                ).to(device=device)
                rr = th.randint(
                    0, self.nb_rels, (n_rmin,), dtype=th.long, device=generator.device, generator=generator
                ).to(device=device)
            rho_rand = mod.role_membership(pos_all[rx], pos_all[ry], rr)
        else:
            rho_rand = None

        dead_min_ids = [
            e for e in range(self.nb_elements) if e != self.bot_id
        ]
        dead_for_min = (
            dead_all[th.tensor(dead_min_ids, dtype=th.long, device=device)]
            if dead_min_ids
            else None
        )
        loss_min = minimality(
            m_rand=m_rand,
            rho_rand=rho_rand,
            dead=dead_for_min,
            eps=self.eps_min,
            eps_r=self.eps_role,
            eps_dead=self.eps_dead,
        )

        total = (
            loss_fact
            + loss_axioms
            + self.lambda_reg * loss_region
            + loss_min
        )
        return {
            "fact": loss_fact,
            "axioms": loss_axioms,
            "region": loss_region,
            "minimality": loss_min,
            "total": total,
        }

    def reseed_idle(
        self,
        generator: Optional[th.Generator] = None,
        reseed_size: float = 0.08,
        prune_unused: bool = True,
    ) -> int:
        """Relocate idle class components to violating/shared elements and deactivate unused phantom slots (§5).

        Uses no reasoner:
        1. Determines for each live element :math:`x` and class :math:`A` whether
           :math:`x` currently needs to belong to :math:`A` (either :math:`x = x_A`
           is :math:`A`'s own element, or :math:`x` is crisp-inside :math:`A`, or
           :math:`x` is crisp-inside the LHS of a ``gci0`` / ``gci1`` / ``class_assertion``
           axiom with RHS :math:`A`).
        2. Assigns each required element :math:`x` of class :math:`A` to a dedicated
           component slot (up to ``n_components``) centred at :math:`p_x` whenever
           :math:`x` is not yet crisp-inside :math:`A` (:math:`m_A(x) < 0.5`) or
           shares a component box with another element while an idle slot is free.
        3. When ``prune_unused=True``, deactivates component slots that contain no
           live element so phantom boxes do not produce spurious region violations.

        Guarantees:
        - Reasoner-free (R1): inspects only current module positions, memberships,
          and training axiom tensors.
        - Returns the total number of component slots re-seeded or pruned.

        Non-guarantees:
        - Does not reset Adam optimiser moments for modified parameter rows.

        Parameters
        ----------
        generator : Optional[torch.Generator], default None
            Optional PyTorch RNG.
        reseed_size : float, default 0.08
            Half-width assigned to newly re-seeded component boxes.
        prune_unused : bool, default True
            Whether to deactivate component slots that contain zero live elements.

        Returns
        -------
        int
            Number of modified component slots.
        """
        mod = self.module
        device = self.device
        dtype = mod.class_centers.dtype
        td = self.training_datasets
        moves = 0

        with th.no_grad():
            pos_all = mod.positions()  # (nb_elements, d)
            dead_c = mod.dead() > 0.5
            if self.bot_id is not None:
                dead_c[self.bot_id] = True
            live_mask = ~dead_c

            # Ensure pairwise separation of distinct element positions if two elements collapsed
            if self.nb_elements > 1:
                diff = pos_all.unsqueeze(1) - pos_all.unsqueeze(0)
                dist_inf = th.amax(th.abs(diff), dim=-1)
                dist_inf.fill_diagonal_(float("inf"))
                min_d = float(dist_inf.min().item())
                if min_d < 4.0 * reseed_size and mod.encoder is None:
                    side = int(math.ceil(self.nb_elements ** (1.0 / self.embed_dim)))
                    coords = th.zeros(
                        (self.nb_elements, self.embed_dim), dtype=dtype, device=device
                    )
                    idx_rem = th.arange(self.nb_elements, dtype=th.long, device=device)
                    for d_i in range(self.embed_dim):
                        coords[:, d_i] = (idx_rem % side).to(dtype=dtype)
                        idx_rem = idx_rem // side
                    mod.elem_pos.weight.copy_(coords)
                    pos_all = mod.positions()
                    eff_delta = reseed_size
                else:
                    eff_delta = (
                        min(reseed_size, 0.25 * min_d) if min_d > 1e-4 else reseed_size
                    )
                    eff_delta = max(eff_delta, 0.02)
            else:
                eff_delta = reseed_size
            raw_w_target = _inv_softplus(eff_delta)

            # Syntactic forward target memberships starting from each live class's own element + ABox
            target_m = th.zeros(
                (self.nb_elements, self.nb_classes), dtype=th.bool, device=device
            )
            for c_id in range(self.nb_classes):
                if (
                    c_id != self.bot_id
                    and c_id != self.top_id
                    and bool(live_mask[c_id].item())
                ):
                    target_m[c_id, c_id] = True
            if "class_assertion" in td and len(td["class_assertion"]) > 0:
                for row in td["class_assertion"][:].detach().cpu().tolist():
                    el_a = self.nb_classes + int(row[0])
                    c_id = int(row[1])
                    if bool(live_mask[el_a].item()) and c_id != self.bot_id:
                        target_m[el_a, c_id] = True

            g0 = (
                [
                    (int(r[0]), int(r[1]))
                    for r in td["gci0"][:].detach().cpu().tolist()
                ]
                if "gci0" in td and len(td["gci0"]) > 0
                else []
            )
            g1 = (
                [
                    (int(r[0]), int(r[1]), int(r[2]))
                    for r in td["gci1"][:].detach().cpu().tolist()
                ]
                if "gci1" in td and len(td["gci1"]) > 0
                else []
            )
            g2 = (
                [
                    (int(r[0]), int(r[1]), int(r[2]))
                    for r in td["gci2"][:].detach().cpu().tolist()
                ]
                if "gci2" in td and len(td["gci2"]) > 0
                else []
            )
            g3 = (
                [
                    (int(r[0]), int(r[1]), int(r[2]))
                    for r in td["gci3"][:].detach().cpu().tolist()
                ]
                if "gci3" in td and len(td["gci3"]) > 0
                else []
            )
            g0_bot = (
                [int(r[0]) for r in td["gci0_bot"][:].detach().cpu().tolist()]
                if "gci0_bot" in td and len(td["gci0_bot"]) > 0
                else []
            )
            g1_bot = (
                [
                    (int(r[0]), int(r[1]))
                    for r in td["gci1_bot"][:].detach().cpu().tolist()
                ]
                if "gci1_bot" in td and len(td["gci1_bot"]) > 0
                else []
            )
            g3_bot = (
                [
                    (int(r[0]), int(r[1]))
                    for r in td["gci3_bot"][:].detach().cpu().tolist()
                ]
                if "gci3_bot" in td and len(td["gci3_bot"]) > 0
                else []
            )

            max_iters = max(2, min(self.nb_classes + 1, 16))
            for _ in range(max_iters):
                changed = False
                for a_id, b_id in g0:
                    if b_id == self.bot_id or b_id == self.top_id:
                        continue
                    new_m = target_m[:, a_id] & live_mask & (~target_m[:, b_id])
                    if bool(new_m.any().item()):
                        target_m[:, b_id] |= new_m
                        changed = True
                for a_id, b_id, c_id in g1:
                    if c_id == self.bot_id or c_id == self.top_id:
                        continue
                    new_m = (
                        target_m[:, a_id]
                        & target_m[:, b_id]
                        & live_mask
                        & (~target_m[:, c_id])
                    )
                    if bool(new_m.any().item()):
                        target_m[:, c_id] |= new_m
                        changed = True
                if g2 and g3:
                    for a_id, r_id, b_id in g2:
                        if not bool(live_mask[b_id].item()):
                            continue
                        for r3_id, b3_id, c3_id in g3:
                            if (
                                r3_id == r_id
                                and c3_id != self.bot_id
                                and c3_id != self.top_id
                                and bool(target_m[b_id, b3_id].item())
                            ):
                                new_m = (
                                    target_m[:, a_id]
                                    & live_mask
                                    & (~target_m[:, c3_id])
                                )
                                if bool(new_m.any().item()):
                                    target_m[:, c3_id] |= new_m
                                    changed = True
                # Check bottom rules (gci0_bot, gci1_bot, gci3_bot, gci2 dead filler)
                newly_dead = th.zeros_like(live_mask)
                for a_id in g0_bot:
                    newly_dead |= target_m[:, a_id] & live_mask
                for a_id, b_id in g1_bot:
                    newly_dead |= target_m[:, a_id] & target_m[:, b_id] & live_mask
                for a_id, r_id, b_id in g2:
                    if not bool(live_mask[b_id].item()):
                        newly_dead |= target_m[:, a_id] & live_mask
                    else:
                        for r3_id, b3_id in g3_bot:
                            if r3_id == r_id and bool(target_m[b_id, b3_id].item()):
                                newly_dead |= target_m[:, a_id] & live_mask
                if bool(newly_dead.any().item()):
                    live_mask = live_mask & (~newly_dead)
                    dead_c = ~live_mask
                    target_m[newly_dead, :] = False
                    mod.dead_logits[newly_dead] = th.clamp(
                        mod.dead_logits[newly_dead], min=6.0
                    )
                    changed = True
                if not changed:
                    break

            k_slots = mod.n_components
            regions = mod.class_regions()  # (nb_classes, K, d)

            for c_id in range(self.nb_classes):
                if c_id == self.top_id or c_id == self.bot_id:
                    continue
                req_elems = (
                    th.nonzero(target_m[:, c_id], as_tuple=False).view(-1).cpu().tolist()
                )
                if not req_elems:
                    if prune_unused and bool(mod.class_active[c_id].any().item()):
                        mod.class_active[c_id, :] = False
                        mod.class_raw_widths[c_id, :].fill_(_INACTIVE_RAW_WIDTH)
                        moves += 1
                    continue

                # Determine which active components currently cover which live elements
                cen_c = regions.cen[c_id]  # (K, d)
                w_c = regions.w[c_id]  # (K, d)
                act_c = regions.active[c_id]  # (K,)
                # Component depth for all live elements: (nb_elements, K)
                c_dep = th.amin(
                    w_c.unsqueeze(0) - th.abs(pos_all.unsqueeze(1) - cen_c.unsqueeze(0)),
                    dim=-1,
                )
                c_dep = c_dep.masked_fill(~act_c.unsqueeze(0), float("-inf"))
                inside_ck = (c_dep > 0.0) & live_mask.unsqueeze(1)  # (nb_elements, K)

                # Check if any required element is missing or if a component covers non-target elements
                missing_elems = [
                    x_el for x_el in req_elems if not bool(inside_ck[x_el].any().item())
                ]
                spurious_mask = (~target_m[:, c_id]) & live_mask
                has_spurious = bool(inside_ck[spurious_mask].any().item())
                multi_covered = bool((inside_ck[req_elems].sum(dim=0) > 1).any().item())

                if len(req_elems) <= k_slots and (
                    missing_elems or has_spurious or prune_unused
                ):
                    # Assign one dedicated δ-box per required element and deactivate unused slots
                    mod.class_active[c_id, :] = False
                    mod.class_raw_widths[c_id, :].fill_(_INACTIVE_RAW_WIDTH)
                    mod.class_centers[c_id, :].zero_()
                    for slot_idx, x_el in enumerate(req_elems):
                        mod.class_centers[c_id, slot_idx].copy_(pos_all[x_el])
                        mod.class_raw_widths[c_id, slot_idx].fill_(raw_w_target)
                        mod.class_active[c_id, slot_idx] = True
                        moves += 1
                elif missing_elems:
                    # K < len(req_elems): relocate least-used component to missing elements
                    usage = inside_ck.sum(dim=0).cpu().numpy()
                    for x_el in missing_elems:
                        k_star = int(np.argmin(usage))
                        mod.class_centers[c_id, k_star].copy_(pos_all[x_el])
                        mod.class_raw_widths[c_id, k_star].fill_(raw_w_target)
                        mod.class_active[c_id, k_star] = True
                        usage[k_star] += 1
                        moves += 1
                elif prune_unused:
                    # Deactivate any active slot that contains 0 live elements
                    used_slots = inside_ck.any(dim=0)
                    unused_active = act_c & (~used_slots)
                    if bool(unused_active.any().item()):
                        mod.class_active[c_id, unused_active] = False
                        mod.class_raw_widths[c_id, unused_active] = _INACTIVE_RAW_WIDTH
                        moves += int(unused_active.sum().item())

        return moves

    def train(
        self,
        epochs: Optional[int] = None,
        validate_every: int = 1,
        epoch_callback: Optional[Callable[[int, "GroundedEL"], None]] = None,
        log_every: Optional[int] = None,
    ) -> List[Dict[str, float]]:
        """Run the GroundedEL training loop (§3, §4, §5).

        Each epoch:
        1. Anneals :math:`\\tau` geometrically from ``tau0`` to ``tau1``.
        2. Optionally calls :meth:`reseed_idle` when ``reseed_every > 0`` and
           ``epoch > 0 and epoch % reseed_every == 0``.
        3. Computes ``L = L_fact + L_axioms + lambda_reg * L_region + L_minimality``
           via :meth:`compute_batch_losses` and performs one Adam step.
        4. Logs per-term losses every ``log_every`` epochs.

        Guarantees:
        - Reasoner-free (R1): never imports or calls any reasoner or semantic
          completion algorithm.
        - Deterministic for a fixed ``self.seed``.
        - Records per-epoch loss dictionaries in :attr:`loss_history` and returns
          them.

        Non-guarantees:
        - When ``epochs == 0``, performs zero gradient steps and returns ``[]``.

        Parameters
        ----------
        epochs : Optional[int], default None
            Number of training epochs :math:`\\ge 0`; defaults to ``self.epochs``
            when ``None``.
        validate_every : int, default 1
            Retained for :class:`~mowl.base_models.elmodel.EmbeddingELModel`
            interface compatibility.
        epoch_callback : Optional[Callable[[int, GroundedEL], None]], default None
            Optional callback ``epoch_callback(epoch, model)`` invoked at the end
            of each epoch.
        log_every : Optional[int], default None
            Logging interval in epochs; defaults to ``self.log_every`` when ``None``.

        Returns
        -------
        List[Dict[str, float]]
            Per-epoch recorded loss dictionaries.
        """
        eff_epochs = self.epochs if epochs is None else _validate_int_ge("epochs", epochs, 0)
        eff_log_every = (
            self.log_every
            if log_every is None
            else _validate_int_ge("log_every", log_every, 0)
        )
        self.loss_history = []
        if eff_epochs == 0:
            return self.loss_history

        optimizer = th.optim.Adam(self.module.parameters(), lr=self.learning_rate)
        gen = th.Generator(device="cpu")
        gen.manual_seed(self.seed + 1337)

        for epoch in range(eff_epochs):
            self.module.train()
            if eff_epochs > 1:
                tau = self.tau0 * (
                    (self.tau1 / self.tau0) ** (epoch / (eff_epochs - 1))
                )
            else:
                tau = self.tau0
            self.module.set_tau(float(tau))

            losses = self.compute_batch_losses(generator=gen)
            total_loss = losses["total"]

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if self.bot_id is not None:
                with th.no_grad():
                    self.module.dead_logits[self.bot_id] = _DEAD_LOGIT

            do_reseed_now = (
                self.reseed_every > 0
                and (
                    (epoch + 1) % self.reseed_every == 0
                    or epoch + 1 == eff_epochs
                )
            ) or (
                self.reseed_every == 0
                and eff_epochs >= 50
                and epoch + 1 == eff_epochs
            )
            if do_reseed_now:
                self.reseed_idle(generator=gen, prune_unused=True)
                optimizer.state.clear()
                with th.no_grad():
                    losses = self.compute_batch_losses(generator=gen)

            rec = {k: float(v.detach().item()) for k, v in losses.items()}
            rec["tau"] = float(tau)
            self.loss_history.append(rec)

            if eff_log_every > 0 and (
                (epoch + 1) % eff_log_every == 0 or epoch == 0 or epoch + 1 == eff_epochs
            ):
                logger.info(
                    "Epoch %d/%d (tau=%.4f): total=%.4e fact=%.4e axioms=%.4e region=%.4e min=%.4e",
                    epoch + 1,
                    eff_epochs,
                    tau,
                    rec["total"],
                    rec["fact"],
                    rec["axioms"],
                    rec["region"],
                    rec["minimality"],
                )

            if epoch_callback is not None:
                epoch_callback(epoch, self)

        if self.model_filepath is not None:
            th.save(self.module.state_dict(), self.model_filepath)

        return self.loss_history

    def _box_corner_samples(
        self,
        cen: th.Tensor,
        w: th.Tensor,
        active: th.Tensor,
        gen: th.Generator,
        max_corners_per_box: int = 32,
    ) -> th.Tensor:
        """Return interior-nudged box corner points ``c ± (1 - 1e-4) * w`` for active non-empty boxes."""
        valid_k = active & (th.amin(w, dim=-1) > 0.0)
        if not bool(valid_k.any().item()):
            return th.empty((0, self.embed_dim), dtype=cen.dtype, device=cen.device)
        c_act = cen[valid_k]  # (M, d)
        w_act = w[valid_k] * (1.0 - 1e-4)  # (M, d)
        m_cnt, dim = c_act.shape
        if dim <= 5 and (1 << dim) <= max_corners_per_box:
            signs = th.tensor(
                [
                    [(1.0 if (mask >> bit) & 1 else -1.0) for bit in range(dim)]
                    for mask in range(1 << dim)
                ],
                dtype=cen.dtype,
                device=cen.device,
            )  # (2^d, d)
            pts = c_act.unsqueeze(1) + w_act.unsqueeze(1) * signs.unsqueeze(0)
            return pts.reshape(-1, dim)
        u_bits = (
            th.randint(
                0,
                2,
                (m_cnt, max_corners_per_box, dim),
                device=gen.device,
                generator=gen,
            ).to(device=cen.device, dtype=cen.dtype)
            * 2.0
            - 1.0
        )
        pts = c_act.unsqueeze(1) + w_act.unsqueeze(1) * u_bits
        return pts.reshape(-1, dim)

    def crisp_report(self, n_eval: int = 10000) -> Dict[str, float]:
        """Compute reasoner-free crisp violation metrics ``elem_viol`` and ``region_viol`` (§7).

        Guarantees:
        - Reasoner-free (R1): never imports or calls any reasoner or
          :class:`~mowl.geometric.completion.CanonicalModel`.
        - Returns ``{"elem_viol": float, "region_viol": float}`` with values in
          ``[0.0, 1.0]``.

        Non-guarantees:
        - Does not compute reasoner-dependent metrics (``exist``, ``dead_ok``,
          ``faithful_named_*``, ``exact``).

        Parameters
        ----------
        n_eval : int, default 10000
            Total evaluation sample budget across axioms for ``region_viol``.

        Returns
        -------
        Dict[str, float]
            ``{"elem_viol": ..., "region_viol": ...}``.
        """
        _validate_int_ge("n_eval", n_eval, 1)
        mod = self.module
        device = self.device
        td = self.training_datasets
        gen = th.Generator(device="cpu")
        gen.manual_seed(self.seed + 9999)

        with th.no_grad():
            pos_all = mod.positions()  # (nb_elements, d)
            dead_c = mod.dead() > 0.5
            if self.bot_id is not None:
                dead_c[self.bot_id] = True
            dead_f = dead_c.to(dtype=th.float32)
            live_mask = ~dead_c
            live_ids = th.nonzero(live_mask, as_tuple=False).view(-1)

            if live_ids.numel() > 256:
                perm = th.randperm(live_ids.numel(), device="cpu", generator=gen)[:256].to(
                    device=device
                )
                eval_elem_ids = live_ids[perm]
            else:
                eval_elem_ids = live_ids
            eval_pos = pos_all[eval_elem_ids]
            eval_dead_f = dead_f[eval_elem_ids]

            # Precompute crisp membership for all elements when nb_elements * nb_classes <= 200_000
            elem_m_all = th.stack(
                [
                    crisp_membership(mod.depth(pos_all, c_id, crisp=True))
                    for c_id in range(self.nb_classes)
                ],
                dim=1,
            )  # (nb_elements, nb_classes)
            elem_m_live = elem_m_all[eval_elem_ids]  # (L, nb_classes)

            elem_viol_count = 0
            elem_total_count = 0

            # Count total TBox/RBox axioms to divide n_eval budget cleanly
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
            n_per_ax = max(16, min(2000, n_eval // max(total_region_axioms, 1)))
            reg_viol_axioms = 0
            reg_total_axioms = 0

            all_regs = mod.class_regions()

            # 1. GCI0: A <= B
            if "gci0" in td and len(td["gci0"]) > 0:
                for row in td["gci0"][:].detach().cpu().tolist():
                    a_id, b_id = int(row[0]), int(row[1])
                    if eval_elem_ids.numel() > 0:
                        h_e = gci0(
                            elem_m_live[:, a_id], elem_m_live[:, b_id], dead=eval_dead_f
                        )
                        elem_viol_count += int((h_e > 0.0).sum().item())
                        elem_total_count += int(h_e.numel())
                    reg_total_axioms += 1
                    s_reg = sample_class_region(
                        mod, a_id, generator=gen, n_samples=n_per_ax
                    )
                    corners = self._box_corner_samples(
                        all_regs.cen[a_id : a_id + 1],
                        all_regs.w[a_id : a_id + 1],
                        all_regs.active[a_id : a_id + 1],
                        gen,
                    )
                    pts_list = [eval_pos]
                    if bool(s_reg.valid.any().item()):
                        pts_list.append(s_reg.points[s_reg.valid])
                    if corners.numel() > 0:
                        pts_list.append(corners)
                    pts = th.cat(pts_list, dim=0)
                    if pts.shape[0] > 0:
                        ma = crisp_membership(mod.depth(pts, a_id, crisp=True))
                        mb = crisp_membership(mod.depth(pts, b_id, crisp=True))
                        if bool((gci0(ma, mb, dead=None) > 0.0).any().item()):
                            reg_viol_axioms += 1

            # 2. GCI0_bot: A <= bot
            if "gci0_bot" in td and len(td["gci0_bot"]) > 0:
                for row in td["gci0_bot"][:].detach().cpu().tolist():
                    a_id = int(row[0])
                    if eval_elem_ids.numel() > 0:
                        h_e = gci0_bot(elem_m_live[:, a_id], dead=eval_dead_f)
                        elem_viol_count += int((h_e > 0.0).sum().item())
                        elem_total_count += int(h_e.numel())
                    reg_total_axioms += 1
                    s_reg = sample_class_region(
                        mod, a_id, generator=gen, n_samples=n_per_ax
                    )
                    if bool(s_reg.valid.any().item()):
                        reg_viol_axioms += 1
                    elif eval_pos.shape[0] > 0:
                        ma = crisp_membership(mod.depth(eval_pos, a_id, crisp=True))
                        if bool((gci0_bot(ma, dead=None) > 0.0).any().item()):
                            reg_viol_axioms += 1

            # 3. GCI1: A cap B <= C
            if "gci1" in td and len(td["gci1"]) > 0:
                for row in td["gci1"][:].detach().cpu().tolist():
                    a_id, b_id, c_id = int(row[0]), int(row[1]), int(row[2])
                    if eval_elem_ids.numel() > 0:
                        h_e = gci1(
                            elem_m_live[:, a_id],
                            elem_m_live[:, b_id],
                            elem_m_live[:, c_id],
                            dead=eval_dead_f,
                        )
                        elem_viol_count += int((h_e > 0.0).sum().item())
                        elem_total_count += int(h_e.numel())
                    reg_total_axioms += 1
                    s_reg = sample_pair_region(
                        mod, a_id, b_id, generator=gen, n_samples=n_per_ax
                    )
                    pts_list = [eval_pos]
                    if bool(s_reg.valid.any().item()):
                        pts_list.append(s_reg.points[s_reg.valid])
                    pts = th.cat(pts_list, dim=0)
                    if pts.shape[0] > 0:
                        ma = crisp_membership(mod.depth(pts, a_id, crisp=True))
                        mb = crisp_membership(mod.depth(pts, b_id, crisp=True))
                        mc = crisp_membership(mod.depth(pts, c_id, crisp=True))
                        if bool((gci1(ma, mb, mc, dead=None) > 0.0).any().item()):
                            reg_viol_axioms += 1

            # 4. GCI1_bot: A cap B <= bot
            if "gci1_bot" in td and len(td["gci1_bot"]) > 0:
                for row in td["gci1_bot"][:].detach().cpu().tolist():
                    a_id, b_id = int(row[0]), int(row[1])
                    if eval_elem_ids.numel() > 0:
                        h_e = gci1_bot(
                            elem_m_live[:, a_id], elem_m_live[:, b_id], dead=eval_dead_f
                        )
                        elem_viol_count += int((h_e > 0.0).sum().item())
                        elem_total_count += int(h_e.numel())
                    reg_total_axioms += 1
                    s_inter = sample_pair_region(
                        mod, a_id, b_id, generator=gen, n_samples=n_per_ax
                    )
                    if bool(s_inter.valid.any().item()):
                        reg_viol_axioms += 1
                    else:
                        s_a = sample_class_region(
                            mod, a_id, generator=gen, n_samples=n_per_ax
                        )
                        pts_list = [eval_pos]
                        if bool(s_a.valid.any().item()):
                            pts_list.append(s_a.points[s_a.valid])
                        pts = th.cat(pts_list, dim=0)
                        if pts.shape[0] > 0:
                            ma = crisp_membership(mod.depth(pts, a_id, crisp=True))
                            mb = crisp_membership(mod.depth(pts, b_id, crisp=True))
                            if bool((gci1_bot(ma, mb, dead=None) > 0.0).any().item()):
                                reg_viol_axioms += 1

            # 5. GCI2: A <= exists r.B
            if "gci2" in td and len(td["gci2"]) > 0:
                for row in td["gci2"][:].detach().cpu().tolist():
                    a_id, r_id, b_id = int(row[0]), int(row[1]), int(row[2])
                    pb = pos_all[b_id]
                    if eval_elem_ids.numel() > 0:
                        rho_e = crisp_membership(
                            mod.role_depth(eval_pos, pb, r_id, crisp=True)
                        )
                        db_e = dead_f[b_id].expand_as(eval_dead_f)
                        h_e = gci2(
                            elem_m_live[:, a_id],
                            rho_e,
                            dead=eval_dead_f,
                            dead_filler=db_e,
                        )
                        elem_viol_count += int((h_e > 0.0).sum().item())
                        elem_total_count += int(h_e.numel())
                    reg_total_axioms += 1
                    s_reg = sample_class_region(
                        mod, a_id, generator=gen, n_samples=n_per_ax
                    )
                    corners = self._box_corner_samples(
                        all_regs.cen[a_id : a_id + 1],
                        all_regs.w[a_id : a_id + 1],
                        all_regs.active[a_id : a_id + 1],
                        gen,
                    )
                    pts_list = [eval_pos]
                    if bool(s_reg.valid.any().item()):
                        pts_list.append(s_reg.points[s_reg.valid])
                    if corners.numel() > 0:
                        pts_list.append(corners)
                    pts = th.cat(pts_list, dim=0)
                    if pts.shape[0] > 0:
                        ma = crisp_membership(mod.depth(pts, a_id, crisp=True))
                        rho_z = crisp_membership(
                            mod.role_depth(pts, pb, r_id, crisp=True)
                        )
                        db_z = dead_f[b_id].expand_as(ma)
                        if bool(
                            (gci2(ma, rho_z, dead=None, dead_filler=db_z) > 0.0)
                            .any()
                            .item()
                        ):
                            reg_viol_axioms += 1

            # 6. GCI3: exists r.B <= C
            if "gci3" in td and len(td["gci3"]) > 0:
                y_eval_ids = (
                    eval_elem_ids[:32]
                    if eval_elem_ids.numel() > 32
                    else eval_elem_ids
                )
                for row in td["gci3"][:].detach().cpu().tolist():
                    r_id, b_id, c_id = int(row[0]), int(row[1]), int(row[2])
                    if eval_elem_ids.numel() > 0 and y_eval_ids.numel() > 0:
                        for y_id in y_eval_ids.tolist():
                            rho_xy = crisp_membership(
                                mod.role_depth(eval_pos, pos_all[y_id], r_id, crisp=True)
                            )
                            mb_y = elem_m_all[y_id, b_id].expand_as(rho_xy)
                            h_e = gci3(
                                rho_xy, mb_y, elem_m_live[:, c_id], dead=eval_dead_f
                            )
                            elem_viol_count += int((h_e > 0.0).sum().item())
                            elem_total_count += int(h_e.numel())
                    reg_total_axioms += 1
                    h_s, t_s = sample_role_region(
                        mod, r_id, generator=gen, n_samples=n_per_ax
                    )
                    violated = False
                    if bool(h_s.valid.any().item()):
                        zx = h_s.points[h_s.valid]
                        zy = t_s.points[t_s.valid]
                        rho_z = crisp_membership(
                            mod.role_depth(zx, zy, r_id, crisp=True)
                        )
                        mb_z = crisp_membership(mod.depth(zy, b_id, crisp=True))
                        mc_z = crisp_membership(mod.depth(zx, c_id, crisp=True))
                        if bool((gci3(rho_z, mb_z, mc_z, dead=None) > 0.0).any().item()):
                            violated = True
                    if violated:
                        reg_viol_axioms += 1

            # 7. GCI3_bot: exists r.B <= bot
            if "gci3_bot" in td and len(td["gci3_bot"]) > 0:
                y_eval_ids = (
                    eval_elem_ids[:32]
                    if eval_elem_ids.numel() > 32
                    else eval_elem_ids
                )
                for row in td["gci3_bot"][:].detach().cpu().tolist():
                    r_id, b_id = int(row[0]), int(row[1])
                    if eval_elem_ids.numel() > 0 and y_eval_ids.numel() > 0:
                        for y_id in y_eval_ids.tolist():
                            rho_xy = crisp_membership(
                                mod.role_depth(eval_pos, pos_all[y_id], r_id, crisp=True)
                            )
                            mb_y = elem_m_all[y_id, b_id].expand_as(rho_xy)
                            h_e = gci3_bot(rho_xy, mb_y, dead=eval_dead_f)
                            elem_viol_count += int((h_e > 0.0).sum().item())
                            elem_total_count += int(h_e.numel())
                    reg_total_axioms += 1
                    h_s, t_s = sample_role_region(
                        mod, r_id, generator=gen, n_samples=n_per_ax
                    )
                    if bool(h_s.valid.any().item()):
                        zx = h_s.points[h_s.valid]
                        zy = t_s.points[t_s.valid]
                        rho_z = crisp_membership(
                            mod.role_depth(zx, zy, r_id, crisp=True)
                        )
                        mb_z = crisp_membership(mod.depth(zy, b_id, crisp=True))
                        if bool((gci3_bot(rho_z, mb_z, dead=None) > 0.0).any().item()):
                            reg_viol_axioms += 1

            # 8. Role inclusion: r <= s
            if "role_inclusion" in td and len(td["role_inclusion"]) > 0:
                y_eval_ids = (
                    eval_elem_ids[:32]
                    if eval_elem_ids.numel() > 32
                    else eval_elem_ids
                )
                for row in td["role_inclusion"][:].detach().cpu().tolist():
                    r_id, s_id = int(row[0]), int(row[1])
                    if eval_elem_ids.numel() > 0 and y_eval_ids.numel() > 0:
                        for y_id in y_eval_ids.tolist():
                            rr = crisp_membership(
                                mod.role_depth(eval_pos, pos_all[y_id], r_id, crisp=True)
                            )
                            rs = crisp_membership(
                                mod.role_depth(eval_pos, pos_all[y_id], s_id, crisp=True)
                            )
                            h_e = role_inclusion(rr, rs)
                            elem_viol_count += int((h_e > 0.0).sum().item())
                            elem_total_count += int(h_e.numel())
                    reg_total_axioms += 1
                    h_s, t_s = sample_role_region(
                        mod, r_id, generator=gen, n_samples=n_per_ax
                    )
                    if bool(h_s.valid.any().item()):
                        zx = h_s.points[h_s.valid]
                        zy = t_s.points[t_s.valid]
                        rr = crisp_membership(mod.role_depth(zx, zy, r_id, crisp=True))
                        rs = crisp_membership(mod.role_depth(zx, zy, s_id, crisp=True))
                        if bool((role_inclusion(rr, rs) > 0.0).any().item()):
                            reg_viol_axioms += 1

            # 9. Role chain: r o s <= t
            if "role_chain" in td and len(td["role_chain"]) > 0:
                c_eval_ids = (
                    eval_elem_ids[:16]
                    if eval_elem_ids.numel() > 16
                    else eval_elem_ids
                )
                c_pos = pos_all[c_eval_ids]
                for row in td["role_chain"][:].detach().cpu().tolist():
                    r_id, s_id, t_id = int(row[0]), int(row[1]), int(row[2])
                    if c_eval_ids.numel() > 0:
                        for y_id in c_eval_ids.tolist():
                            rho_xy = crisp_membership(
                                mod.role_depth(c_pos, pos_all[y_id], r_id, crisp=True)
                            )
                            for z_id in c_eval_ids.tolist():
                                rho_yz = crisp_membership(
                                    mod.role_depth(
                                        pos_all[y_id], pos_all[z_id], s_id, crisp=True
                                    )
                                ).expand_as(rho_xy)
                                rho_xz = crisp_membership(
                                    mod.role_depth(
                                        c_pos, pos_all[z_id], t_id, crisp=True
                                    )
                                )
                                h_e = role_chain(rho_xy, rho_yz, rho_xz)
                                elem_viol_count += int((h_e > 0.0).sum().item())
                                elem_total_count += int(h_e.numel())
                    reg_total_axioms += 1
                    hx, hy = sample_role_region(
                        mod, r_id, generator=gen, n_samples=n_per_ax
                    )
                    _, tz = sample_role_region(
                        mod, s_id, generator=gen, n_samples=n_per_ax
                    )
                    v_c = hx.valid & tz.valid
                    if bool(v_c.any().item()):
                        zx = hx.points[v_c]
                        zy = hy.points[v_c]
                        zz = tz.points[v_c]
                        rr = crisp_membership(mod.role_depth(zx, zy, r_id, crisp=True))
                        rs = crisp_membership(mod.role_depth(zy, zz, s_id, crisp=True))
                        rt = crisp_membership(mod.role_depth(zx, zz, t_id, crisp=True))
                        if bool((role_chain(rr, rs, rt) > 0.0).any().item()):
                            reg_viol_axioms += 1

            # 10. ABox assertions
            if "class_assertion" in td and len(td["class_assertion"]) > 0:
                for row in td["class_assertion"][:].detach().cpu().tolist():
                    el_a = self.nb_classes + int(row[0])
                    c_id = int(row[1])
                    h_e = class_assertion(elem_m_all[el_a : el_a + 1, c_id])
                    elem_viol_count += int((h_e > 0.0).sum().item())
                    elem_total_count += 1

            if (
                "object_property_assertion" in td
                and len(td["object_property_assertion"]) > 0
            ):
                for row in td["object_property_assertion"][:].detach().cpu().tolist():
                    el_a = self.nb_classes + int(row[0])
                    r_id = int(row[1])
                    el_b = self.nb_classes + int(row[2])
                    rho_ab = crisp_membership(
                        mod.role_depth(pos_all[el_a], pos_all[el_b], r_id, crisp=True)
                    )
                    h_e = role_assertion(rho_ab)
                    elem_viol_count += int((h_e > 0.0).sum().item())
                    elem_total_count += 1

            elem_viol = (
                float(elem_viol_count / elem_total_count)
                if elem_total_count > 0
                else 0.0
            )
            region_viol = (
                float(reg_viol_axioms / reg_total_axioms)
                if reg_total_axioms > 0
                else 0.0
            )
            return {
                "elem_viol": elem_viol,
                "region_viol": region_viol,
            }

    def evaluate_model(
        self,
        use_reasoner: bool = True,
        n_eval: int = 10000,
    ) -> Dict[str, Any]:
        """Compute §7 evaluation metrics dictionary.

        Lazily imports :class:`~mowl.geometric.completion.CanonicalModel` inside
        this method ONLY when ``use_reasoner=True``.

        Guarantees:
        - Returns a dict with keys ``"elem_viol"``, ``"region_viol"``, ``"exist"``,
          ``"dead_ok"``, ``"faithful_named_precision"``, ``"faithful_named_recall"``,
          ``"faithful_named"``, and ``"exact"``.
        - ``exact`` is ``True`` if and only if ``elem_viol == 0.0``,
          ``region_viol == 0.0``, ``exist == 1.0``, and ``dead_ok is True``.

        Non-guarantees:
        - When ``use_reasoner=False``, ``exist`` is evaluated over live classes
          (``dead <= 0.5``) and ``faithful_named_*`` are ``None``.

        Parameters
        ----------
        use_reasoner : bool, default True
            Whether to compute ground-truth satisfiability and subsumptions via
            :class:`~mowl.geometric.completion.CanonicalModel`.
        n_eval : int, default 10000
            Sample budget for ``region_viol`` and region inclusion checks.

        Returns
        -------
        Dict[str, Any]
            §7 evaluation metrics dictionary.
        """
        base = self.crisp_report(n_eval=n_eval)
        elem_viol = base["elem_viol"]
        region_viol = base["region_viol"]

        mod = self.module
        device = self.device
        gen = th.Generator(device="cpu")
        gen.manual_seed(self.seed + 2026)

        with th.no_grad():
            pos_all = mod.positions()
            dead_cls = (mod.dead()[: self.nb_classes] > 0.5).cpu()
            if self.bot_id is not None:
                dead_cls[self.bot_id] = True

            self_inside = th.tensor(
                [
                    bool(
                        (
                            mod.depth(pos_all[c_id], c_id, crisp=True) > 0.0
                        ).item()
                    )
                    and not bool(dead_cls[c_id].item())
                    for c_id in range(self.nb_classes)
                ],
                dtype=th.bool,
            )

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
                return {
                    "elem_viol": elem_viol,
                    "region_viol": region_viol,
                    "exist": exist_val,
                    "dead_ok": dead_ok,
                    "faithful_named_precision": None,
                    "faithful_named_recall": None,
                    "faithful_named": (None, None),
                    "exact": exact,
                }

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
            return {
                "elem_viol": elem_viol,
                "region_viol": region_viol,
                "exist": exist_val,
                "dead_ok": dead_ok,
                "faithful_named_precision": precision,
                "faithful_named_recall": recall,
                "faithful_named": (precision, recall),
                "exact": exact,
            }
