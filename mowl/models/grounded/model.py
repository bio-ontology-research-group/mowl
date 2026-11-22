r"""GroundedEL model: element-grounded EL++ embedding model, training loop, and evaluation.

Implements §3, §4, §5, §7, and §8 of the GroundedEL design:

1. **Syntactic domain construction (:math:`\Delta`):** Builds one element per
   class in :attr:`class_index_dict` (including normaliser auxiliary classes and
   :math:`\top, \bot` if present) followed by one element per individual in
   :attr:`individual_index_dict`.
2. **Reasoner-free training loop (:meth:`train`):** Each epoch anneals
   :math:`\tau` geometrically from ``tau0`` to ``tau1``, samples mini-batches
   per non-empty axiom type together with candidate elements (§3) and region
   points (§4), sums :math:`L_{\text{fact}} + L_{\text{axioms}} + \lambda_{\text{reg}} L_{\text{region}} + L_{\text{minimality}}`,
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
  at insufficient component capacity :math:`K`."""
from __future__ import annotations

import logging
import math
import random
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union
import torch as th
import torch.nn as nn
from mowl.base_models.elmodel import EmbeddingELModel
from mowl.nn.el.grounded import GroundedELModule
from mowl.nn.el.grounded.module import (
    _DEAD_LOGIT,
    _validate_int_ge,
    _validate_positive_float,
)
from mowl.representation.library.grounded_logic import (validate_margin as _validate_margin,
                                                        validate_nonneg_weight as _validate_nonneg_weight)
from mowl.nn.el.structural import Regions
from mowl.datasets.rows import sort_rows
from mowl.representation.library.finite_model_placement import (close_labels, edge_triples,
                                                                 interpretation_labels,
                                                                 syntactic_edges)
from mowl.representation.library.inductive_membership import (MembershipCrossEntropy,
                                                              SoftClassScores)
from mowl.representation.library.union_reseeding import OwnPointReseeding, slot_point_counts
from mowl.owlapi.defaults import BOT as BOTTOM, TOP
from mowl.models.grounded.common import _resolve_inverse_of, _validate_index_dict, _coerce_gci_table
from mowl.models.grounded.evaluation import GroundedEvaluation
from mowl.models.grounded.objective import GroundedConstraints

logger = logging.getLogger(__name__)

__all__ = ["GroundedEL"]


class GroundedEL(GroundedConstraints, GroundedEvaluation, EmbeddingELModel):
    """Element-grounded multi-box EL++ embedding model with product roles.

    The model is a combination of parts (see :func:`mowl.representation.library.solved_grounded`):
    ``self.module`` is the geometry (:class:`~mowl.nn.GroundedELModule`: the union of boxes per
    class, the relation model, the positions of the elements; ``self.module.parts`` names them),
    the objective is the grounded constraints
    (:class:`~mowl.models.grounded.objective.GroundedConstraints`: ``compute_batch_losses``),
    the evaluation regime is :class:`~mowl.models.grounded.evaluation.GroundedEvaluation`
    (``crisp_report``, ``evaluate_model``), and the training regime is
    :class:`~mowl.training.GroundedELTraining` (assign another with ``training_regime``).

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
    - When ``neg_weight > 0``, corrupted pairs are not filtered against the ABox
      (a corruption may coincide with a stated or entailed fact).
    - :attr:`dual_weights` are plain floats transient across module
      ``state_dict`` checkpoints.

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
    eps_dead : float, default 1e-3
        Minimality weight :math:`\\varepsilon_\\delta \\ge 0` on element dead scores
        (kept small but nonzero because :math:`\\delta = 1` everywhere is a trivial
        zero of all hinges).
    lambda_reg : float, default 1.0
        Weight :math:`\\lambda_{\\text{reg}} \\ge 0` on region soundness hinges (§4).
    neg_weight : float, default 0.0
        Weight :math:`\\ge 0` on corrupted-fact negatives sampled by replacing
        the head or tail of ``object_property_assertion`` rows with a random
        individual element.
    neg_k : int, default 1
        Number of corrupted negative rows sampled per positive
        ``object_property_assertion`` batch row (:math:`\\ge 1`).
    constraint_mode : {"fixed", "lagrangian"}, default "fixed"
        Constraint weighting mode for ``fact``, ``axioms``, and ``region``
        satisfaction terms: ``"fixed"`` keeps all weights at ``1.0``, whereas
        ``"lagrangian"`` updates :attr:`dual_weights` via dual ascent
        (:meth:`dual_update`).
    dual_rho : float, default 1.0
        Dual-ascent step size :math:`\\rho_{\\text{dual}} \\ge 0` used by
        :meth:`dual_update` in ``"lagrangian"`` mode.
    dual_max : float, default 1e3
        Upper bound :math:`\\ge 1.0` on each weight in :attr:`dual_weights`.
    dual_target : float, default 0.0
        Target violation tolerance :math:`\\ge 0` subtracted from per-term mean
        losses in :meth:`dual_update`.
    n_region_samples : int, default 4
        Number of region points sampled per axiom in each step :math:`\\ge 0`.
    n_adv : int, default 2
        Number of signed-gradient ascent steps :math:`\\ge 0` for adversarial
        region sampling.
    region_margin : float, default 0.0
        Containment margin for geometric region depth hinges (§4).
    n_random_elements : int, default 4
        Number of uniform random elements :math:`\\ge 0` sampled per axiom row.
    n_inside_elements : int, default 32
        Maximum number of elements currently crisp-inside the row's LHS class(es)
        evaluated per axiom row (reasoner-free; reads learned memberships only).
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
    dead_init_logit : float, default 0.0
        Initial value for element deadness :math:`\\delta_x \\in [0, 1]`.
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
    n_filler_samples : int, default 64
        Maximum number of candidate filler elements per batch row in ``gci3``
        and ``gci3_bot`` (always including the row's own filler :math:`x_B`).
    region_lhs_grad : bool, default True
        Whether to keep reparametrised region sample points :math:`z` attached to
        the LHS boxes so gradients flow to both LHS and RHS boxes in the region
        hinges (``False`` detaches :math:`z` for ablation).
    dead_from_bot_only : bool, default True
        Whether element deadness :math:`\\delta_x` receives gradient only from
        :math:`\\bot`-derived hinges (`gci0_bot`, `gci1_bot`, `gci3_bot` element
        hinges and `gci2_dead` filler-dead propagation). When ``False``,
        reproduces the ablation behaviour where violated `gci0` hinges also route
        gradient to :math:`\\delta_x`.
    logic : str, default "lukasiewicz"
        The fuzzy logic in which the element constraints are read (a part,
        :data:`~mowl.representation.library.grounded_logic.RULES`): ``"lukasiewicz"`` (the
        published reading, linear inside the hinge), ``"godel"`` or ``"product"``.
    class_model, relation_model, measure : optional
        Replace a part of the geometry (:class:`~mowl.nn.GroundedELModule`): a name of
        :data:`~mowl.representation.library.grounded_parts.GROUNDED_PARTS`, a factory or a
        part. The relation model replaces ``role_mode``.
    objective, element_sampler, region_sampler, crisp_regime : optional
        Replace the objective (:class:`~mowl.representation.library.grounded_objective.GroundedObjective`),
        the samplers (:class:`~mowl.representation.library.grounded_sampling.ElementSampler`,
        :class:`~mowl.representation.library.grounded_sampling.RegionSampler`) or the
        evaluation regime of :meth:`crisp_report`
        (:class:`~mowl.representation.library.grounded_evaluation.CrispViolations`): a part
        or a name. By default they are built from the options above at every step.
    """

    scored_forms = ("class_assertion",)

    def __init__(
        self,
        dataset=None,
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
        eps_dead: float = 1e-3,
        lambda_reg: float = 1.0,
        n_region_samples: int = 4,
        n_adv: int = 2,
        region_margin: float = 0.0,
        n_random_elements: int = 4,
        n_inside_elements: int = 32,
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
        dead_init_logit: float = 0.0,
        element_features: Optional[th.Tensor] = None,
        encoder: Optional[nn.Module] = None,
        model_filepath: Optional[str] = None,
        load_normalized: bool = False,
        load_role_axioms: bool = True,
        n_filler_samples: int = 64,
        region_lhs_grad: bool = True,
        dead_from_bot_only: bool = True,
        role_mode: str = "product",
        inverse_of: Optional[
            Union[Sequence[int], th.Tensor, Mapping[Union[str, int], Union[str, int]]]
        ] = None,
        neg_weight: float = 0.0,
        neg_k: int = 1,
        constraint_mode: str = "fixed",
        dual_rho: float = 1.0,
        dual_max: float = 1e3,
        dual_target: float = 0.0,
        *,
        class_index_dict: Optional[Mapping[str, int]] = None,
        object_property_index_dict: Optional[Mapping[str, int]] = None,
        individual_index_dict: Optional[Mapping[str, int]] = None,
        named_class_index_dict: Optional[Mapping[str, int]] = None,
        gci0: Optional[Any] = None,
        gci1: Optional[Any] = None,
        gci2: Optional[Any] = None,
        gci3: Optional[Any] = None,
        gci0_bot: Optional[Any] = None,
        gci1_bot: Optional[Any] = None,
        gci3_bot: Optional[Any] = None,
        role_inclusion: Optional[Any] = None,
        role_chain: Optional[Any] = None,
        class_assertion: Optional[Any] = None,
        object_property_assertion: Optional[Any] = None,
        logic: str = "lukasiewicz",
        class_model: Any = None,
        relation_model: Any = None,
        measure: Any = None,
        objective: Any = None,
        element_sampler: Any = None,
        region_sampler: Any = None,
        crisp_regime: Any = None,
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
        neg_weight_f = _validate_nonneg_weight("neg_weight", neg_weight)
        _validate_int_ge("neg_k", neg_k, 1)
        if not isinstance(constraint_mode, str) or constraint_mode not in (
            "fixed",
            "lagrangian",
        ):
            raise ValueError(
                f"constraint_mode must be 'fixed' or 'lagrangian', got {constraint_mode!r}"
            )
        dual_rho_f = _validate_nonneg_weight("dual_rho", dual_rho)
        dual_max_f = _validate_nonneg_weight("dual_max", dual_max)
        if dual_max_f < 1.0:
            raise ValueError(f"dual_max must be >= 1.0, got {dual_max!r}")
        dual_target_f = _validate_nonneg_weight("dual_target", dual_target)
        _validate_int_ge("n_region_samples", n_region_samples, 0)
        _validate_int_ge("n_adv", n_adv, 0)
        region_margin_f = _validate_margin("region_margin", region_margin)
        _validate_int_ge("n_random_elements", n_random_elements, 0)
        _validate_int_ge("n_inside_elements", n_inside_elements, 0)
        _validate_int_ge("n_filler_samples", n_filler_samples, 1)
        if not isinstance(region_lhs_grad, bool):
            raise ValueError(f"region_lhs_grad must be a bool, got {region_lhs_grad!r}")
        if not isinstance(dead_from_bot_only, bool):
            raise ValueError(
                f"dead_from_bot_only must be a bool, got {dead_from_bot_only!r}"
            )
        if not isinstance(role_mode, str) or role_mode not in (
            "product",
            "translation",
        ):
            raise ValueError(
                f"role_mode must be 'product' or 'translation', got {role_mode!r}"
            )
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

        if dataset is None and class_index_dict is None:
            raise TypeError(
                "GroundedEL requires either a mowl Dataset or prebuilt tables (class_index_dict=... or GroundedEL.from_tables(...))"
            )

        if dataset is not None:
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
        else:
            # Prebuilt normal-form table construction without an OWL Dataset
            if model_filepath is not None and not isinstance(model_filepath, str):
                raise TypeError(
                    f"Optional parameter model_filepath must be of type str, got {type(model_filepath).__name__}"
                )
            self.dataset = None
            self.embed_dim = int(embed_dim)
            self.batch_size = int(batch_size)
            self._extended = True
            self._model_filepath = model_filepath
            self._load_normalized = bool(load_normalized)
            self.device = device
            self.learning_rate = float(learning_rate)
            self.neg_sampling_gcis = []
            self._load_role_axioms = bool(load_role_axioms)
            self._eval_gci_name: Optional[str] = None

            cls_map = _validate_index_dict(
                "class_index_dict", class_index_dict, allow_empty=False  # type: ignore[arg-type]
            )
            rel_map = _validate_index_dict(
                "object_property_index_dict",
                object_property_index_dict if object_property_index_dict is not None else {},
                allow_empty=True,
            )
            ind_map = _validate_index_dict(
                "individual_index_dict",
                individual_index_dict if individual_index_dict is not None else {},
                allow_empty=True,
            )
            if named_class_index_dict is not None:
                named_cls_map = dict(named_class_index_dict)
            else:
                named_cls_map = {
                    k: v for k, v in cls_map.items() if k not in (TOP, BOTTOM)
                }

            self._class_index_dict = cls_map
            self._class_index_dict_cache = cls_map
            self._object_property_index_dict = rel_map
            self._object_property_index_dict_cache = rel_map
            self._individual_index_dict = ind_map
            self._named_class_index_dict = named_cls_map
            self._auxiliary_classes = []
            self._auxiliary_object_properties = []

            n_c = len(cls_map)
            n_r = len(rel_map)
            n_i = len(ind_map)
            b_id = cls_map.get(BOTTOM, None)

            td_prebuilt: Dict[str, Any] = {
                "gci0": _coerce_gci_table("gci0", gci0, 2, (n_c, n_c), device),
                "gci1": _coerce_gci_table("gci1", gci1, 3, (n_c, n_c, n_c), device),
                "gci2": _coerce_gci_table("gci2", gci2, 3, (n_c, n_r, n_c), device),
                "gci3": _coerce_gci_table("gci3", gci3, 3, (n_r, n_c, n_c), device),
                "gci0_bot": _coerce_gci_table(
                    "gci0_bot",
                    gci0_bot,
                    2,
                    (n_c, n_c),
                    device,
                    allow_short_bot_col=True,
                    bot_id=b_id,
                ),
                "gci1_bot": _coerce_gci_table(
                    "gci1_bot",
                    gci1_bot,
                    3,
                    (n_c, n_c, n_c),
                    device,
                    allow_short_bot_col=True,
                    bot_id=b_id,
                ),
                "gci3_bot": _coerce_gci_table(
                    "gci3_bot",
                    gci3_bot,
                    3,
                    (n_r, n_c, n_c),
                    device,
                    allow_short_bot_col=True,
                    bot_id=b_id,
                ),
            }
            if role_inclusion is not None or self._load_role_axioms:
                td_prebuilt["role_inclusion"] = _coerce_gci_table(
                    "role_inclusion", role_inclusion, 2, (n_r, n_r), device
                )
            if role_chain is not None or self._load_role_axioms:
                td_prebuilt["role_chain"] = _coerce_gci_table(
                    "role_chain", role_chain, 3, (n_r, n_r, n_r), device
                )
            if class_assertion is not None:
                td_prebuilt["class_assertion"] = _coerce_gci_table(
                    "class_assertion", class_assertion, 2, (n_i, n_c), device
                )
            if object_property_assertion is not None:
                td_prebuilt["object_property_assertion"] = _coerce_gci_table(
                    "object_property_assertion",
                    object_property_assertion,
                    3,
                    (n_i, n_r, n_i),
                    device,
                )

            self._training_datasets = td_prebuilt
            self._validation_datasets = {}
            self._testing_datasets = {}
            self._datasets_loaded = True
            self._dataloaders_loaded = False

        self.n_components = int(n_components)
        self.n_role_components = int(n_role_components)
        self.epochs = int(epochs)
        self.tau0 = tau0_f
        self.tau1 = tau1_f
        self.eps_min = eps_min_f
        self.eps_role = eps_role_f
        self.eps_dead = eps_dead_f
        self.lambda_reg = lambda_reg_f
        self.neg_weight = neg_weight_f
        self.neg_k = int(neg_k)
        self.constraint_mode = constraint_mode
        self.dual_rho = dual_rho_f
        self.dual_max = dual_max_f
        self.dual_target = dual_target_f
        self.dual_weights: Dict[str, float] = {
            "fact": 1.0,
            "axioms": 1.0,
            "region": 1.0,
        }
        self.n_region_samples = int(n_region_samples)
        self.n_adv = int(n_adv)
        self.region_margin = region_margin_f
        self.n_random_elements = int(n_random_elements)
        self.n_inside_elements = int(n_inside_elements)
        self.n_filler_samples = int(n_filler_samples)
        self.region_lhs_grad = bool(region_lhs_grad)
        self.dead_from_bot_only = bool(dead_from_bot_only)
        self.role_mode = role_mode
        # The logic of the element constraints (a part: Łukasiewicz is the published reading).
        from mowl.representation.library.grounded_logic import get_rules
        self.rules = get_rules(logic)
        self.logic = self.rules.name
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
        self.inverse_of: Optional[List[int]] = _resolve_inverse_of(
            inverse_of, self.object_property_index_dict, self.nb_rels
        )
        self.top_id: Optional[int] = self.class_index_dict.get(TOP, None)
        self.bot_id: Optional[int] = self.class_index_dict.get(BOTTOM, None)

        self.loss_history: List[Dict[str, float]] = []
        self.optimizer: Optional[th.optim.Optimizer] = None
        self._filler_by_role: Optional[Dict[int, th.Tensor]] = None
        self._lhs_by_role: Optional[Dict[int, th.Tensor]] = None
        self._filler_mask_by_role: Optional[th.Tensor] = None
        self._live_cls_ids: Optional[th.Tensor] = None
        self._ord_cls_ids: Optional[th.Tensor] = None
        self._dead_min_ids: Optional[th.Tensor] = None
        self.generator: th.Generator = th.Generator(device="cpu")
        self.generator.manual_seed(self.seed + 1337)

        # the parts given instead of the defaults
        self._geometry_parts = {"class_model": class_model, "relation_model": relation_model,
                                "measure": measure}
        from mowl.representation.library.grounded_parts import GROUNDED_PARTS
        for name, kind, part in (("objective", "objective", objective),
                                 ("element_sampler", "element sampler", element_sampler),
                                 ("region_sampler", "region sampler", region_sampler),
                                 ("crisp_regime", "evaluation", crisp_regime)):
            if isinstance(part, str):
                part = GROUNDED_PARTS[kind][part]()
            if part is not None:
                setattr(self, name, part)
        self.init_module()

    @classmethod
    def from_tables(
        cls,
        class_index_dict: Mapping[str, int],
        object_property_index_dict: Optional[Mapping[str, int]] = None,
        gci0: Optional[Any] = None,
        gci1: Optional[Any] = None,
        gci2: Optional[Any] = None,
        gci3: Optional[Any] = None,
        gci0_bot: Optional[Any] = None,
        gci1_bot: Optional[Any] = None,
        gci3_bot: Optional[Any] = None,
        role_inclusion: Optional[Any] = None,
        role_chain: Optional[Any] = None,
        class_assertion: Optional[Any] = None,
        object_property_assertion: Optional[Any] = None,
        individual_index_dict: Optional[Mapping[str, int]] = None,
        named_class_index_dict: Optional[Mapping[str, int]] = None,
        role_mode: str = "product",
        inverse_of: Optional[
            Union[Sequence[int], th.Tensor, Mapping[Union[str, int], Union[str, int]]]
        ] = None,
        **kwargs: Any,
    ) -> "GroundedEL":
        """Construct :class:`GroundedEL` directly from prebuilt normal-form index tables without an OWL file.

        Produces the exact same ``training_datasets`` dictionary structure
        (``"gci0"``, ``"gci1"``, ``"gci2"``, ``"gci3"``, ``"gci0_bot"``,
        ``"gci1_bot"``, ``"gci3_bot"``, ``"role_inclusion"``, ``"role_chain"``)
        consumed by :meth:`compute_batch_losses` and :meth:`crisp_report`.

        Guarantees:

        - Reasoner-free and JVM-free: never starts a JVM or reads an OWL file.
        - Validates bijective ``0 .. N - 1`` index dictionaries and column bounds
          of all supplied GCI tensors.
        - Leaves the existing :class:`~mowl.datasets.Dataset` constructor path
          completely untouched.

        Non-guarantees:

        - Calling :meth:`evaluate_model` with ``use_reasoner=True`` on a
          table-constructed instance raises :class:`ValueError` because no OWL
          ontology is attached; pass ``use_reasoner=False``.

        Parameters
        ----------
        class_index_dict : Mapping[str, int]
            Bijective mapping from class names/IRIs (including ``TOP`` and
            ``BOTTOM`` if used) to integer indices ``0 .. nb_classes - 1``.
        object_property_index_dict : Optional[Mapping[str, int]], default None
            Bijective mapping from relation names/IRIs to ``0 .. nb_rels - 1``.
        gci0 : Optional[Any], default None
            2-D integer tensor/array of shape ``(N0, 2)`` with rows ``[C, D]``
            for :math:`C \\sqsubseteq D`.
        gci1 : Optional[Any], default None
            2-D integer tensor/array of shape ``(N1, 3)`` with rows ``[C, D, E]``
            for :math:`C \\sqcap D \\sqsubseteq E`.
        gci2 : Optional[Any], default None
            2-D integer tensor/array of shape ``(N2, 3)`` with rows ``[C, R, D]``
            for :math:`C \\sqsubseteq \\exists R.D`.
        gci3 : Optional[Any], default None
            2-D integer tensor/array of shape ``(N3, 3)`` with rows ``[R, C, D]``
            for :math:`\\exists R.C \\sqsubseteq D`.
        gci0_bot : Optional[Any], default None
            Optional 2-D integer tensor/array for :math:`C \\sqsubseteq \\bot`
            (shape ``(N, 2)`` or ``(N, 1)``).
        gci1_bot : Optional[Any], default None
            Optional 2-D integer tensor/array for :math:`C \\sqcap D \\sqsubseteq \\bot`
            (shape ``(N, 3)`` or ``(N, 2)``).
        gci3_bot : Optional[Any], default None
            Optional 2-D integer tensor/array for :math:`\\exists R.C \\sqsubseteq \\bot`
            (shape ``(N, 3)`` or ``(N, 2)``).
        role_inclusion : Optional[Any], default None
            Optional 2-D integer tensor/array of shape ``(N, 2)`` for :math:`R \\sqsubseteq S`.
        role_chain : Optional[Any], default None
            Optional 2-D integer tensor/array of shape ``(N, 3)`` for :math:`R \\circ S \\sqsubseteq T`.
        class_assertion : Optional[Any], default None
            Optional 2-D integer tensor/array of shape ``(N, 2)`` for :math:`C(a)`.
        object_property_assertion : Optional[Any], default None
            Optional 2-D integer tensor/array of shape ``(N, 3)`` for :math:`R(a, b)`.
        individual_index_dict : Optional[Mapping[str, int]], default None
            Optional bijective mapping from individual names/IRIs to ``0 .. nb_inds - 1``.
        named_class_index_dict : Optional[Mapping[str, int]], default None
            Optional subset of ``class_index_dict`` representing named classes.
        role_mode : str, default "product"
            Role representation mode (``"product"`` or ``"translation"``).
        inverse_of : Optional[Union[Sequence[int], torch.Tensor, Mapping[Union[str, int], Union[str, int]]]], default None
            Optional inverse-role mapping (role-name -> role-name or index sequence).
        kwargs : Any
            Additional keyword arguments (``**kwargs``) forwarded to :meth:`__init__`.

        Returns
        -------
        GroundedEL
            Initialised model instance backed by the prebuilt tables.
        """
        return cls(
            None,
            class_index_dict=class_index_dict,
            object_property_index_dict=object_property_index_dict,
            individual_index_dict=individual_index_dict,
            named_class_index_dict=named_class_index_dict,
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
            role_mode=role_mode,
            inverse_of=inverse_of,
            **kwargs,
        )

    @property
    def individual_index_dict(self) -> Dict[str, int]:
        """Return the mapping from individual IRI to integer index (``0`` to ``nb_inds - 1``)."""
        cached = getattr(self, "_individual_index_dict", None)
        if cached is not None:
            return cached
        if self.dataset is None:
            self._individual_index_dict = {}
            return self._individual_index_dict
        return super().individual_index_dict

    @property
    def named_class_index_dict(self) -> Dict[str, int]:
        """Return mapping from named class IRI to integer index in ``class_index_dict``."""
        cached = getattr(self, "_named_class_index_dict", None)
        if cached is not None:
            return cached
        if self.dataset is None:
            self._named_class_index_dict = {
                k: v
                for k, v in self.class_index_dict.items()
                if k not in (TOP, BOTTOM)
            }
            return self._named_class_index_dict
        return super().named_class_index_dict

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
        self._live_cls_ids = None
        self._ord_cls_ids = None
        self._dead_min_ids = None
        self._filler_by_role = None
        self._lhs_by_role = None
        self._filler_mask_by_role = None
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
            role_mode=self.role_mode,
            inverse_of=self.inverse_of,
            **getattr(self, "_geometry_parts", {}),
        ).to(self.device)
        self.role_mode = self.module.role_mode

        if self.bot_id is not None:
            # bottom has no element and no box
            self.module.elements.clamp_dead(self.bot_id, _DEAD_LOGIT)
            self.module.class_model.silence(self.bot_id)

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

    def _load_datasets(self) -> None:
        """Load and deterministically sort ``training_datasets`` rows so training is invariant to JVM HashSet order."""
        was_loaded = bool(self._datasets_loaded)
        if not was_loaded:
            random.seed(self.seed)
        super()._load_datasets()
        if not was_loaded and self._training_datasets is not None:
            for ds_obj in self._training_datasets.values():
                sort_rows(getattr(ds_obj, "data", None))

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

        elements = dict(self.class_index_dict)
        for ind_iri, ind_idx in self.individual_index_dict.items():
            elements[ind_iri] = self.nb_classes + int(ind_idx)
        labels = close_labels(
            interpretation_labels(raw_labels, self.nb_elements, self.class_index_dict, elements,
                                  device=self.device),
            self.nb_classes, top_id=self.top_id, bot_id=self.bot_id)
        if raw_edges is not None:
            return labels, edge_triples(raw_edges, self.object_property_index_dict)
        # edges derived syntactically from the labels and the normalised axioms
        return labels, syntactic_edges(labels, self.training_datasets, self.nb_classes,
                                       inverse_of=self.inverse_of)

    def _ensure_static_id_tensors(self) -> None:
        """Lazily build cached class and element index tensors on ``self.device``."""
        device = self.device
        if self._live_cls_ids is None:
            self._live_cls_ids = th.tensor(
                [c for c in range(self.nb_classes) if c != self.bot_id],
                dtype=th.long,
                device=device,
            )
        if self._ord_cls_ids is None:
            self._ord_cls_ids = th.tensor(
                [
                    c
                    for c in range(self.nb_classes)
                    if c != self.top_id and c != self.bot_id
                ],
                dtype=th.long,
                device=device,
            )
        if self._dead_min_ids is None:
            self._dead_min_ids = th.tensor(
                [e for e in range(self.nb_elements) if e != self.bot_id],
                dtype=th.long,
                device=device,
            )

    def _ensure_role_tables(self) -> None:
        """Build cached per-role filler and LHS element index tables from ``training_datasets``."""
        if (
            self._filler_by_role is not None
            and self._lhs_by_role is not None
            and self._filler_mask_by_role is not None
        ):
            return
        td = self.training_datasets
        fillers: Dict[int, Dict[int, None]] = {r: {} for r in range(self.nb_rels)}
        lhs_map: Dict[int, Dict[int, None]] = {r: {} for r in range(self.nb_rels)}

        if "gci2" in td and len(td["gci2"]) > 0:
            for row in td["gci2"][:].detach().cpu().tolist():
                a_id, r_id, b_id = int(row[0]), int(row[1]), int(row[2])
                fillers[r_id][b_id] = None
                lhs_map[r_id][a_id] = None

        if "object_property_assertion" in td and len(td["object_property_assertion"]) > 0:
            for row in td["object_property_assertion"][:].detach().cpu().tolist():
                ind_a, r_id, ind_b = int(row[0]), int(row[1]), int(row[2])
                el_a = self.nb_classes + ind_a
                el_b = self.nb_classes + ind_b
                fillers[r_id][el_b] = None
                lhs_map[r_id][el_a] = None

        self._filler_by_role = {
            r: th.tensor(list(vals.keys()), dtype=th.long, device=self.device)
            for r, vals in fillers.items()
        }
        self._lhs_by_role = {
            r: th.tensor(list(vals.keys()), dtype=th.long, device=self.device)
            for r, vals in lhs_map.items()
        }
        filler_mask = th.zeros(
            (self.nb_rels, self.nb_elements), dtype=th.bool, device=self.device
        )
        for r, f_ids in self._filler_by_role.items():
            if f_ids.numel() > 0:
                filler_mask[r, f_ids] = True
        self._filler_mask_by_role = filler_mask

    @property
    def tau_s(self) -> th.Tensor:
        """Return the learnable inductive score temperature :math:`\\tau_s = \\max(\\operatorname{softplus}(\\text{raw}), 10^{-3})`."""
        return self.module.tau_s

    def reset_generator(self, seed: Optional[int] = None) -> th.Generator:
        """Reset and return ``self.generator`` for external training loops.

        Guarantees:

        - Seeds ``self.generator`` with ``seed`` when provided, or ``self.seed + 1337``
          (matching :meth:`train`) when ``seed is None``.

        Non-guarantees:

        - Mutates ``self.generator`` in-place.
        """
        eff_seed = (self.seed + 1337) if seed is None else int(seed)
        self.generator.manual_seed(eff_seed)
        return self.generator

    def scheduled_tau(self, step: int, total_steps: int) -> float:
        """Return geometrically annealed :math:`\\tau` at ``(step, total_steps)`` from ``tau0`` to ``tau1``.

        Guarantees:

        - Pure function of ``(step, total_steps, self.tau0, self.tau1)``.
        - Returns ``self.tau0`` when ``total_steps <= 1`` or ``step <= 0``.
        - Returns ``self.tau1`` at ``step == total_steps - 1`` when ``total_steps > 1``.

        Non-guarantees:

        - Does not mutate ``self.module.tau`` (use :meth:`step_tau` to update in-place).
        """
        _validate_int_ge("step", step, 0)
        _validate_int_ge("total_steps", total_steps, 1)
        if total_steps <= 1:
            return float(self.tau0)
        frac = min(max(step / (total_steps - 1), 0.0), 1.0)
        return float(self.tau0 * ((self.tau1 / self.tau0) ** frac))

    def step_tau(self, step: int, total_steps: int) -> float:
        """Update ``self.module.tau`` according to the geometric annealing schedule and return the new ``tau``."""
        tau_val = self.scheduled_tau(step, total_steps)
        self.module.set_tau(tau_val)
        return tau_val

    def scheduled_lr(
        self,
        step: int,
        total_steps: int,
        lr: Optional[float] = None,
        lr_final: Optional[float] = None,
        lr_schedule: str = "cosine",
    ) -> float:
        """Compute scheduled learning rate at ``(step, total_steps)`` for external training loops."""
        _validate_int_ge("step", step, 0)
        _validate_int_ge("total_steps", total_steps, 1)
        base_lr = self.learning_rate if lr is None else _validate_positive_float("lr", lr)
        eff_final = (
            None
            if lr_final is None
            else _validate_positive_float("lr_final", lr_final)
        )
        if not isinstance(lr_schedule, str) or lr_schedule not in ("cosine", "linear"):
            raise ValueError(
                f"lr_schedule must be 'cosine' or 'linear', got {lr_schedule!r}"
            )
        return self._scheduled_lr(step, total_steps, base_lr, eff_final, lr_schedule)

    def clamp_dead_logits(self) -> None:
        """Clamp ``self.module.dead_logits`` to ``[0, 1]`` and keep ``dead_logits[bot_id] == +100.0``."""
        self.module.elements.clamp_dead(self.bot_id, _DEAD_LOGIT)

    def project_parameters(self) -> None:
        """The projection after an optimiser step (:class:`~mowl.training.sampled_objective.SampledObjectiveTraining`):
        :meth:`clamp_dead_logits`."""
        self.clamp_dead_logits()

    def dual_update(self, mean_losses: Mapping[str, float]) -> Dict[str, float]:
        """One dual-ascent step: w_t <- min(dual_max, w_t + dual_rho * max(0, mean_losses[t] - dual_target)) for
        t in ("fact", "axioms", "region"); a no-op in "fixed" mode. Returns the (new) weights.

        Guarantees:

        - Reasoner-free (R1).
        - In ``constraint_mode == "fixed"``, leaves :attr:`dual_weights` unchanged
          at ``1.0`` and returns them.
        - In ``constraint_mode == "lagrangian"``, updates each ``t`` in
          ``("fact", "axioms", "region")`` by
          ``min(dual_max, w_t + dual_rho * max(0.0, mean_losses[t] - dual_target))``
          and caps at ``self.dual_max``.

        Non-guarantees:

        - Mutates :attr:`dual_weights` in-place when ``constraint_mode == "lagrangian"``.

        Parameters
        ----------
        mean_losses : Mapping[str, float]
            Per-term mean losses mapping ``"fact"``, ``"axioms"``, and ``"region"``
            to numeric values (or 0-D scalar tensors).

        Returns
        -------
        Dict[str, float]
            The (new) ``{"fact": ..., "axioms": ..., "region": ...}`` dual weights.
        """
        if not isinstance(mean_losses, Mapping):
            raise TypeError(
                f"mean_losses must be a Mapping[str, float], got {type(mean_losses).__name__}"
            )
        if self.constraint_mode == "fixed":
            for t in ("fact", "axioms", "region"):
                self.dual_weights[t] = 1.0
            return dict(self.dual_weights)
        from mowl.training.constraints import dual_ascent_step
        dual_ascent_step(self.dual_weights, mean_losses, rho=self.dual_rho,
                         max_weight=self.dual_max, target=self.dual_target)
        return dict(self.dual_weights)

    def reset_dual_weights(self) -> Dict[str, float]:
        """Reset :attr:`dual_weights` to ``{"fact": 1.0, "axioms": 1.0, "region": 1.0}`` and return them.

        Guarantees:

        - Resets ``self.dual_weights["fact"] = self.dual_weights["axioms"] = self.dual_weights["region"] = 1.0``.

        Non-guarantees:

        - Mutates :attr:`dual_weights` in-place.

        Returns
        -------
        Dict[str, float]
            Reset ``{"fact": 1.0, "axioms": 1.0, "region": 1.0}`` dictionary.
        """
        self.dual_weights = {
            "fact": 1.0,
            "axioms": 1.0,
            "region": 1.0,
        }
        return dict(self.dual_weights)

    def _encode_features(self, features: th.Tensor) -> th.Tensor:
        """Validate ``features`` and map through ``self.module.encoder`` to ``(B, embed_dim)`` positions."""
        mod = self.module
        enc = mod.encoder if mod.encoder is not None else self.encoder
        if enc is None:
            raise ValueError(
                "GroundedEL.encoder is None; pass encoder=... at construction to use compute_data_loss or score"
            )
        if mod.encoder is None:
            mod.encoder = enc.to(self.device)
        if not isinstance(features, th.Tensor):
            raise TypeError(
                f"features must be a torch.Tensor, got {type(features).__name__}"
            )
        if features.ndim == 1:
            feat_2d = features.unsqueeze(0)
        elif features.ndim == 2:
            feat_2d = features
        else:
            raise ValueError(
                f"features must be 2-D (B, feat_dim) (or 1-D (feat_dim,)), got shape {tuple(features.shape)}"
            )
        feat_dev = feat_2d.to(device=self.device, dtype=mod.class_centers.dtype)
        pos = enc(feat_dev)
        if pos.ndim != 2 or pos.shape[0] != feat_dev.shape[0] or pos.shape[1] != self.embed_dim:
            raise ValueError(
                f"encoder(features) must return shape ({feat_dev.shape[0]}, {self.embed_dim}), got {tuple(pos.shape)}"
            )
        return pos

    def _chunked_term_scores(
        self,
        pos: th.Tensor,
        term_ids: th.Tensor,
        max_elements: int = 8_388_608,
    ) -> th.Tensor:
        """Compute ``sigmoid(depth(pos, term_ids, crisp=True) / tau_s)`` in term chunks bounded by ``max_elements``."""
        return SoftClassScores(max_elements)(self.module, pos, term_ids, self.module.tau_s)

    def compute_data_loss(
        self,
        features: th.Tensor,
        label_matrix: th.Tensor,
        term_class_ids: Union[Sequence[int], th.Tensor],
        max_elements: int = 8_388_608,
    ) -> Tuple[th.Tensor, th.Tensor]:
        """Compute inductive binary cross-entropy loss and scores for encoded data elements (§6 / E3).

        Computes ``positions = self.encoder(features)`` (where ``self.encoder`` is a
        trainable submodule registered on ``self.module`` and part of the model's
        parameters/optimizer), crisp box depths ``depth(positions, term_class_ids, crisp=True)``,
        and scores ``s = sigmoid(depth(positions, term_class_ids, crisp=True) / tau_s)``
        with learnable scalar temperature ``tau_s = max(softplus(raw_tau_s), 1e-3)``
        (initialised to ``0.1``). Returns ``(BCE(s, label_matrix), s)`` with mean
        reduction.

        Data elements never get a dead score (:math:`\\delta \\equiv 0`) or a
        minimality penalty on their positions: there are no data elements in
        :math:`\\Delta` here; proteins/data instances exist only through
        ``self.encoder`` at forward/scoring time.

        Guarantees:

        - Reasoner-free (R1): never imports or calls any reasoner or completion algorithm.
        - Chunks computation over ``term_class_ids`` whenever
          ``B * n_terms * K * d > max_elements`` so peak intermediate memory stays
          bounded even on large ontologies (e.g. GO BP with ``128 * 21356 * K * d``).
        - Propagates gradients to ``self.encoder``, ``self.module.raw_tau_s``, and
          the queried class boxes in ``self.module``.
        - Never creates or reads ``dead_logits`` or minimality terms for the encoded
          data positions.

        Non-guarantees:

        - Under ``crisp=True`` (hard ``max`` over active component slots), box
          gradients for each ``(protein, term)`` pair route only to the single
          active component slot attaining the maximum depth.

        Parameters
        ----------
        features : torch.Tensor
            Input feature matrix of shape ``(B, feat_dim)``.
        label_matrix : torch.Tensor
            Binary target matrix of shape ``(B, n_terms)`` with entries in ``[0, 1]``.
        term_class_ids : Union[Sequence[int], torch.Tensor]
            1-D sequence or tensor of ``n_terms`` class indices in
            ``[0, nb_classes)`` (e.g. the ``Aux_G`` class indices).
        max_elements : int, default 8_388_608
            Maximum ``B * term_chunk * K * d`` elements per chunk before splitting
            across ``term_class_ids``.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            ``(loss, scores)`` where ``loss`` is the mean binary cross-entropy
            scalar tensor and ``scores`` has shape ``(B, n_terms)``.
        """
        _validate_int_ge("max_elements", max_elements, 1)
        pos = self._encode_features(features)
        bsz = int(pos.shape[0])
        term_ids = self.module._to_index_tensor(
            term_class_ids, self.nb_classes, "term_class_ids"
        )
        if term_ids.ndim != 1 or term_ids.numel() == 0:
            raise ValueError(
                f"term_class_ids must be a non-empty 1-D sequence/tensor, got shape {tuple(term_ids.shape)}"
            )
        n_terms = int(term_ids.numel())
        if not isinstance(label_matrix, th.Tensor):
            raise TypeError(
                f"label_matrix must be a torch.Tensor, got {type(label_matrix).__name__}"
            )
        if label_matrix.shape != (bsz, n_terms):
            raise ValueError(
                f"label_matrix must have shape ({bsz}, {n_terms}), got {tuple(label_matrix.shape)}"
            )
        targets = label_matrix.to(device=self.device, dtype=pos.dtype)
        return MembershipCrossEntropy(SoftClassScores(max_elements))(
            self.module, pos, term_ids, targets, self.module.tau_s)

    @th.no_grad()
    def score(
        self,
        features: th.Tensor,
        term_class_ids: Union[Sequence[int], th.Tensor],
        max_elements: int = 8_388_608,
    ) -> th.Tensor:
        """Compute inductive prediction scores ``sigmoid(depth(encoder(features), term_class_ids, crisp=True) / tau_s)`` without gradients.

        Data elements never get a dead score (:math:`\\delta \\equiv 0`) or a
        minimality penalty on their positions: there are no data elements in
        :math:`\\Delta` here; proteins/data instances exist only through
        ``self.encoder``.

        Guarantees:

        - Runs under :func:`torch.no_grad` and returns a tensor of shape
          ``(B, n_terms)`` (or ``(1, n_terms)`` when ``features`` is 1-D) with
          values in ``(0, 1)``.
        - Chunks over ``term_class_ids`` when ``B * n_terms * K * d > max_elements``.

        Non-guarantees:

        - Does not toggle ``self.module.eval()``; callers using dropout/batchnorm in
          ``self.encoder`` should set ``model.module.eval()`` during evaluation.

        Parameters
        ----------
        features : torch.Tensor
            Input feature matrix of shape ``(B, feat_dim)`` (or 1-D ``(feat_dim,)``).
        term_class_ids : Union[Sequence[int], torch.Tensor]
            1-D sequence or tensor of ``n_terms`` class indices in ``[0, nb_classes)``.
        max_elements : int, default 8_388_608
            Maximum ``B * term_chunk * K * d`` elements per chunk.

        Returns
        -------
        torch.Tensor
            Predicted score matrix of shape ``(B, n_terms)``.
        """
        _validate_int_ge("max_elements", max_elements, 1)
        pos = self._encode_features(features)
        term_ids = self.module._to_index_tensor(
            term_class_ids, self.nb_classes, "term_class_ids"
        )
        if term_ids.ndim != 1 or term_ids.numel() == 0:
            raise ValueError(
                f"term_class_ids must be a non-empty 1-D sequence/tensor, got shape {tuple(term_ids.shape)}"
            )
        return self._chunked_term_scores(pos, term_ids, max_elements=max_elements)

    def _compute_slot_live_counts(
        self,
        regions: Regions,
        live_pos: th.Tensor,
        query_slot_mask: th.Tensor,
        chunk_size: int = 8192,
    ) -> th.Tensor:
        """Exact count of live elements crisp-inside each queried ``(c_id, k)`` slot
        (:func:`~mowl.representation.library.union_reseeding.slot_point_counts`)."""
        return slot_point_counts(regions, live_pos, query_slot_mask, chunk_size=chunk_size)

    def reseed_idle(
        self,
        generator: Optional[th.Generator] = None,
        reseed_size: float = 0.08,
        prune_unused: bool = True,
    ) -> int:
        """Relocate idle/least-used class components to uncovered own elements and prune empty slots (§5).

        Performs ONLY:

        1. For each live class :math:`A` whose own element :math:`x_A` is not
           crisp-inside :math:`A`: places one component box of half-width
           ``reseed_size`` at :math:`p_{x_A}`, using an inactive slot if any,
           else the active slot that crisp-contains the fewest live elements
           (breaking ties by smallest volume); same for ABox class assertions
           :math:`C(a)` (element of individual :math:`a`).
        2. When ``prune_unused=True``: deactivates active components that
           crisp-contain no live element (inspecting element positions only).

        Guarantees:

        - Reasoner-free (R1): never reads ``gci0``, ``gci1``, ``gci2``, ``gci3``,
          ``gci0_bot``, ``gci1_bot``, ``gci3_bot``, ``role_inclusion``,
          ``role_chain``, or ``object_property_assertion`` tables.
        - Never modifies ``elem_pos.weight`` or ``dead_logits``, and never
          rebuilds whole classes.
        - Computes per-class live element depths in chunks of at most 8192
          elements (:math:`O(|\\Delta| \\cdot K \\cdot d)` memory per chunk; no
          :math:`|\\Delta|^2` tensor anywhere).
        - Returns the total number of modified component slots.

        Non-guarantees:

        - Does not reset Adam optimiser moments directly; :meth:`train` calls
          ``optimizer.state.clear()`` after invoking :meth:`reseed_idle`.

        Parameters
        ----------
        generator : Optional[torch.Generator], default None
            Optional PyTorch RNG (retained for signature compatibility).
        reseed_size : float, default 0.08
            Positive half-width assigned to newly re-seeded component boxes.
        prune_unused : bool, default True
            Whether to deactivate active component slots that crisp-contain zero
            live elements.

        Returns
        -------
        int
            Number of modified component slots.
        """
        rule = OwnPointReseeding(size=reseed_size, prune=prune_unused)
        assertions = None
        if self.nb_inds > 0:
            td = self.training_datasets
            if "class_assertion" in td and len(td["class_assertion"]) > 0:
                assertions = td["class_assertion"][:]
        moves = rule.apply(self.module, assertions, top_id=self.top_id, bot_id=self.bot_id)
        self.module.last_reseed_count = int(moves)
        return moves

    @property
    def eval_gci_name(self) -> Optional[str]:
        """Return the GCI / assertion name used for validation and evaluation."""
        return self._eval_gci_name

    @eval_gci_name.setter
    def eval_gci_name(self, value: str) -> None:
        valid_gci_names = ["gci0", "gci1", "gci2", "gci3", "class_assertion"]
        if self._extended:
            valid_gci_names.extend(["gci0_bot", "gci1_bot", "gci3_bot"])
        if value not in valid_gci_names:
            raise ValueError(
                f"eval_gci_name must be one of {valid_gci_names}, got {value!r}"
            )
        self._eval_gci_name = value

    @staticmethod
    def _scheduled_lr(
        epoch: int,
        epochs: int,
        lr: float,
        lr_final: Optional[float],
        lr_schedule: str = "cosine",
    ) -> float:
        """Compute per-epoch learning rate as a pure function of ``(epoch, epochs, lr, lr_final, lr_schedule)``.

        Guarantees:

        - Pure function of its arguments; independent of optimiser state or
          idle-component re-seeding.
        - Returns ``lr`` when ``lr_final is None``, when ``epoch == 0``, or when
          ``epochs <= 1``.
        - Returns ``lr_final`` (to floating-point precision) at
          ``epoch == epochs - 1`` when ``epochs > 1`` and ``lr_final is not None``.

        Non-guarantees:

        - Assumes caller has already validated ``lr_final`` and ``lr_schedule``.
        """
        if lr_final is None or epochs <= 1:
            return float(lr)
        t = epoch / (epochs - 1)
        if lr_schedule == "cosine":
            w = 0.5 * (1.0 + math.cos(math.pi * t))
            return float(w * lr + (1.0 - w) * lr_final)
        return float((1.0 - t) * lr + t * lr_final)

    def train(
        self,
        epochs: Optional[int] = None,
        validate_every: int = 1,
        epoch_callback: Optional[Callable[[int, "GroundedEL"], None]] = None,
        log_every: Optional[int] = None,
        lr_final: Optional[float] = None,
        lr_schedule: str = "cosine",
    ) -> List[Dict[str, float]]:
        """Run the GroundedEL training loop (§3, §4, §5).

        Each epoch:

        1. Anneals :math:`\\tau` geometrically from ``tau0`` to ``tau1`` and, when
           ``lr_final is not None``, updates the Adam learning rate across
           ``optimizer.param_groups`` from ``learning_rate`` to ``lr_final``
           according to ``lr_schedule`` (``"cosine"`` or ``"linear"``).
        2. Computes ``L = L_fact + L_axioms + lambda_reg * L_region + L_minimality``
           via :meth:`compute_batch_losses` and performs one Adam step.
        3. Optionally calls :meth:`reseed_idle` when ``reseed_every > 0`` and
           ``(epoch + 1) % reseed_every == 0`` (never at the last epoch unless
           ``(epoch + 1) % reseed_every == 0``), followed by
           ``optimizer.state.clear()`` to reset Adam moments.
        4. Logs per-term losses every ``log_every`` epochs.
        5. Invokes ``epoch_callback(epoch, self)`` when provided, and performs
           validation / checkpoint saving every ``validate_every`` epochs when
           configured.

        The training design is a :class:`mowl.training.TrainingRegime`. Without one
        assigned (``model.training_regime is None``), the model uses
        :meth:`mowl.training.GroundedELTraining.for_model`: temperature annealing, the
        optional learning-rate decay, idle-component re-seeding and Lagrangian dual
        weights, each a shared regime configured by the model's hyperparameters and
        the arguments below. An assigned regime replaces that composition as a whole
        (``lr_final`` and ``lr_schedule`` are then not used); its hooks
        ``optimizer``, ``on_epoch_start``, ``negatives`` and ``filter_negatives`` (for
        the corrupted object property assertions, when ``neg_weight > 0``),
        ``additional_loss``, ``on_epoch_end``, ``should_stop`` and ``on_train_end``
        are honoured. The loss itself (:meth:`compute_batch_losses`) and the
        projection of the dead scores are part of the method.

        Guarantees:

        - Reasoner-free (R1): never imports or calls any reasoner or semantic
          completion algorithm.
        - Deterministic for a fixed ``self.seed``.
        - When ``lr_final is None``, keeps ``learning_rate`` constant across all
          epochs. When ``lr_final is not None``, computes the epoch learning rate
          as a pure function of ``(epoch, epochs, learning_rate, lr_final)`` so
          :meth:`reseed_idle` never resets the learning-rate schedule.
        - Calls :meth:`reseed_idle` if and only if ``reseed_every > 0`` and
          ``(epoch + 1) % reseed_every == 0``.
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
            Validation / checkpoint interval in epochs (:math:`\\ge 1`). When
            ``validate_every > epochs``, end-of-loop checkpoint overwriting is
            suppressed if ``epoch_callback`` has already written a checkpoint to
            :attr:`model_filepath`.
        epoch_callback : Optional[Callable[[int, GroundedEL], None]], default None
            Optional callback ``epoch_callback(epoch, model)`` invoked at the end
            of each epoch.
        log_every : Optional[int], default None
            Logging interval in epochs; defaults to ``self.log_every`` when ``None``.
        lr_final : Optional[float], default None
            Optional final learning rate :math:`> 0` at the last epoch. ``None``
            keeps a constant learning rate equal to ``self.learning_rate``.
        lr_schedule : {"cosine", "linear"}, default "cosine"
            Learning-rate decay schedule used when ``lr_final is not None``.

        Returns
        -------
        List[Dict[str, float]]
            Per-epoch recorded loss dictionaries.
        """
        eff_epochs = self.epochs if epochs is None else _validate_int_ge("epochs", epochs, 0)
        eff_val_every = _validate_int_ge("validate_every", validate_every, 1)
        eff_log_every = (
            self.log_every
            if log_every is None
            else _validate_int_ge("log_every", log_every, 0)
        )
        eff_lr_final: Optional[float] = (
            None
            if lr_final is None
            else _validate_positive_float("lr_final", lr_final)
        )
        if not isinstance(lr_schedule, str) or lr_schedule not in ("cosine", "linear"):
            raise ValueError(
                f"lr_schedule must be 'cosine' or 'linear', got {lr_schedule!r}"
            )
        self.loss_history = []
        if eff_epochs == 0:
            return self.loss_history
        # The training design is a regime: the one assigned to the model, or the
        # composition of shared regimes given by the model's hyperparameters; the loop is
        # the one of every sampled objective.
        from mowl.training.grounded import GroundedELTraining
        from mowl.training.sampled_objective import SampledObjectiveTraining
        regime = self.training_regime
        if regime is None:
            regime = GroundedELTraining.for_model(
                self, lr_final=eff_lr_final, lr_schedule=lr_schedule)
            regime.attach(self)
        return SampledObjectiveTraining(seed_offset=1337).run(
            self, regime, eff_epochs, validate_every=eff_val_every,
            epoch_callback=epoch_callback, log_every=eff_log_every)

    def epoch_record(self, rec: Dict[str, float]) -> None:
        """Fields of the epoch record besides the loss terms: the dual weights in
        ``constraint_mode="lagrangian"``."""
        if self.constraint_mode == "lagrangian":
            rec["w_fact"] = float(self.dual_weights["fact"])
            rec["w_axioms"] = float(self.dual_weights["axioms"])
            rec["w_region"] = float(self.dual_weights["region"])

    def log_epoch(self, epoch: int, epochs: int, rec: Mapping[str, float]) -> None:
        """Log the terms of an epoch record."""
        logger.info(
            "Epoch %d/%d (tau=%.4f): total=%.4e fact=%.4e axioms=%.4e region=%.4e min=%.4e neg=%.4e w=(%.2f,%.2f,%.2f)",
            epoch + 1, epochs, rec["tau"], rec["total"], rec["fact"], rec["axioms"],
            rec["region"], rec["minimality"], rec["neg"], rec["w_fact"], rec["w_axioms"],
            rec["w_region"])

    def score_query(
        self,
        structure: Any,
        queries: Sequence[Any],
        method: str = "beam",
        **kwargs: Any,
    ) -> th.Tensor:
        """Score every individual as an answer of logical queries: the protocol of
        :class:`mowl.evaluation.LogicalQueryAnswering`.

        Anchors and answers are individuals (indices of ``individual_index_dict``),
        roles are indices of ``object_property_index_dict``.

        :param structure: Query structure, e.g. ``("e", ("r", "r"))`` for two hops
            (:data:`mowl.evaluation.QUERY_STRUCTURES`).
        :param queries: Instances of ``structure``.
        :param method: ``"beam"`` (max-min search over witness elements,
            :func:`mowl.nn.el.grounded.queries.score_query_beam`) or ``"box"`` (the query
            compiled to a union of boxes, :func:`~mowl.nn.el.grounded.queries.compile_query`
            and :func:`~mowl.nn.el.grounded.queries.score_entities`).
        :param kwargs: Passed to the scorer (``k`` for the beam; ``max_boxes``,
            ``exact``, ``max_witnesses`` for the boxes).
        :return: Scores of shape ``(len(queries), number of individuals)``; higher is
            better, positive means inside the query region.

        Needs ``role_mode="translation"``.
        """
        from mowl.nn.el.grounded import queries as Q

        if method not in ("beam", "box"):
            raise ValueError(f"method must be 'beam' or 'box', got {method!r}")
        device = self.module.elem_pos.weight.device
        individuals = self.nb_classes + th.arange(self.nb_inds, dtype=th.long, device=device)
        with th.no_grad():
            if method == "beam":
                return Q.score_query_beam(
                    self.module, structure, list(queries),
                    entity_to_element=individuals,
                    candidate_element_ids=individuals, **kwargs)
            region = Q.compile_query(
                self.module, structure, list(queries),
                entity_to_element=individuals, **kwargs)
            return Q.score_entities(self.module, region, candidate_element_ids=individuals)

    def reseed_components(self, epoch: int = 0) -> int:
        """Re-seed idle components (:meth:`reseed_idle` with pruning): the protocol of
        :class:`mowl.training.IdleComponentReseeding`. Inside :meth:`train` the
        training generator is used, so a run stays a function of the seed."""
        gen = getattr(self, "_train_generator", None)
        return self.reseed_idle(generator=gen, prune_unused=True)
