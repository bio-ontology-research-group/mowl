# Learnable geometric models: exact initialisation (R1) and MultiBoxEL (R2) — design

Date: 2026-10-29. Author: Claude (design, control, verification) for Robert Hoehndorf.
Status: v1, approved for execution by Robert's instruction ("You design the experiments,
agy-auto runs them, you check, validate, interpret").

Companion documents: `2026-10-29-geometric-models-design.md` (v4, the `mowl.geometric`
package this builds on) and `notes/learnable-methods.md` in the paper repository
(routes R1–R4; this spec implements R1 and R2 and the evaluation protocol).

## 1. Intent

The exact constructions in `mowl.geometric` are strongly faithful and predict nothing
beyond the entailed axioms. The box-based learnable methods (ELBE/ELBox, Box²EL) sit at
the opposite corner: they generalise but, by the Helly obstruction, cannot be
non-degenerate models of ontologies containing a certificate triangle (three names whose
pairwise conjunctions are named and satisfiable but whose triple is unsatisfiable, the
T₃ pattern). We want learnable methods that sit on the trade-off curve and an experiment
that measures where each method sits:

* **R1 — exact initialisation / anchor.** Initialise the class boxes of ELBE and Box²EL
  from the exact convex polyhedral model (random projection of the canonical-model
  element points, bounding box per class), optionally pulling the boxes back towards that
  initialisation during training. Hypothesis: equal ranking quality, much better
  faithfulness ("model-ness", Helly-violation rate, ordering of entailed/undetermined/
  contradicted axioms).
* **R2 — MultiBoxEL.** Box²EL semantics with every class represented by a union of
  `k` axis-parallel boxes. `k` is the relaxation knob: `k = 1` is Box²EL exactly;
  `k ≥ 2` escapes the Helly obstruction (multiboxes are complete for ELHO(∘)⊥,
  Federico, Leemhuis & Kutz, JOWO 2026, who list differentiable bounded multiboxes as
  open).
* **Experiments.** A learning-free diagnostic (how many certificate triangles exist in
  Pizza, FoodOn, GO), a synthetic Helly stress suite, and the three subsumption benchmarks
  (Pizza-EL built here; GO and FoodOn as shipped by mOWL), reporting ranking, faithfulness
  and cost metrics side by side for ELBE, Box²EL (random and exact init, with/without
  anchor) and MultiBoxEL (`k ∈ {1,2,3,4}`).

Non-goals: prisms (R3), the canonical-element mixture model (R4), role-region
initialisation from the canonical relation, hyper-parameter search beyond the small grid
in §6, ABox/individual multiboxes (individuals stay single boxes), GPU multi-node runs.

Anti-goals: changing the published semantics of Box²EL or ELBE (their losses are not
edited; new behaviour is opt-in via new keyword arguments); reporting any number that was
not produced by the code in the repositories; silently dropping a dataset or a
configuration from the sweep.

## 2. Repositories, branches, environment

* mOWL code: worktree `~/Public/software/mowl-wt/geo`, branch `learnable-geometric`
  created from `geometric-models-rebased` (= PR #167 rebased onto upstream `main`,
  b83c067). PR target: `main`, after PR #167.
* Experiment code and results: `~/Public/software/faithful-dl-embedding`, branch
  `experiments`, directory `experiments/` (public repository; no data files larger than
  1 MB are committed; raw ontologies live in `~/tmp/gfe-data/`).
* Python: `~/.venvs/mowl-gpu` (Python 3.11, torch cu128, mOWL editable from the
  worktree). GPU: the laptop's RTX PRO 2000 (8 GB). Never write to `/tmp`.
* Implementers: agy-auto (Gemini 4 Argon) per task; reviewers: Sonnet/Opus subagents;
  design, rulings and final verification: the controlling session.

## 3. R1 — `mowl.geometric.init`

### 3.1 `ExactBoxInit`

```python
@dataclass(frozen=True)
class ExactBoxInit:
    centers: np.ndarray    # (n_classes, k, d) float32; NaN rows = "no initialisation"
    offsets: np.ndarray    # (n_classes, k, d) float32, >= 0 where not NaN
    class_to_id: Dict[str, int]
    method: str            # "projected" | "milp"
    n_components: int      # k
    seed: Optional[int]
    report: Dict[str, Any] # sizes, timings, fallbacks, n_initialised, n_skipped

    @classmethod
    def from_canonical_model(cls, cm, class_to_id, embed_dim, *, n_components=1,
                             method="projected", seed=0, pad=0.05, scale=1.0,
                             milp_time_limit=60.0, milp_d_max=None) -> "ExactBoxInit"
    @classmethod
    def from_dataset(cls, dataset, embed_dim, **kw) -> "ExactBoxInit"   # builds cm
    def apply_to_module(self, module, *, device=None) -> int   # returns rows written
    def anchor_loss(self, module) -> torch.Tensor               # unweighted
    def as_tensors(self, device=None) -> Tuple[Tensor, Tensor, Tensor]  # centers, offsets, mask
```

`class_to_id` is `dataset.class_to_id` (string IRI → row index, the module's embedding
row order). `cm.names` uses the same IRIs; `owl:Thing`/`owl:Nothing` are `TOP`/`BOTTOM`.

### 3.2 Projected polyhedral initialisation (`method="projected"`)

The exact convex polyhedral model places canonical element `e` at simplex vertex
`v_e ∈ R^{n-1}`; a class is the face spanned by its extension. We project with a Gaussian
map: element `e` gets `p_e = scale · g_e / sqrt(d)` with `g_e ~ N(0, I_d)` drawn from
`numpy.random.default_rng(seed)`, in the order of `cm.names`. For a class `C` with
extension `E_C` (`cm.extension(C)`, element indices):

* `k = 1`: `lo = min_{e∈E_C} p_e − pad`, `hi = max_{e∈E_C} p_e + pad` (componentwise);
  `center = (lo+hi)/2`, `offset = (hi−lo)/2`.
* `k > 1`: k-means (`k' = min(k, |E_C|)`, deterministic: `scipy.cluster.vq.kmeans2`
  with `seed`, `minit="++"`) on `{p_e}`; one padded bounding box per cluster; the
  remaining `k − k'` components duplicate the first box (so `k` components are always
  defined and `k' = 1` reproduces the `k = 1` box in every component).
* `TOP`: bounding box of all element points (padded). `BOTTOM`: centre `0`, offset `0`.
* Unsatisfiable names (empty extension, not `BOTTOM`): centre `0`, offset `0`.
* Classes in `class_to_id` absent from `cm.names`: NaN rows (left untouched by
  `apply_to_module`; counted in `report["n_skipped"]`).

Guarantee (tested): with `k = 1`, for every entailed atomic subsumption `C ⊑ D` with `C` satisfiable
(`cm.entails(C, D)`), `box_C ⊆ box_D` exactly, because `E_C ⊆ E_D` and padded bounding
boxes are monotone; hence every `gci0` training axiom has zero Box²EL/ELBE loss at
margin 0 immediately after initialisation. Disjointness is **not** guaranteed (bounding
boxes overlap); that is the point of the experiment. Axioms whose subclass is unsatisfiable are not covered (unsatisfiable names are points at the origin), and no placement can cover them in general.

### 3.3 MILP initialisation (`method="milp"`)

`BoxModelBuilder(cm, d_max=milp_d_max or embed_dim, time_limit=milp_time_limit).build()`;
the resulting `d'`-dimensional boxes occupy coordinates `0..d'−1` unchanged (no padding
there: padding would break the disjointness the MILP established); coordinates
`d'..d−1` hold one interval shared by every class, `[−1−pad, 1+pad]`, so containment and
disjointness of the box model are preserved in `R^d`.
On `NotBoxRepresentable` or `BoxModelNotFound` the constructor falls back to
`"projected"` and records `report["fallback"] = "<exception class>: <message>"`.
`n_components > 1` with `method="milp"` duplicates the single box into all components.

### 3.4 Applying and anchoring

`apply_to_module(module)` writes into `module.class_center.weight` / `module.class_offset.weight`
(Box²EL, MultiBoxEL; for MultiBoxEL the embedding row is the flattened `(k, d)` block) or
`module.class_embed.weight` / `module.class_offset.weight` (ELBE), for rows that are not
NaN, under `torch.no_grad()`; raises `UnsupportedRegion` if the module exposes neither
pair or if the module's `k`/`d` do not match. ELBE stores the *half-width* in
`class_offset` exactly like Box²EL, so the same `(center, offset)` is written.

`anchor_loss(module)` = mean over initialised rows of
`‖center − center₀‖² + ‖|offset| − offset₀‖²` (absolute value because both modules read
offsets through `abs`). Models multiply it by `anchor_weight`.

### 3.5 Model-side hooks

`BoxSquaredEL`, `ELBE` and `MultiBoxEL` gain keyword arguments
`init_boxes: Optional[ExactBoxInit] = None` and `anchor_weight: float = 0.0`.
`init_module()` applies `init_boxes` after constructing the module;
`get_regularization_loss()` adds `anchor_weight * init_boxes.anchor_loss(self.module)`
to whatever the base returns. Default behaviour (both unset) is unchanged, byte for byte.

## 4. R2 — MultiBoxEL

### 4.1 Parameters (`mowl/nn/el/multiboxel/module.py`, class `MultiBoxELModule(ELModule)`)

```python
MultiBoxELModule(nb_ont_classes, nb_rels, nb_inds=None, embed_dim=50, n_components=2,
                 gamma=0.0, delta=2.0, reg_factor=0.05, component_reg=0.0)
```

* `class_center`, `class_offset`: `nn.Embedding(nb_ont_classes, n_components * embed_dim)`,
  viewed as `(B, k, d)`; initialised like Box²EL (uniform(−1,1), row-normalised per
  component).
* `bump_classes`: `(nb_ont_classes, d)` — one bump per class, shared by its components
  (keeps `k = 1` identical to Box²EL and keeps role semantics simple).
* Roles: `head_center/head_offset/tail_center/tail_offset` `(nb_rels, d)` as Box²EL.
* Individuals (if `nb_inds`): single boxes and bumps exactly as Box²EL.
* `neg_capable_gcis = {"gci2", "object_property_assertion"}`; `role_axiom_capable = False`.
* `effective_components(eps=1e-3) -> Tensor[nb_ont_classes]`: number of components whose
  offsets are all `> eps` **and** whose box is not contained in another component of the
  same class (containment checked exactly on corners); used for cost reporting.

### 4.2 Losses (`mowl/nn/el/multiboxel/losses.py`)

Notation: boxes `X_i = (c_i, |o_i|)`, `i ∈ 1..k`; `incl(X, Y) = ‖relu(|c_X − c_Y| − o_X − o_Y + 2 o_X − γ)‖₂`
(Box²EL's `inclusion_score`), computed on broadcast shapes `(B, k_X, k_Y)`;
`empt(X, Y) = ‖relu(lo − up)‖₂` with `lo = max(c_X − o_X, c_Y − o_Y)`, `up = min(c_X + o_X, c_Y + o_Y)`;
`I(X, Y) = ((lo+up)/2, |up−lo|/2)` (Box²EL's phantom intersection); `md(X, Y) = ‖relu(dist(X,Y) + γ)‖₂`.

| GCI | Loss (per batch row) | `k = 1` reduces to Box²EL? |
|---|---|---|
| gci0 `C ⊑ D` | `mean_i min_j incl(C_i, D_j)` | yes |
| gci0_bot `C ⊑ ⊥` | `mean_i ‖o_{C_i}‖₂` | yes |
| gci1 `C ⊓ D ⊑ E` | let `(i*, j*) = argmin_{ij} empt(C_i, D_j)` (no gradient through the choice); `g_ij = 1` if `lo_ij ≤ up_ij` in every coordinate or `(i,j) = (i*,j*)`, else `0` (detached); loss `= empt(C_{i*}, D_{j*}) + Σ_ij g_ij · min_m incl(I(C_i, D_j), E_m) / Σ_ij g_ij` | yes (single pair is always designated) |
| gci1_bot `C ⊓ D ⊑ ⊥` | `mean_ij ‖relu(−dist(C_i, D_j) − γ)‖₂` | yes |
| gci2 `C ⊑ ∃r.D` | `(mean_i incl(C_i + b_D, H_r) + min_j incl(D_j + b_C, T_r)) / 2` | yes |
| gci2 negative | `min_i (δ − md(C_i + b_D, H_r))² + min_j (δ − md(D_j + b_C, T_r))²` | yes |
| gci3 `∃r.C ⊑ D` | `min_m incl(H_r − b_C, D_m)` | yes |
| gci3_bot `∃r.C ⊑ ⊥` | `‖o_{H_r}‖₂` | yes |
| class_assertion `C(a)` | `min_m incl(a, C_m)` | yes |
| object_property_assertion | Box²EL's function unchanged (individuals are single boxes) | yes |

Rationale for gci1: every pair of components whose boxes meet imposes an inclusion on
the intersection; the designated pair is additionally pushed to be non-empty so that
`C ⊓ D` does not collapse (Box²EL's `additional_loss`), and for that pair the inclusion
is always enforced, which is what makes `k = 1` coincide with Box²EL. Pairs that are
empty impose nothing — that is exactly the freedom that lets a multibox satisfy T₃.

`regularization_loss()` = Box²EL's bump regulariser (`reg_factor * ‖bump_classes‖` as in
upstream `BoxSquaredELModule.regularization_loss`) `+ component_reg * mean_C mean_{i<j} relu(τ − ‖c_{C,i} − c_{C,j}‖₂)`
with `τ = 0.1` (anti-collapse; default `component_reg = 0.0`, i.e. off; unused
components are allowed to collapse and are reported through `effective_components`).

The `k = 1` identities are **tested**: construct a `BoxSquaredELModule` and a
`MultiBoxELModule(n_components=1)` with copied weights and assert all ten losses agree to
`1e-6` on random batches, positive and negative where applicable, on a batch that
contains both empty and non-empty intersections.

### 4.3 Model (`mowl/models/multiboxel/model.py`, class `MultiBoxEL(EmbeddingELModel)`)

Mirror of upstream `BoxSquaredEL` with extra `n_components=2`, `component_reg=0.0`,
`init_boxes=None`, `anchor_weight=0.0`; exported from `mowl.models` and `mowl.nn`.
Scoring for evaluators is `module(batch, "gci0")` etc. (lower is better) as for Box²EL.

### 4.4 Solvability test (slow)

Train `MultiBoxEL(n_components=2, embed_dim=2)` on `tests/geometric/fixtures/t3.owl`
(full batch, Adam, lr 0.01, 1500 epochs, seed 0, γ = 0) and assert that the summed
training loss over all normal forms is `< 1e-4` and that `helly_violation_rate` (§5.3)
is `0` while `degeneracy_rate` is `0`; train `BoxSquaredEL` identically and assert that
its loss is `> 1e-3` (it must either leave a Helly triple non-empty or collapse a pairwise
conjunction). This is the mechanised form of the paper's T₃ claim at the level of the
learnable models.

## 5. Experiments (`faithful-dl-embedding/experiments/`)

### 5.1 Datasets

| id | source | task | notes |
|---|---|---|---|
| `tk3`…`tk6` | generated by `experiments/data/make_synthetic.py` | satisfaction only | T_k: names `X_1..X_k`, for each `i` a name `A_i ≡ ⊓_{j≠i} X_j` (binary-normalised with auxiliary names), `A_i ⊓ A_j ⊑ ⊥` for all `i<j`; plus one unconstrained name `Z`. No held-out axioms. |
| `pizza` | `~/tmp/gfe-data/pizza/pizza.owl` → `experiments/data/make_pizza.py` | subsumption | EL fragment via `mowl.geometric.normalise` + `to_owl`; hold out 10 %/10 % of asserted atomic subsumptions `A ⊑ B` (stratified by nothing, seed 0) into `valid.owl`/`test.owl`; `ontology.owl` = all remaining axioms; `ontology_deductive_closure.owl` = all entailed atomic subsumptions between names (ELK via the canonical model, `cm.entails`) — the same layout as mOWL's `SubsumptionDataset`, loaded with `PathDataset`+`deductive_closure_ontology` shim in `experiments/datasets.py`. |
| `go` | `mowl.datasets.builtin.GOSubsumptionDataset` | subsumption | as shipped (chen2020b protocol). |
| `foodon` | `mowl.datasets.builtin.FoodOnSubsumptionDataset` | subsumption | as shipped. |

A `dataset_stats.md` table: classes, roles, normalised axioms per NF, ⊥-axioms, number of
certificate triangles in the *training* ontology (§5.3), train/valid/test sizes.

### 5.2 Methods and grid

| method id | model | k | init | anchor_weight |
|---|---|---|---|---|
| `elbe` | `ELBE` | – | random | 0 |
| `elbe-init` | `ELBE` | – | projected exact | 0 |
| `elbe-anchor` | `ELBE` | – | projected exact | 0.1 |
| `box2el` | `BoxSquaredEL` | – | random | 0 |
| `box2el-init` | `BoxSquaredEL` | – | projected exact | 0 |
| `box2el-anchor` | `BoxSquaredEL` | – | projected exact | 0.1 |
| `multibox-k{1,2,3,4}` | `MultiBoxEL` | 1–4 | random | 0 |
| `multibox-k2-init` | `MultiBoxEL` | 2 | projected exact (k-means) | 0 |

Hyper-parameters (fixed, documented in `experiments/config.py`): `embed_dim` 50 for
`pizza/go/foodon`, `{2, 10}` for `tk*`; Box²EL/MultiBoxEL `margin(γ)=0.02, delta=2.5,
reg_factor=0.2, num_negs=4` (upstream `BoxSquaredEL` defaults); ELBE `margin=0.02`;
`learning_rate=0.001`; `batch_size=4096*8`; epochs: `tk*` 2000, `pizza` 2000, `go`
and `foodon` 300 with validation every 10 epochs and best-validation checkpointing
(`eval_gci_name="gci0"`); seeds `{0,1,2}` for `tk*`/`pizza`, `{0}` for `go`/`foodon`
(a second seed is run only if the first sweep finishes within the 12 GPU-hour budget;
the results table states the number of seeds per cell). `milp` initialisation is run
*in addition* for `pizza` only if the diagnostic reports no nondegenerate certificates
(then `pizza-box2el-milp`); otherwise it is reported as not applicable.

### 5.3 Metrics (`experiments/metrics.py`) — every run writes one JSON with all of these

* **Ranking** (subsumption datasets): mOWL `SubsumptionEvaluator(dataset).evaluate(module, dataset.testing, filter_ontologies=[train, valid, deductive_closure])` → MRR, Hits@1/3/10, AUC, filtered variants, both as returned; `mode` default.
* **Faithfulness ordering** (subsumption datasets), built from the canonical model of the
  training ontology: `T+` = test subsumptions (entailed, held out); `T−` = pairs `(A, B)`
  of satisfiable names with `cm` proving `A ⊓ B ⊑ ⊥` (`not cm.joint_satisfiable({A,B})`),
  sampled up to 5000 with seed 0; `T?` = pairs neither entailed nor contradicted, sampled
  up to 5000. Scores `f = module(·, "gci0")` (lower = more plausible). Report
  `AUC(+/−)`, `Pair-FVR = P[f(?) ≥ f(−)]` over `T? × T−` (sampled 10⁵ pairs),
  `FVR = P[¬(f(−) > f(?) > f(+))]` over sampled triples, and the sizes of the three sets.
  If `|T−| < 50` the ordering metrics are reported as `null` with the size.
* **Model-ness**: fraction of training normalised axioms, per GCI name and overall, whose
  loss is `≤ 1e-6` when the module's margin is set to `0` for the measurement (restored
  afterwards). For ELBE the margin parameter is `margin`; for Box²EL/MultiBoxEL `gamma`.
* **Helly metrics**: certificates = `certificate_triangles(cm)` of the training ontology
  (§5.4). For each certificate `(X1, X2, X3)` with pairwise names `(A12, A13, A23)`:
  `triple_nonempty` = some triple of components `X1_i ∩ X2_j ∩ X3_l ≠ ∅` (exact corner
  arithmetic, tolerance `1e-6`); `degenerate` = some `A_ij` has a component set that is
  entirely empty (`min_d offset ≤ 1e-3` in every component). `helly_violation_rate` =
  mean `triple_nonempty`; `degeneracy_rate` = mean `degenerate`; `n_certificates`.
  By Helly's theorem a single-box model has `triple_nonempty ∨ degenerate` for every
  certificate at a true zero of the loss; a multibox need not.
* **Cost**: parameter count, mean epoch time (s), total train time, `n_components`,
  mean effective components (MultiBoxEL), peak GPU memory.

### 5.4 Certificate triangles in mOWL (`mowl.geometric.helly.certificate_triangles`)

```python
def certificate_triangles(cm: CanonicalModel, *, max_triangles: Optional[int] = None) -> List[HellyCertificate]
```

Scalable, targeted search: collect named binary conjunctions `N ≡ A ⊓ B` from `NF2`
axioms `A ⊓ B ⊑ N` with `cm.entails(N, A) and cm.entails(N, B)` and `N`, `A`, `B`
satisfiable; build the graph with edge `{A, B}` labelled `N`; enumerate triangles; keep
those with all three pairwise conjunctions satisfiable and the triple unsatisfiable.
Returns `HellyCertificate(c1=(X1,), c2=(X2,), c3=(X3,), pairwise_names=(N12, N13, N23))`
sorted deterministically. Every returned certificate is a nondegenerate certificate in
the sense of the v4 spec (tested against `helly_check` on `t3.owl` and on random small
propositional TBoxes where `helly_check` is complete: every singleton-based nondegenerate
certificate of `helly_check` with named pairwise conjunctions that are *direct* NF2
definitions appears in the triangle list). It does not claim completeness (names defined
through chains of equivalences are not found); the docstring says so.

### 5.5 Runner, sweep, aggregation

* `experiments/run.py --dataset D --method M --dim d --seed s [--epochs E] --out results/raw/` → one JSON per run (config, metrics, timings, git hashes of both repos, torch/CUDA versions), plus the model state dict under `~/tmp/gfe-runs/` (not committed).
* `experiments/sweep.py --suite {synthetic,pizza,real,all} [--dry-run]` → enumerates §5.2 × seeds, skips runs whose JSON exists, runs sequentially on `cuda`, logs under `experiments/logs/`.
* `experiments/aggregate.py` → `results/tables/*.md` and `results/all_runs.csv`: one table per dataset with rows = methods, columns = the §5.3 metrics (mean ± sd over seeds, with `n`), plus a "curve" table for MultiBoxEL over `k`.
* Smoke mode (`--epochs 3`) must run every (dataset, method) once end to end before the full sweep; its outputs go to `results/smoke/` and are committed as evidence.

## 6. Acceptance

1. mOWL: `pytest tests/geometric tests/nn tests/models -m "not slow"` green on the branch; the slow solvability test (§4.4) and the ELK differential test green at least once (log committed in the PR description).
2. The `k = 1` identity tests pass.
3. The exact-initialisation containment guarantee test passes on `t3.owl`, `roles.owl`, and a 200-class random TBox.
4. Experiments: smoke sweep complete for all 11 methods × 7 datasets; full sweep complete for `synthetic` and `pizza`; `go`/`foodon` complete for at least seed 0 of every method, or the missing cells are listed explicitly in `results/tables/README.md` with the reason.
5. Every number in the final write-up traces to a JSON under `results/raw/`.

## 7. Risks and rulings in advance

* GO/FoodOn full-batch epochs may not fit 8 GB with `k = 4` and `d = 50` (FoodOn has
  ~30k classes → 30k·4·50·2 floats ≈ 24 MB of parameters; the gci1 pairwise tensor is
  `(B, k, k, k, d)` for a batch of `B` axioms: with `B = 32768`, `k = 4`, `d = 50` that is
  105 M floats ≈ 420 MB, acceptable). If memory fails, the runner halves `batch_size`
  for that run and records it; it never silently changes `k` or `d`.
* `SubsumptionEvaluator` evaluates all classes as tails: GO has ~45k classes and ~10k test
  axioms → 4.5·10⁸ scores per mode; batched on GPU this is minutes, acceptable.
* Pizza after EL normalisation is small (~100 names) and heavily disjoint; if the
  held-out split leaves fewer than 20 test axioms the split fractions are raised to
  20 %/20 % and the change recorded in `dataset_stats.md`.
* The upstream `train()` loop is full-batch per GCI type per epoch; we use it unchanged
  for comparability with upstream Box²EL/ELBE behaviour.
