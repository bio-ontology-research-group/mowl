# Learnable Geometric Models (R1 exact init, R2 MultiBoxEL, experiments) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add exact-model initialisation for box embeddings and the MultiBoxEL model to mOWL, and run the diagnostic, synthetic and subsumption-benchmark experiments that compare them with ELBE and Box²EL.

**Architecture:** Two repositories. In mOWL (worktree `~/Public/software/mowl-wt/geo`, branch `learnable-geometric`): `mowl.geometric.init` (R1), `mowl.geometric.helly.certificate_triangles`, `mowl.nn.el.multiboxel` + `mowl.models.multiboxel` (R2), two keyword arguments on `BoxSquaredEL`/`ELBE`. In `faithful-dl-embedding` (branch `experiments`, directory `experiments/`): dataset builders, metrics, runner, sweep, aggregation, results.

**Tech Stack:** Python 3.11 (`~/.venvs/mowl-gpu`), torch 2.11 cu128, numpy, scipy, mOWL 1.x (JPype/OWLAPI, ELK), pytest.

**Spec:** `docs/superpowers/specs/2026-10-29-learnable-geometric-design.md` (this repository).

## Global Constraints

- Never write to `/tmp`; scratch goes to `~/tmp/…` or inside the repository's ignored dirs.
- Python: `~/.venvs/mowl-gpu/bin/python`; run tests as `cd ~/Public/software/mowl-wt/geo && ~/.venvs/mowl-gpu/bin/python -m pytest …`. JVM: tests use the existing `tests/geometric/helpers.py` JVM bootstrap (`mowl.init_jvm`), never a second JVM.
- Do not edit the loss formulas of Box²EL or ELBE (`mowl/nn/el/boxsquaredel/losses.py`, `mowl/nn/el/elbe/losses.py`); new behaviour is opt-in through new keyword arguments whose defaults leave behaviour byte-for-byte unchanged.
- Tolerances: equality of losses `1e-6` (float32, `torch.allclose(atol=1e-6, rtol=1e-5)`); box emptiness `offset <= 1e-3`; triple-intersection tolerance `1e-6`.
- Every new public function/class has a NumPy-style docstring stating guarantees and non-guarantees; every claim of exactness in a docstring has a test.
- Commit after each task with a message starting `feat(geometric):`, `feat(nn):`, `feat(models):`, `docs:`, `exp:` or `test:`; do not push.
- Data: raw ontologies in `~/tmp/gfe-data/`; model checkpoints in `~/tmp/gfe-runs/`; only JSON/CSV/MD results under 1 MB are committed to `faithful-dl-embedding`.

## Review Focus

1. A class present in `dataset.class_to_id` but absent from the canonical model (e.g., appears only in valid/test) must keep its random initialisation and must not produce NaN parameters — `test_init_skips_unknown_classes` (Task 2).
2. `MultiBoxEL` with `n_components=1` must reproduce `BoxSquaredEL` losses on batches containing *empty* intersections, not only non-empty ones — `test_k1_identity_gci1_with_empty_intersections` (Task 3).
3. The anchor must never pull rows that were not initialised (NaN rows) — `test_anchor_loss_ignores_uninitialised_rows` (Task 2).
4. The experiment runner must fail loudly (non-zero exit, no JSON written) when a metric cannot be computed, rather than writing a JSON with silently missing keys — `test_run_fails_without_json_on_metric_error` (Task 8).
5. Model-ness must be measured at margin 0 and the margin must be restored afterwards — `test_modelness_restores_margin` (Task 7).

---

## Setup (controller)

- Wait for the rebase (`geometric-models-rebased`) to be green; `git checkout -b learnable-geometric geometric-models-rebased` in the worktree; commit the spec and this plan.
- `faithful-dl-embedding`: branch `experiments` (created by the diagnostic task), directory `experiments/`.

---

### Task 1: `certificate_triangles` in `mowl.geometric.helly`

**Files:**
- Modify: `mowl/geometric/helly.py` (append function; export in `mowl/geometric/__init__.py`)
- Test: `tests/geometric/test_helly_triangles.py`

**Interfaces:**
- Consumes: `CanonicalModel` (`.names`, `.axioms`, `.entails(a,b)`, `.joint_satisfiable(names)`, `.is_satisfiable(n)`), `NF2(sub1, sub2, sup)` — read the dataclass field names in `normal_forms.py`, `HellyCertificate`, `BOTTOM`, `TOP`.
- Produces: `certificate_triangles(cm, *, max_triangles=None) -> List[HellyCertificate]` with `c1=(X1,)`, `c2=(X2,)`, `c3=(X3,)`, `pairwise_names=(N12, N13, N23)`, `X1 < X2 < X3` lexicographically, list sorted.

- [ ] **Step 1: Write failing tests**

```python
def test_t3_triangle_matches_helly_check(t3_cm):            # fixture from helpers
    tri = certificate_triangles(t3_cm)
    assert len(tri) == 1
    c = tri[0]
    assert (c.c1, c.c2, c.c3) == ((NS+"X1",), (NS+"X2",), (NS+"X3",))
    assert set(c.pairwise_names) == {NS+"A", NS+"B", NS+"C"}
    rep = helly_check(t3_cm)
    assert {(x.c1, x.c2, x.c3) for x in rep.certificates_nondegenerate} == {(c.c1, c.c2, c.c3)}

def test_helly_ok_has_no_triangles(helly_ok_cm):
    assert certificate_triangles(helly_ok_cm) == []

def test_random_propositional_subset_of_helly_check():
    # 30 random propositional TBoxes (seeded 0..29), <= 12 names, with named binary
    # conjunctions and random disjointness; build via tests/geometric/helpers (existing
    # random TBox builder used by test_completion) — every triangle must be a
    # nondegenerate certificate of helly_check (as a set of three singleton names), and
    # every helly_check nondegenerate certificate whose three pairwise names are direct
    # NF2 definitions must be returned.

def test_max_triangles_truncates(t3_cm):
    assert certificate_triangles(t3_cm, max_triangles=0) == []
```

- [ ] **Step 2: Run tests, verify they fail with ImportError**
- [ ] **Step 3: Implement `certificate_triangles`** — edge map `{frozenset({A,B}): N}` from NF2 axioms with `sup != BOTTOM`, `cm.entails(N, A) and cm.entails(N, B)`, all three satisfiable; adjacency lists; triangle enumeration with `A < B < C`; keep if `cm.joint_satisfiable({A,B})`, `({A,C})`, `({B,C})` and `not cm.joint_satisfiable({A,B,C})`. Docstring: states what it finds, that it is not complete (chains of equivalences), complexity `O(Σ deg²)`.
- [ ] **Step 4: Run tests, verify pass**
- [ ] **Step 5: Commit** `feat(geometric): scalable certificate_triangles finder`

---

### Task 2: `mowl.geometric.init` — `ExactBoxInit`

**Files:**
- Create: `mowl/geometric/init.py`
- Modify: `mowl/geometric/__init__.py` (export `ExactBoxInit`)
- Test: `tests/geometric/test_init.py`

**Interfaces:**
- Consumes: `CanonicalModel.from_dataset`, `cm.names`, `cm.extension(name) -> FrozenSet[int]` (indices into `cm.names`), `cm.is_satisfiable`, `BoxModelBuilder`, `NotBoxRepresentable`, `BoxModelNotFound`, `UnsupportedRegion`; `BoxSquaredELModule`, `ELBEModule`, `MultiBoxELModule` (Task 3; until then test with Box²EL/ELBE only).
- Produces: `ExactBoxInit` exactly as spec §3.1 (fields, `from_canonical_model`, `from_dataset`, `apply_to_module`, `anchor_loss`, `as_tensors`).

- [ ] **Step 1: Write failing tests**

```python
def test_projected_k1_containment_t3(t3_dataset):
    init = ExactBoxInit.from_dataset(t3_dataset, embed_dim=7, seed=0)
    cm = CanonicalModel.from_dataset(t3_dataset)
    c2i = t3_dataset.class_to_id
    for a in cm.names:
        for b in cm.names:
            if cm.entails(a, b) and cm.is_satisfiable(a):
                lo_a, hi_a = box(init, c2i[a]); lo_b, hi_b = box(init, c2i[b])
                assert (lo_b - 1e-6 <= lo_a).all() and (hi_a <= hi_b + 1e-6).all()

def test_projected_k1_zero_gci0_loss_after_apply(t3_dataset):
    # BoxSquaredEL(t3_dataset, embed_dim=7, margin=0.0, init_boxes=init) -> module;
    # for every gci0 row of model.training_datasets["gci0"].data: module(data,"gci0") <= 1e-6
    # (uses Task 4's kwarg; until Task 4 lands, construct BoxSquaredELModule and call apply_to_module)

def test_bottom_and_unsat_are_points(t3_dataset): ...   # offset == 0, centre == 0 for BOTTOM
def test_init_skips_unknown_classes(roles_dataset):
    # add a class only to class_to_id (use a dataset whose test ontology has an extra class,
    # build one with PathDataset(ontology, testing=extra.owl) from fixtures/make_fixtures.py)
    # -> that row is NaN in init.centers and unchanged in the module after apply_to_module
def test_anchor_loss_ignores_uninitialised_rows(...)   # perturb the NaN row: anchor unchanged
def test_kmeans_components_duplicate_when_extension_small(t3_dataset):
    init = ExactBoxInit.from_dataset(t3_dataset, embed_dim=3, n_components=3, seed=0)
    # for a singleton-extension class all 3 components are identical and equal the k=1 box
def test_apply_to_elbe_and_box2el_write_same_values(...)
def test_milp_falls_back_on_t3(t3_dataset):
    init = ExactBoxInit.from_dataset(t3_dataset, embed_dim=4, method="milp", milp_time_limit=20)
    assert init.method == "projected" and init.report["fallback"].startswith("NotBoxRepresentable")
def test_milp_on_helly_ok_preserves_disjointness(helly_ok_dataset):
    # method="milp" succeeds; for every NF2 ⊥-axiom A ⊓ B ⊑ ⊥ the boxes of A and B are disjoint
def test_apply_rejects_dimension_mismatch(...)         # raises UnsupportedRegion
```

- [ ] **Step 2: Run, verify ImportError**
- [ ] **Step 3: Implement `mowl/geometric/init.py`** per spec §3.2–3.4. Signatures:

```python
class ExactBoxInit:
    @classmethod
    def from_canonical_model(cls, cm, class_to_id, embed_dim, *, n_components=1, method="projected",
                             seed=0, pad=0.05, scale=1.0, milp_time_limit=60.0, milp_d_max=None): ...
    @classmethod
    def from_dataset(cls, dataset, embed_dim, **kw): ...
    def apply_to_module(self, module, *, device=None) -> int: ...
    def anchor_loss(self, module) -> "torch.Tensor": ...
    def as_tensors(self, device=None): ...   # centers (n,k,d), offsets (n,k,d), mask (n,) bool
```
  Module detection in `apply_to_module`: if `hasattr(module, "n_components")` → MultiBoxEL layout (row = `(k*d)` flattened `(k, d)`); elif `hasattr(module, "class_center")` → Box²EL; elif `hasattr(module, "class_embed")` → ELBE; else raise `UnsupportedRegion`. For Box²EL/ELBE require `n_components == 1`.
- [ ] **Step 4: Run tests, verify pass**
- [ ] **Step 5: Commit** `feat(geometric): ExactBoxInit (projected polyhedral and MILP box initialisation, anchor)`

---

### Task 3: MultiBoxEL losses and module

**Files:**
- Create: `mowl/nn/el/multiboxel/__init__.py`, `mowl/nn/el/multiboxel/losses.py`, `mowl/nn/el/multiboxel/module.py`
- Modify: `mowl/nn/__init__.py` (export `MultiBoxELModule`)
- Test: `tests/nn/__init__.py` (if missing), `tests/nn/test_multiboxel_losses.py`

**Interfaces:**
- Consumes: `mowl.nn.el.boxsquaredel.losses` (`inclusion_score`, `box_distance`, `object_property_assertion_loss`, `reg_loss`, `gci2_score`, `gci2_score_neg` signatures), `ELModule`.
- Produces: `MultiBoxELModule(nb_ont_classes, nb_rels, nb_inds=None, embed_dim=50, n_components=2, gamma=0.0, delta=2.0, reg_factor=0.05, component_reg=0.0)` with attributes `n_components`, `embed_dim`, `class_center`, `class_offset` (`nn.Embedding(n, k*d)`), `bump_classes`, `head_center/head_offset/tail_center/tail_offset`, optional individual embeddings, methods `gci0_loss … object_property_assertion_loss(data, neg=False)`, `regularization_loss()`, `effective_components(eps=1e-3)`, `class_boxes(idx) -> (centers (B,k,d), offsets (B,k,d))`. Loss helpers in `losses.py`: `multi_inclusion(cen_a, off_a, cen_b, off_b, gamma) -> (B,ka,kb)`, `multi_intersection(cen_a, off_a, cen_b, off_b) -> (lower, upper) (B,ka,kb,d)`, and one `gciX_loss(...)` per table row in spec §4.2.

- [ ] **Step 1: Write failing tests** (`tests/nn/test_multiboxel_losses.py`)

```python
def make_pair(seed, n_classes=12, n_rels=3, d=5, n_inds=4):
    # BoxSquaredELModule(...) and MultiBoxELModule(..., n_components=1) with weights copied
    # (class_center/offset/bump_classes/head*/tail*/ind*); returns both

@pytest.mark.parametrize("gci,cols", [("gci0",2),("gci0_bot",1),("gci1",3),("gci1_bot",2),("gci2",3),("gci3",3),("gci3_bot",1),("class_assertion",2),("object_property_assertion",3)])
def test_k1_identity(gci, cols): ...   # allclose(atol=1e-6) on 256 random rows, seeds 0..2

def test_k1_identity_gci1_with_empty_intersections():
    # set offsets tiny and centres far apart so that >= 50% of rows have empty intersections;
    # assert allclose and assert that at least one row is empty (sanity on the batch)

def test_k1_identity_gci2_negative(): ...  # neg=True
def test_shapes_k3(): ...   # every loss returns shape (B,) for k=3
def test_gci0_zero_iff_each_component_inside_some_component(): # hand-built boxes, gamma=0
def test_gci1_empty_pairs_impose_nothing():
    # C has 2 components, D has 1; component C_1 meets D inside E, component C_2 is disjoint
    # from D and far outside E -> gci1 loss == 0 at gamma=0
def test_gci1_bot_requires_all_pairs_disjoint(): ...
def test_effective_components_counts_nested_and_degenerate(): ...
def test_regularization_matches_box2el_when_component_reg_zero(): ...
```

- [ ] **Step 2: Run, verify ImportError**
- [ ] **Step 3: Implement `losses.py` and `module.py`** per spec §4.1–4.2. Implementation notes: reshape embeddings `(B, k, d)` with `.view(B, k, d)`; use `torch.amin/argmin` (no gradient through the designated-pair choice; use `.detach()` for the gate tensor); compute gci1 inclusion for all `(i, j, m)` as a `(B,k,k,k)` tensor then mask; `effective_components` on CPU with numpy corners.
- [ ] **Step 4: Run tests, verify pass**
- [ ] **Step 5: Commit** `feat(nn): MultiBoxELModule (unions of k boxes with Box²EL semantics)`

---

### Task 4: `MultiBoxEL` model, init/anchor hooks, solvability test

**Files:**
- Create: `mowl/models/multiboxel/__init__.py`, `mowl/models/multiboxel/model.py`
- Modify: `mowl/models/__init__.py`, `mowl/models/boxsquaredel/model.py`, `mowl/models/elbe/model.py` (add `init_boxes=None, anchor_weight=0.0`; apply in `init_module`; extend `get_regularization_loss`)
- Test: `tests/models/__init__.py` (if missing), `tests/models/test_multiboxel.py`, `tests/models/test_box_init_hooks.py`

**Interfaces:**
- Consumes: `ExactBoxInit` (Task 2), `MultiBoxELModule` (Task 3), `EmbeddingELModel` (`init_module`, `get_regularization_loss`, `train(epochs, validate_every)`), fixtures `tests/geometric/fixtures/*.owl`.
- Produces: `MultiBoxEL(dataset, embed_dim=50, n_components=2, margin=0.02, reg_norm=1, learning_rate=0.001, batch_size=4096*8, delta=2.5, reg_factor=0.2, component_reg=0.0, num_negs=4, model_filepath=None, device='cpu', neg_sampling_gcis=None, init_boxes=None, anchor_weight=0.0)`; `BoxSquaredEL(..., init_boxes=None, anchor_weight=0.0)`; `ELBE(..., init_boxes=None, anchor_weight=0.0)`.

- [ ] **Step 1: Write failing tests**

```python
def test_multiboxel_trains_on_t3(t3_dataset, tmp_path):
    m = MultiBoxEL(t3_dataset, embed_dim=2, n_components=2, model_filepath=str(tmp_path/"m.pt"))
    m.train(epochs=3)                      # smoke: no exception, loss finite

def test_init_boxes_applied_box2el(t3_dataset, tmp_path):
    init = ExactBoxInit.from_dataset(t3_dataset, embed_dim=7)
    m = BoxSquaredEL(t3_dataset, embed_dim=7, init_boxes=init, model_filepath=...)
    c, o, mask = init.as_tensors()
    assert torch.allclose(m.module.class_center.weight[mask], c[mask, 0])

def test_init_boxes_applied_elbe(...): ...
def test_anchor_weight_adds_to_regularisation(t3_dataset, tmp_path):
    # anchor_weight=0 -> get_regularization_loss equals base; anchor_weight=1 after perturbing
    # class_center by +1 -> increases by anchor_loss (allclose)
def test_defaults_unchanged(t3_dataset, tmp_path):
    # BoxSquaredEL without new kwargs: state_dict equal to one built with the same torch seed
    # before the change (seed both; compare to a freshly constructed module under the same seed)

@pytest.mark.slow
def test_t3_solvability_multibox_vs_box2el(t3_dataset, tmp_path):
    # spec §4.4: MultiBoxEL(k=2,d=2,margin=0,lr=0.01) 1500 epochs full batch, seed 0 ->
    # total training loss (sum over all GCI datasets of mean loss) < 1e-4,
    # helly metrics via experiments-independent helper here: triple of components empty for
    # (X1,X2,X3) and no pairwise name A,B,C degenerate; BoxSquaredEL same setup -> loss > 1e-3
```
  For the slow test implement a tiny local helper `_triple_nonempty(boxes_x1, boxes_x2, boxes_x3)` with corner arithmetic (the experiments repo will have its own, Task 7; duplication of 15 lines is accepted to keep mOWL self-contained).

- [ ] **Step 2: Run, verify failures**
- [ ] **Step 3: Implement** model class (copy the structure of `mowl/models/boxsquaredel/model.py`), the kwargs in the two upstream models (apply init at the end of `init_module`, and
  `get_regularization_loss = super() + anchor_weight * init_boxes.anchor_loss(self.module)` when `init_boxes is not None and anchor_weight > 0`).
- [ ] **Step 4: Run fast tests, then the slow test (`-m slow -k solvability`), verify pass; paste both summaries in the report**
- [ ] **Step 5: Commit** `feat(models): MultiBoxEL model; init_boxes/anchor_weight hooks for BoxSquaredEL and ELBE`

---

### Task 5: Documentation and changelog

**Files:**
- Modify: `docs/source/geometric/api.rst`, `docs/source/geometric/index.rst`, `docs/source/geometric/examples.rst` (new section "Initialising box embeddings from the exact model" and "MultiBoxEL on T₃"), `CHANGELOG.md`; create `docs/source/embedding_el/multiboxel.rst` if the docs tree has per-model pages (check `docs/source/embedding_el/`), else add to the geometric docs.

- [ ] **Step 1: Write the docs** — the examples must be the code from the tests, trimmed; state the `k = 1` identity, the gci1 gating rule, the containment guarantee, and the non-guarantees (disjointness after projected init; incompleteness of `certificate_triangles`).
- [ ] **Step 2: Build docs if sphinx is available (`~/.venvs/mowl-gpu/bin/python -m sphinx -b html docs/source ~/tmp/mowl-docs -q`); otherwise `python -m rst2html` check is not required — record what was run**
- [ ] **Step 3: Commit** `docs: learnable geometric models (ExactBoxInit, MultiBoxEL, certificate_triangles)`

---

### Task 6: Experiment datasets (`faithful-dl-embedding/experiments/`)

**Files:**
- Create: `experiments/README.md`, `experiments/config.py`, `experiments/datasets.py`, `experiments/data/make_synthetic.py`, `experiments/data/make_pizza.py`, `experiments/data/README.md`, `experiments/results/dataset_stats.md`
- Test: `experiments/tests/test_datasets.py`

**Interfaces:**
- Consumes: `mowl.geometric.normalise`, `to_owl`, `CanonicalModel`, `certificate_triangles`; `mowl.datasets.PathDataset`, `GOSubsumptionDataset`, `FoodOnSubsumptionDataset` (`data_root=~/tmp/gfe-data/mowl`).
- Produces: `load_dataset(name: str) -> DatasetBundle` where `DatasetBundle(name, dataset: mowl Dataset, deductive_closure_ontology: Optional[OWLOntology], task: "satisfaction"|"subsumption", paths: dict)`; `DATASETS = ["tk3","tk4","tk5","tk6","pizza","go","foodon"]`; synthetic files under `experiments/data/synthetic/tk{k}.owl` (committed, tiny); pizza files under `~/tmp/gfe-data/pizza-el/{ontology,valid,test,ontology_deductive_closure}.owl` (regenerable, script committed; also committed if < 1 MB total).

- [ ] **Step 1: Write failing tests**

```python
def test_tk_has_expected_certificates():   # certificate_triangles on tk3 == 1 (names X1,X2,X3); tk4: the (k-1)-wise pattern gives nondegenerate certificates only via helly_check (sets of size 4): assert helly_check(cm, k_max=4).certificates_nondegenerate non-empty for tk4 and certificate_triangles(tk4) == [] — document.
def test_pizza_split_disjoint_and_closed():  # train/valid/test atomic subsumptions pairwise disjoint; every valid/test axiom entailed by the deductive closure of the *full* EL fragment; all classes of valid/test occur in train
def test_load_dataset_all():                 # every id loads; task field correct
```
  (The tk4 remark: T_k for k ≥ 4 names its (k−1)-wise conjunctions, not pairwise ones, so the certificate *triangles* are absent while `helly_check` finds the size-k minimal unsatisfiable set. Record this in `dataset_stats.md`; the Helly metrics on `tk4..tk6` therefore use the `helly_check` certificates with `c3` a tuple — Task 7 handles tuples.)

- [ ] **Step 2: Implement generators and loader**; synthetic generator writes OWL functional syntax via OWLAPI (through mOWL's JVM) or RDF/XML by hand — either, as long as `PathDataset` loads it and `normalise` yields the intended NF2/NF1 axioms; Pizza builder exactly per spec §5.1 (seed 0, 10 %/10 %, raise to 20 %/20 % if `< 20` test axioms, record).
- [ ] **Step 3: Produce `results/dataset_stats.md`** (classes, roles, NF counts, ⊥-axioms, `n_certificate_triangles` on the training ontology, split sizes; for `go`/`foodon` also download time and file sizes).
- [ ] **Step 4: Run tests, verify pass; Commit** `exp: datasets (synthetic T_k, Pizza-EL subsumption split, GO/FoodOn loaders) and stats`

---

### Task 7: Metrics

**Files:**
- Create: `experiments/metrics.py`
- Test: `experiments/tests/test_metrics.py`

**Interfaces:**
- Consumes: trained mOWL model (`model.module`, `model.training_datasets`), `DatasetBundle`, `SubsumptionEvaluator`, `CanonicalModel`, `HellyCertificate`, `helly_check`, `certificate_triangles`.
- Produces:
  - `ranking_metrics(model, bundle, device) -> dict` (keys as returned by mOWL, prefixed `rank_`)
  - `faithfulness_metrics(model, bundle, cm, seed=0, n_sample=5000) -> dict` (`auc_pos_neg`, `pair_fvr`, `fvr`, `n_pos`, `n_und`, `n_neg`; `None` metrics when `n_neg < 50`)
  - `modelness(model) -> dict` (`modelness_<gci>` per GCI dataset present, `modelness_all`)
  - `helly_metrics(model, bundle, cm, certificates) -> dict` (`helly_violation_rate`, `degeneracy_rate`, `n_certificates`); works for certificates whose `c3` is a tuple of several names (intersection over all names in `c1 ∪ c2 ∪ c3`; degeneracy over `pairwise_names` that are not `None`)
  - `cost_metrics(model, timings) -> dict`
  - `class_regions(model) -> (centers (n,k,d) numpy, offsets (n,k,d))` for ELBE/Box²EL (`k = 1`) and MultiBoxEL.

- [ ] **Step 1: Write failing tests** — `test_modelness_restores_margin`, `test_helly_metrics_box2el_random_init_violates_t3` (random Box²EL on t3: `helly_violation_rate == 1.0` or `degeneracy_rate == 1.0` is *not* asserted; assert only that the function returns rates in `[0,1]` and `n_certificates == 1`), `test_helly_metrics_handmade_multibox_zero` (hand-set a 2-component solution of T₃ in 1-D: `X1=[0,1]∪[2,3]`, `X2=[0,1]∪[4,5]`, `X3=[2,3]∪[4,5]`, `A=[0,1]`, `B=[4,5]`, `C=[2,3]` → both rates `0`), `test_faithfulness_sets_are_disjoint_and_sized`, `test_class_regions_shapes`.
- [ ] **Step 2: Implement**; `faithfulness_metrics` samples `T−`/`T?` from `cm.names` pairs with `rng = numpy.random.default_rng(seed)`, rejection sampling with a cap of `50 * n_sample` draws.
- [ ] **Step 3: Run tests, verify pass; Commit** `exp: metrics (ranking, faithfulness ordering, model-ness, Helly, cost)`

---

### Task 8: Runner, sweep, aggregation, smoke sweep

**Files:**
- Create: `experiments/run.py`, `experiments/sweep.py`, `experiments/aggregate.py`, `experiments/results/smoke/` (outputs), `experiments/results/tables/README.md`
- Test: `experiments/tests/test_run.py`

**Interfaces:**
- Consumes: Tasks 6–7; mOWL models `ELBE`, `BoxSquaredEL`, `MultiBoxEL`, `ExactBoxInit`.
- Produces: `run.py` CLI per spec §5.5 writing `results/raw/{dataset}__{method}__d{dim}__s{seed}.json` with keys `config`, `metrics`, `timings`, `env` (git hashes of both repos, torch/cuda, GPU name); `sweep.py --suite {synthetic,pizza,real,all} [--smoke] [--dry-run]`; `aggregate.py` producing `results/tables/{dataset}.md`, `results/tables/multibox_curve.md`, `results/all_runs.csv`.
- Method table: exactly spec §5.2 (`METHODS` dict in `config.py` mapping id → model class, kwargs, init method, anchor weight).

- [ ] **Step 1: Write failing tests** — `test_run_fails_without_json_on_metric_error` (monkeypatch `metrics.modelness` to raise; `run.main([...])` exits non-zero and no JSON exists), `test_run_smoke_tk3_multibox_k2_cpu` (3 epochs, writes JSON with all metric keys), `test_sweep_dry_run_lists_all_cells` (`len == 11 methods × datasets × dims × seeds` for the suite).
- [ ] **Step 2: Implement**; training uses `model.train(epochs, validate_every)` with `eval_gci_name="gci0"` for subsumption datasets; for satisfaction datasets (no validation) train for the fixed epochs and evaluate the final parameters; load best checkpoint for subsumption datasets; `torch.manual_seed(seed)`, `numpy` seed; record `batch_size` halving on CUDA OOM (spec §7) — catch `torch.cuda.OutOfMemoryError`, halve, retry at most 3 times, record in `config.effective_batch_size`.
- [ ] **Step 3: Run the smoke sweep** `sweep.py --suite all --smoke` (epochs 3, every dataset × method, dims per config, seed 0) on CUDA; outputs to `results/smoke/`; `aggregate.py --raw results/smoke --out results/smoke/tables`.
- [ ] **Step 4: Run tests; Commit** `exp: runner, sweep, aggregation; smoke sweep outputs`

---

### Task 9: Full sweep

**Files:**
- Outputs: `experiments/results/raw/*.json`, `experiments/results/tables/*.md`, `experiments/results/all_runs.csv`, `experiments/logs/sweep-*.log` (logs only if < 1 MB each, else summarised), `experiments/results/tables/README.md` (which cells exist, seeds per cell, wall-clock totals, any batch-size halving).

- [ ] **Step 1: `sweep.py --suite synthetic`** (3 seeds, dims 2 and 10) — expected minutes.
- [ ] **Step 2: `sweep.py --suite pizza`** (3 seeds, dim 50).
- [ ] **Step 3: `sweep.py --suite real`** seed 0 for all methods on `go` and `foodon`; measure the first run's time per epoch and report the projected total before continuing; stop at 12 GPU-hours total and record what is missing.
- [ ] **Step 4: `aggregate.py`**; commit results after each suite: `exp: results — synthetic`, `exp: results — pizza`, `exp: results — go/foodon (seed 0)`.
- [ ] **Step 5: Report** — summary tables pasted into the report file, plus anomalies (NaN losses, non-converging runs, OOM halvings).
