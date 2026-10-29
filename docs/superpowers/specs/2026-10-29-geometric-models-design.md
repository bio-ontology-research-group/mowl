# Design: `mowl.geometric` — constructive geometric models of EL++ ontologies

Date: 2026-10-29. Author: Robert Hoehndorf (design conversation with Claude).
Status: revised after builder check (v2, 2026-10-29). Target branch: `geometric-models`,
PR to `main`. Changes in v2 are marked **[v2]**.

## 1. Intent and why

mOWL's geometric embedding methods (ELEmbeddings, ELBoxEmbeddings, Box²EL) *learn*
regions and only approximately satisfy an ontology. We now know precisely when exact
region models exist:

* **Boxes** (Helly number 2) cannot model the six-axiom theory
  `T₃ = {A ≡ X₁⊓X₂, B ≡ X₂⊓X₃, C ≡ X₁⊓X₃, A⊓B ⊑ ⊥, B⊓C ⊑ ⊥, A⊓C ⊑ ⊥}` in any dimension
  (Leemhuis & Kutz, NeSy 2025; our Lean-checked proof in
  `bio-ontology-research-group/faithful-dl-embedding`).
* **Balls** cannot either.
* **Convex polyhedra** can model every EL++ ontology with a finite canonical model,
  exactly, using one dimension per canonical-model element (Gutiérrez-Basulto &
  Schockaert, KR 2018, Prop. 3; Lacerda, Ozaki & Guimarães, TGDK 2024, Sec. 5), and
  the dimension cannot in general be reduced below (model size − 1) (Helly).
* **Unions of intervals on the real line** model every finite interpretation exactly
  (Lacerda et al. 2024, Sec. 4). More generally, *multiboxes* (finite unions of boxes)
  model every satisfiable ELHO(∘)⊥ ontology in any dimension n ≥ 1 (Federico, Leemhuis &
  Kutz, JOWO 2026, Thm. 3); our interval-union model is the n = 1 case. The same group's
  *prism* embeddings (triangle × box, DL 2026) escape Helly number 2 while staying
  convex and intersection-closed; they are a candidate later addition, not part of v1.
* An ontology has a **box model** only if it is *Helly-satisfiable* (Leemhuis & Kutz
  2025, Def. 6); for the propositional fragment this is decidable by a simple
  criterion (Sec. 5.3 below).

The goal of this work is to put these *constructive* results into mOWL as a new
package, so that (i) users can build an exact geometric model of an ontology and use it
as a baseline, an initialiser, or an oracle; (ii) the library can *check* whether a
learned or constructed region assignment is a model; and (iii) the box-representability
limit becomes a diagnostic users can run on their ontology.

## 2. Goals, non-goals, anti-goals

Goals

* G1. `mowl.geometric.CanonicalModel`: a finite canonical model of a normalised EL++
  TBox, computed in Python by an ELK-style completion (port of the `moose/elpp`
  completion, the Python sibling of the kobayashi-marust reasoner), with an optional
  cross-check against the kobayashi-marust `elc` binary when it is available.
* G2. `ConvexPolyhedralModel.from_ontology(...)`: exact convex model in ℝ^(n−1),
  n = number of canonical-model elements; classes = faces of a simplex (convex hulls
  of element points), roles = convex hulls of pair vectors in ℝ^(2(n−1)). **[v2]**
  Faces, not thickened half-space sets: see 5.3.
* G3. `IntervalUnionModel.from_ontology(...)`: exact one-dimensional model; classes =
  finite unions of intervals, roles = finite unions of rectangles.
* G4. `helly_check(...)`: Helly-satisfiability diagnostic with a certificate (a triple of
  definable concepts that are pairwise jointly satisfiable but not jointly satisfiable),
  exact for the propositional fragment, sound (necessary condition) in general.
* G5. `BoxModelBuilder`: for ontologies in the propositional fragment (GCI0, GCI1 and
  their ⊥ forms) that pass the Helly check, *search* for an axis-parallel box model by
  mixed-integer linear programming (`scipy.optimize.milp`), verified by the model
  checker. **[v2]** A passing Helly check does not guarantee that a box model exists
  (no such theorem is known to us), so the builder's contract is: a found model is
  verified and returned; otherwise it reports *inconclusive* with the budget it used.
  Only a Helly certificate proves non-existence.
* G6. `ModelChecker`: decides, exactly, whether a given region assignment satisfies each
  normalised axiom (GCI0–GCI3 and ⊥ forms), for the three region families above.
* G7. Every model object exposes `is_model()`, `is_nondegenerate()`,
  `satisfies(axiom)`, `class_regions`, `role_regions`, and an export of class/role
  embeddings usable by the existing `mowl.models` / `mowl.evaluation` machinery
  (points, boxes, or bounding boxes, with the loss of exactness documented).
* G8. Tests, Sphinx docs, CHANGELOG entry, examples.

Non-goals (this iteration)

* Axis-aligned cone models of ALC (Özçep et al.) — explicitly deferred.
* Box models with roles (Box²EL-style head/tail/bump semantics) — the box builder is
  propositional in v1; role axioms raise `NotImplementedError` with a clear message.
* Learning. Nothing here trains anything; outputs may seed trainers.
* GPU / torch tensors inside `mowl.geometric` (numpy + scipy only).

Anti-goals

* Claiming a model when it is not one. Every constructed model is passed through
  `ModelChecker` before it is returned; a failed check is a bug and raises.
* Claiming exactness or completeness the mathematics does not give. **[v2]** The
  docstrings state exactly which guarantee each procedure has (see 5.3).
* Silent approximation. Any export that loses exactness (e.g. bounding boxes of
  polyhedra) is a separate, documented method, never the default `class_regions`.

## 3. Constraints

* Python ≥ 3.8 per `setup.py` (use typing compatible with 3.8: no `list[str]` in
  signatures without `from __future__ import annotations`). numpy, scipy already in
  `install_requires`; no new mandatory dependency. `scipy.optimize.milp` needs
  scipy ≥ 1.9: guard with a clear error if missing.
* Ontologies come in as mOWL `Dataset` objects or OWLAPI `OWLOntology` (JPype); the
  normalised axioms come from `mowl.ontology.normalize.ELNormalizer` (GCI0, GCI1, GCI2,
  GCI3 and `*_BOT` classes; role inclusions and chains are read directly from the
  OWLAPI ontology: `SubObjectPropertyOf`, `SubPropertyChainOf`).
* Must run on a laptop for small/medium ontologies (hundreds to a few thousand
  classes). Document that the polyhedral model has dimension ≈ |classes| and that LP
  checks scale accordingly; the interval model and the Helly check scale to large
  ontologies.
* Development environment: there is no working mOWL environment on the laptop yet.
  The implementer creates `~/.venvs/mowl-dev` (pyenv 3.10.4), `pip install -e .`,
  builds the Java gateway jars with `./build_jars.sh` (gradle and Java 21 are
  installed), and runs `pytest -m "not slow" -p no:faulthandler tests/geometric`.
  mOWL tests use `unittest.TestCase` classes under `tests/`, with datasets from
  `tests/datasetFactory.py` (`FamilyDataset` is the small one).

## 4. Package layout (decided)

```
mowl/geometric/
  __init__.py          public API re-exports
  completion.py        EL++ normal forms (NF1–NF7) + ELK-style saturation + CanonicalModel
  regions.py           Region types: Polyhedron (H-rep + known points), IntervalUnion,
                       Box; PairRegion types for roles
  checker.py           ModelChecker: exact satisfaction tests per axiom type and family
  polyhedral.py        ConvexPolyhedralModel
  intervals.py         IntervalUnionModel
  helly.py             helly_check, HellyReport, minimal-unsat-set enumeration
  boxes.py             BoxModelBuilder (MILP), BoxModel
  export.py            conversions to embeddings (points / boxes / bounding boxes)
  errors.py            NotBoxRepresentable, InconsistentOntology, UnsupportedAxiom
tests/geometric/
  test_completion.py   vs. ELK (OWLAPI reasoner through mowl.reasoning) on fixtures
  test_polyhedral.py   is_model, nondegenerate, strong faithfulness on names, T3
  test_intervals.py    same
  test_helly.py        T3 certificate; propositional exactness lemma cases
  test_boxes.py        Helly-satisfiable fixtures → box model verified; T3 → raises
  fixtures/            small OWL files: t3.owl, helly_ok.owl, roles.owl, family (existing)
docs/source/geometric/ index.rst, api.rst, examples
docs/superpowers/specs/ this file
CHANGELOG.md           Unreleased → Added
```

## 5. Semantics (decided; the builder implements exactly this)

### 5.1 Normal forms and canonical model

Input: normalised axioms
NF1 `A ⊑ B`, NF2 `A ⊓ B ⊑ C`, NF3 `A ⊑ ∃r.B`, NF4 `∃r.A ⊑ B`, NF5 `A ⊑ ⊥` (mOWL's
GCI0_BOT), NF2 with `C = ⊥` (GCI1_BOT), NF4 with `B = ⊥` (GCI3_BOT), NF6 `r ⊑ s`,
NF7 `r ∘ s ⊑ t`. `⊤` is a concept name with `A ⊑ ⊤` for all `A`.

Completion (port of `moose/elpp/completion.py`; cite Kazakov, Krötzsch & Simančík,
*The Incredible ELK*, JAR 2014): worklist saturation producing
`S(A) = {B : O ⊨ A ⊑ B}` for every concept name `A`, and
`R(r) = {(A, B)}` edges derived from NF3 (`A ⊑ ∃r.B` with `A` live) closed under NF6/NF7
and the ELK rules R1–R7 (including the ⊥ propagation `R(r) ∋ (A,B), ⊥ ∈ S(B) ⇒ ⊥ ∈ S(A)`).

`CanonicalModel`:
* domain `Δ = {d_A : A ∈ N_C ∪ {⊤}, ⊥ ∉ S(A)}` (unsatisfiable names have no element);
* `A^I = {d_B : A ∈ S(B)}`;
* `r^I = {(d_B, d_C) : (B, C) ∈ R(r)}`;
* if `⊥ ∈ S(⊤)` raise `InconsistentOntology`.

Guarantees (tested): `I ⊨ O`; for concept names `A, B`: `A^I ⊆ B^I ⇔ O ⊨ A ⊑ B`;
`A^I = ∅ ⇔ O ⊨ A ⊑ ⊥`. Test by comparing `S` with ELK's `getSubClasses` via
`mowl.reasoning.MOWLReasoner` on the fixtures, and optionally with kobayashi-marust
`elc` (`KM_ELC_BIN` env var; input `{"clauses": [...]}` per
`kobayashi-marust/docs/INPUT-FORMATS.md`; output `{"subsumptions": ...}`). Skip the
optional test when the binary is absent.

### 5.2 Geometric interpretations and the checker

A geometric interpretation over a *universe* `U ⊆ ℝ^m` assigns to each concept name a
region `η(A) ⊆ U` and to each role name a region `η(r) ⊆ U × U ⊆ ℝ^(2m)` (pairs `(v,u)`),
with `η(⊤) = U`, `η(⊥) = ∅`, `η(C ⊓ D) = η(C) ∩ η(D)` and
`η(∃r.C) = {v : ∃u ∈ η(C), (v,u) ∈ η(r)}` (Lacerda et al. 2024, Def. 2; Gutiérrez-Basulto &
Schockaert 2018). `η ⊨ C ⊑ D ⇔ η(C) ⊆ η(D)`. Non-degenerate: `η(A) ≠ ∅` for every
satisfiable name `A`. Strongly faithful on names: `η(A) ⊆ η(B) ⇔ O ⊨ A ⊑ B`.
**[v2]** The universe is part of the interpretation object (`eta.universe`): for the
constructed models it is the region spanned by all canonical elements (the simplex,
resp. the union of all element intervals), so that axioms with `⊤` on the left
(`⊤ ⊑ A`, `⊤ ⊑ ∃r.B`, which the normaliser produces from domain/range axioms) are
checkable and hold exactly when they hold in the canonical model. For learned models
the universe is `ℝ^m`. `⊤` is also an ordinary canonical element `d_⊤` with
`⊤^I = Δ`, so `η(⊤) = conv/union of all element regions = U` is consistent.

`ModelChecker.satisfies(eta, axiom)` must be exact for the three families:

* IntervalUnion (1-D) and Box: inclusion, intersection and emptiness are decided
  combinatorially (interval set algebra; corner comparisons). `∃r.C` for interval
  unions: `η(r)` is a finite union of rectangles `I × J`; `η(∃r.C) = ⋃{I : J ∩ η(C) ≠ ∅}`.
* Polyhedron (H-representation `Ax ≤ b`, plus the known generating points): inclusion
  `P ⊆ Q` by one LP per facet of `Q` (maximise facet violation over `P`; `P ⊆ Q` iff all
  maxima ≤ 0); emptiness by LP feasibility; roles as V-polytopes `conv{p_x ⊕ p_y}`;
  GCI3 `∃r.A ⊑ B`: for each facet `h` of `η(B)`, LP over `(v,u) ∈ η(r)`, `u ∈ η(A)`
  maximising `h(v)`; satisfied iff all maxima ≤ 0. GCI2 `A ⊑ ∃r.B`: `η(∃r.B)` is the
  projection of the convex set `η(r) ∩ (ℝ^m × η(B))`, hence convex, so it suffices to
  check the vertices of `η(A)`: for each such `v`, LP feasibility of
  `∃u: (v,u) ∈ η(r), u ∈ η(B)`. **[v2]** This is valid because in the polyhedral model
  every class region *is* the convex hull of its stored element points (a simplex
  face); `Polyhedron` stores both the vertex list and the H-representation of the face
  (`λ_j(x) = 0` for `d_j ∉ S`, `λ_j ≥ 0`, `Σλ = 1`), and the checker refuses (raises
  `UnsupportedRegion`) to run GCI2 on a polyhedron that does not carry a vertex list.

`is_model(O)` = all normalised axioms satisfied. `is_nondegenerate()` and
`is_strongly_faithful_on_names(O)` as above (the latter uses the CanonicalModel's `S`).

### 5.3 Constructions

**Convex polyhedral** (G2) **[v2, corrected]**. Let `Δ = {d_1..d_n}`, `k = n − 1`,
points `v_1..v_n` = vertices of the standard simplex in ℝ^k: `v_i = e_i` for `i < n`,
`v_n = 0`; barycentric coordinates `λ_j(x) = x_j` (j < n), `λ_n(x) = 1 − Σ_j x_j`.
For `S ⊆ Δ`: `F(S) = conv{v_i : d_i ∈ S}` (the face of the simplex spanned by `S`;
`F(∅) = ∅`). `η(A) = F(A^I)`, `U = η(⊤) = F(Δ)` (the whole simplex), and
`η(r) = conv{v_i ⊕ v_j : (d_i, d_j) ∈ r^I} ⊆ ℝ^(2k)`. This is exactly the construction
of Gutiérrez-Basulto & Schockaert 2018, Prop. 3 (one dimension per element, hull of
one-hot points, relations as hulls of concatenations), restated in the simplex
coordinates; their proof gives `I ⊨ O ⇒ η ⊨ O` for all normal forms including GCI2/GCI3,
and `F(S) ∩ F(S') = F(S ∩ S')`, `F(S) ⊆ F(S') ⇔ S ⊆ S'` give exact conjunction and
strong faithfulness on names. Regions are lower-dimensional faces (no interior); that is
by design — the earlier idea of thickened half-space sets `P(S)` is **rejected**: it is
unbounded and breaks GCI2 (a bounded role region cannot cover an unbounded class) and
GCI3. Non-degeneracy means non-empty, nothing more. Optionally, for ontologies *without
role axioms*, `thicken=True` may return `P(S) ∩ F(Δ)` regions with interior (then
`ε < 1/n`), documented as propositional-only. Store per class: vertex list `{v_i}` and
the face H-representation. Exactness of GCI2/GCI3 must still be verified by the checker
in tests; a failing fixture is a finding to report, not to paper over.

**Interval unions** (G3). Element `d_i ↦ [i − ¼, i + ¼]` (i = 1..n).
`η(A) = ⋃_{d_i ∈ A^I} [i − ¼, i + ¼]`, `U = η(⊤) = ⋃_i [i−¼, i+¼]`,
`η(r) = ⋃_{(d_i,d_j) ∈ r^I} [i−¼, i+¼] × [j−¼, j+¼]`.
This is Lacerda et al. 2024, Definition 12 (with their `μ̄`), and the 1-D case of the
multibox construction of Federico, Leemhuis & Kutz 2026 (Thm. 3); exact for all axiom
types.

**Helly check** (G4). Definitions follow Leemhuis & Kutz 2025: a family of sets has
Helly's property (HP) if any finite subfamily whose members pairwise intersect has a
common point; an ontology is Helly-satisfiable if it has a model in which the family
of extensions of all definable concepts has HP; box interpretations always have HP
(their Prop. 4), so non-Helly-satisfiable ⇒ no box model (their Prop. 5).

**[v2, made precise.]** What the check proves. A *certificate* is a triple of concepts
`C₁, C₂, C₃`, each a conjunction of names, such that `O ⊨ C₁ ⊓ C₂ ⊓ C₃ ⊑ ⊥` while each
pairwise conjunction `Cᵢ ⊓ Cⱼ` is satisfiable. In any box interpretation of `O` in which
those three pairwise conjunctions are non-empty, the three boxes `η(Cᵢ)` pairwise
intersect, hence (Helly number 2) have a common point, contradicting the axiom. So a
certificate proves: *no box model of `O` exists in which `C₁⊓C₂`, `C₂⊓C₃`, `C₁⊓C₃` are all
non-empty*. Two readings are reported:
* `certificates_nondegenerate`: certificates whose three pairwise conjunctions are each
  `O`-equivalent to a satisfiable *concept name* (so any non-degenerate box model is
  excluded; this is the `T₃` situation, `X₁⊓X₂ ≡ A` etc.);
* `certificates_strong`: all certificates (they exclude strongly faithful box models,
  where every satisfiable conjunction must be non-empty).
This is the TBox-only counterpart of Leemhuis & Kutz's Helly antipattern (their
Example 2 forces the witnesses with an ABox; here names or strong faithfulness do).
Note that `T₃` without the names `A, B, C` is Helly-satisfiable in the sense of Leemhuis
& Kutz, because a model need not realise the pairwise conjunctions; the docstring must
make this distinction.

Enumeration (propositional fragment, no NF3/NF4): let `𝒰` be the family of
inclusion-minimal sets of concept names that are jointly unsatisfiable
(`O ⊨ ⨅U ⊑ ⊥`). **Lemma.** There is a `certificates_strong` certificate iff some member
of `𝒰` has size ≥ 3; for `U = {u₁,…,u_k}`, `k ≥ 3`, the triple `u₁, u₂, ⨅(U∖{u₁,u₂})` is
one (pairwise conjunctions are proper subsets of `U`, hence satisfiable by minimality;
the triple conjunction is `⨅U`). Conversely a certificate `C₁,C₂,C₃` yields the
unsatisfiable name set `names(C₁)∪names(C₂)∪names(C₃)`, which contains a minimal
unsatisfiable subset `U`; `|U| ≥ 3` because every pair of names inside the union lies
inside some satisfiable `Cᵢ ⊓ Cⱼ`. The builder enumerates minimal unsatisfiable name
sets by a grow/shrink procedure using the completion (joint satisfiability of
`{A₁..A_m}` = satisfiability of a fresh name `Z` with `Z ⊑ Aᵢ`, decided by one extra
completion run, or incrementally), restricted to candidate sets whose members are
pairwise satisfiable, up to size `k_max` (default 4) and `max_certificates` (default
10). `HellyReport.complete` is `True` only when the enumeration was not cut off by
`k_max`/`max_certificates` and the ontology has no role axioms; the "exact for the
propositional fragment" claim holds only in that case, **and only for the strong
reading**, and the docstring says so. The auxiliary names introduced by the normaliser
count as names (they are in `N_C`).

**[v3, after Task 7 review]** The Lemma is complete for *strong* certificates only.
Whether a certificate is nondegenerate depends on how the minimal set is split into
`C₁, C₂, C₃`, and the union of a nondegenerate certificate need not itself be a minimal
set (counterexample found in review: `a⊓c ⊑ AC, AC⊓b ⊑ N1, b⊓d ⊑ BD, BD⊓a ⊑ N2,
c⊓d ⊑ N3, N1 ⊑ a,b,c, N2 ⊑ a,b,d, N3 ⊑ c,d, N1⊓d ⊑ ⊥` has the nondegenerate
certificate `(a⊓b, c, d)` but the enumeration's splits classify every minimal set as
strong-only). Therefore `box_representable_nondegenerate` is `False` if nondegenerate
certificates were found, `True` only if `complete` and `certificates_strong` is empty
(nondegenerate ⊆ strong), and `None` otherwise. Pruning that keeps the enumeration
sound and makes `complete` reachable: only *⊥-relevant* names are candidates, where
relevance is the backward closure from the ⊥-axioms through NF1/NF2 (a name is relevant
if it is a left-hand name of an NF2 with ⊥ on the right, or the `sub` of an NF1 / a
`sub1`/`sub2` of an NF2 whose `sup` is relevant); in the propositional fragment every
derivation of `Z ⊑ ⊥` uses only relevant names, so minimal unsatisfiable sets consist
of relevant names. If there is no ⊥-axiom, `𝒰 = ∅` and the report is complete with no
certificates.

With roles: run the same enumeration over the *occurring* definable concepts
(names, left-hand conjunctions of NF2, `∃r.A` of NF4) and report `complete=False`
in the `HellyReport`; a certificate is still a proof of non-box-representability.
(Leemhuis & Kutz 2026, Prop. 38, show that with roles even their ABox-closure test is
necessary but not sufficient, so `complete=False` is the honest answer here; the
docstring must say so.)

**Box model** (G5) **[v2, contract corrected]**, propositional fragment only, run
after the Helly check. Unknowns per satisfiable concept name `A` and coordinate
`i ∈ 1..d`: `L_{A,i} ≤ U_{A,i}` (continuous, in `[0, M]`); one witness point `w_A`
per satisfiable name (and, with `faithful="strong"`, one witness `w_{A,B}` per
non-entailed `A ⊑ B` between satisfiable names, and one per satisfiable left-hand
conjunction of the TBox). Constraints:
* non-degeneracy `L_{A,i} ≤ w_{A,i} ≤ U_{A,i}` for all `i`;
* with `faithful="strong"`: `w_{A,B} ∈ η(A)` and `w_{A,B} ∉ η(B)` (the latter is a
  disjunction over coordinates: binaries with big-M, margin `δ`). The default is
  `faithful="weak"` (axioms + non-degeneracy only); the canonical model's membership
  pattern is *not* pinned, since pinning it searches only for strongly faithful models
  and can need many more dimensions;
* GCI0 `A ⊑ B`: `L_{B,i} ≤ L_{A,i}`, `U_{A,i} ≤ U_{B,i}`;
* GCI1 `A ⊓ B ⊑ C`: either the intersection is empty on some coordinate (binary
  per coordinate: `U_{A,i} + δ ≤ L_{B,i}` or `U_{B,i} + δ ≤ L_{A,i}`) or on every coordinate
  `L_{C,i} ≤ max(L_{A,i}, L_{B,i})` and `min(U_{A,i}, U_{B,i}) ≤ U_{C,i}` (binaries select
  which argument attains the max/min);
* ⊥ forms: `A ⊓ B ⊑ ⊥` ⇒ empty intersection on some coordinate; `A ⊑ ⊥` ⇒ `A` has no
  box (drop it; `η(A) = ∅`).
Objective: none (feasibility), or minimise total box width. Dimension schedule:
start with `d = 1`, double up to `d_max` (default 8); the first feasible `d` wins. The
returned `BoxModel` is passed through `ModelChecker` before return. Outcomes:
* a Helly certificate of the requested kind exists → raise `NotBoxRepresentable`
  carrying the `HellyReport` (a proof);
* no certificate and MILP infeasible for every `d ≤ d_max`, or time limit exceeded →
  raise `BoxModelNotFound` (inconclusive; message states `d_max`, `time_limit`, and
  that no theorem guarantees existence). In the §7 property test this outcome is
  logged, never asserted as failure.
Document: feasibility is checked by MILP with `scipy.optimize.milp` (HiGHS); exact but
exponential in the worst case; intended for small ontologies (≤ ~200 names).

### 5.4 Exports (G7)

* `ConvexPolyhedralModel.element_points()` → `np.ndarray (n, k)`; `class_points(A)`.
* `IntervalUnionModel.class_intervals(A)` → list of `(lo, hi)`.
* `BoxModel.boxes()` → dict `A ↦ (L, U)` arrays, directly usable as an initialiser for
  `ELBoxEmbeddings` / `BoxSquaredEL` modules (document the mapping of class index
  dicts, which come from `Dataset.class_to_id`).
* `to_bounding_boxes()` on the polyhedral model (lossy; documented as such).

## 6. API sketch

```python
from mowl.geometric import (CanonicalModel, ConvexPolyhedralModel, IntervalUnionModel,
                            helly_check, BoxModelBuilder, ModelChecker)

cm = CanonicalModel.from_dataset(dataset)          # or .from_ontology(owl_ontology)
cm.entails("A", "B"); cm.is_satisfiable("A"); cm.domain; cm.extension("A"); cm.role("r")

poly = ConvexPolyhedralModel.from_canonical_model(cm)   # or .from_dataset(dataset)
assert poly.is_model(cm.axioms) and poly.is_nondegenerate()
poly.dimension; poly.class_region("A"); poly.satisfies(axiom)

ivl = IntervalUnionModel.from_canonical_model(cm)

report = helly_check(cm)            # HellyReport(satisfiable, certificates, complete)
box = BoxModelBuilder(cm, d_max=8, time_limit=60).build()   # raises NotBoxRepresentable
```

Axioms are the mOWL `GCI*` objects from `ELNormalizer` plus two small classes for role
inclusions and chains (`RI`, `RC`) defined in `mowl.geometric.completion`. **[v2]** An
adapter from learned `ELEmbeddings`/`ELBoxEmbeddings` parameters to `Box`/`Ball`
regions is *out of v1 scope*; the `Box` region type and the checker are written so
that adding it later is a small task.

## 7. Testing requirements (decided)

* Unit tests per module; every construction tested against `ModelChecker` on every
  fixture; `T3` fixture: polyhedral and interval models exist and are non-degenerate,
  `helly_check` returns a size-3 certificate `{X1, X2, X3}`, `BoxModelBuilder` raises.
* Completion cross-checked against ELK. **[v2, made precise]** The single source of
  truth for `I ⊨ O` is the normalised axiom set `N(O)` (ELNormalizer output + role
  inclusions/chains read from the OWL ontology; chains longer than two are split with
  fresh roles). Two comparisons: (a) exact agreement of all concept-name subsumptions
  and unsatisfiable names with ELK run on `N(O)` re-serialised as OWL (the fixture
  ontologies and `FamilyDataset`); (b) soundness against the original: every
  subsumption between original names derived by the completion is derived by ELK on
  the original ontology (the converse may fail because `ELNormalizer.preprocess_ontology`
  drops axioms it cannot normalise; the test logs the dropped axiom count). `N(O)`'s
  auxiliary names belong to `N_C` and count towards the dimension.
* A fixture with roles (`A ⊑ ∃r.B`, `∃r.B ⊑ C`, `r ⊑ s`, chain) where GCI2/GCI3 checks
  must pass for both constructions.
* A Helly-satisfiable propositional fixture where the box builder succeeds in `d ≤ 2`.
* Property test: random small propositional TBoxes (≤ 8 names, ≤ 12 axioms): every
  certificate reported by `helly_check` is confirmed by the completion (the three
  pairwise conjunctions satisfiable, the triple unsatisfiable); whenever the box
  builder returns a model, the checker accepts it and it is non-degenerate; `d` used
  and `BoxModelNotFound` outcomes are logged (not failures).

## 8. What is left to the builder

* Exact file-level code organisation inside the layout above; naming of private
  helpers; LP/MILP matrix assembly details; big-M and δ defaults (document them).
* How to obtain role inclusions/chains from the OWLAPI ontology in mOWL's JPype idiom
  (look at `mowl/ontology/normalize.py` and `mowl/projection` for examples).
* Sphinx pages mirroring `docs/source/embedding_el`.
* Environment setup as in Sec. 3; record the exact commands in `docs/source/geometric/`
  and in the PR description.

## 9. References to cite in docstrings

Gutiérrez-Basulto & Schockaert, KR 2018 (arXiv:1805.10461) Prop. 3, Prop. 4;
Lacerda, Ozaki, Guimarães, TGDK 2(3):2, 2024, Defs. 2, 8, 12, Thms. 13, 16, 18, 25;
Leemhuis & Kutz, NeSy 2025 (PMLR 284:303–321) Defs. 1, 3, 6, Props. 4, 5, 8;
Bourgaux, Guimarães, Koudijs, Lacerda, Ozaki, KR 2024, Defs. 5–6, Remark 2;
Kazakov, Krötzsch, Simančík, JAR 2014 (ELK); `moose/elpp` (Hoehndorf lab) for the
Python completion port; kobayashi-marust for the optional `elc` oracle.
