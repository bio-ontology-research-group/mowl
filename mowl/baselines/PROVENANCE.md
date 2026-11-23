# Provenance of the baselines

Every file under `mowl/baselines` is the direct implementation as it was on the
`new-methods` branch before the recomposition: commit 2d0ed505 (2026-11-22 17:56), the
last commit before the baseline area was added (6fb48678) and before the first recomposition
merge (3ae7cf6d). The source commit given is the last commit before 2d0ed505 that touched the
source file. Import paths (to the frozen copies here) and docstrings are not compared.

Status:

- **original**: the code is that of the source at the commit; the library code it came
  from was rewritten during the recomposition, so the comparison in `tests/baselines` is
  against the code before the rewrite.
- **unchanged since**: the code is that of the source at the commit, and the library file
  has the same code today (a copy of today's library is then the same as the original).
- **adapted**: what was changed, and why.

Checked by a script that compares the syntax tree of every function and method (without
docstrings and imports) with the source at 2d0ed505.

| baseline file | source @ commit | status |
|---|---|---|
| `base_models/elmodel.py` | `mowl/base_models/elmodel.py` @ 81f4910f | original (import paths inside functions point to the frozen copies) |
| `evaluation/ranks.py` | `mowl/evaluation/ont_ranking.py` @ c7559d14 | unchanged since (only `ordinal_ranks_by_counting` of the source file) |
| `geometric/attributed.py` | `mowl/geometric/attributed.py` @ 8f79c7bf | original; library code rewritten since |
| `geometric/cones.py` | `mowl/geometric/cones.py` @ 53bff7d3 | adapted: the original `ALConeInterpretation` split in two: the class in `geometric/core/cones.py`, its two constructions here in a subclass (code unchanged) |
| `geometric/convex_regions.py` | `mowl/geometric/convex_regions.py` @ 6d74667d | original; library code rewritten since |
| `geometric/core/attributed.py` | `mowl/geometric/attributed.py` @ 8f79c7bf | unchanged since |
| `geometric/core/checker.py` | `mowl/geometric/checker.py` @ 6d74667d | unchanged since |
| `geometric/core/completion.py` | `mowl/geometric/completion.py` @ 6d74667d | unchanged since |
| `geometric/core/cones.py` | `mowl/geometric/cones.py` @ 53bff7d3 | unchanged since (constructions moved to `geometric/cones.py` (code unchanged)) |
| `geometric/core/convex_regions.py` | `mowl/geometric/convex_regions.py` @ 6d74667d | unchanged since |
| `geometric/core/errors.py` | `mowl/geometric/errors.py` @ 6d74667d | unchanged since |
| `geometric/core/faithfulness.py` | `mowl/geometric/faithfulness.py` @ 74a4533f | unchanged since |
| `geometric/core/membership.py` | `mowl/geometric/membership.py` @ 6d74667d | original; library code rewritten since |
| `geometric/core/normal_forms.py` | `mowl/geometric/normal_forms.py` @ 6d74667d | unchanged since |
| `geometric/core/regions.py` | `mowl/geometric/regions.py` @ 6d74667d | unchanged since |
| `geometric/core/temporal_cones.py` | `mowl/geometric/temporal_cones.py` @ 8f79c7bf | original (`from_lasso_model` moved to `geometric/temporal_cones.py` (code unchanged)); library code rewritten since |
| `geometric/intervals.py` | `mowl/geometric/intervals.py` @ 6d74667d | original; library code rewritten since |
| `geometric/polyhedral.py` | `mowl/geometric/polyhedral.py` @ 6d74667d | original; library code rewritten since |
| `geometric/temporal_cones.py` | `mowl/geometric/temporal_cones.py` @ 8f79c7bf | adapted: the original `TemporalALConeInterpretation` split in two: the class in `geometric/core/temporal_cones.py`, `from_lasso_model` here in a subclass (code unchanged) |
| `models/acone/model.py` | `mowl/models/acone/model.py` @ a2c7affd | original; library code rewritten since |
| `models/alc_base.py` | `mowl/base_models/alcmodel.py` @ 7c50fee0 | original; library code rewritten since |
| `models/alcone/model.py` | `mowl/models/alcone/model.py` @ 2f6cbce0 | original; library code rewritten since |
| `models/bcbox/model.py` | `mowl/models/bcbox/model.py` @ cd672bd8 | original; library code rewritten since |
| `models/betae/model.py` | `mowl/models/betae/model.py` @ a2c7affd | original; library code rewritten since |
| `models/box_lattice/model.py` | `mowl/models/box_lattice/model.py` @ 7cae2e22 | original; library code rewritten since |
| `models/boxe/model.py` | `mowl/models/boxe/model.py` @ 238bae7e | original; library code rewritten since |
| `models/boxel/model.py` | `mowl/models/boxel/model.py` @ c20dff13 | adapted: pass-through option `load_role_axioms` of the base class added (24898c8f, default False = original behaviour) |
| `models/boxlite/embedding.py` | `mowl/models/boxlite/embedding.py` @ 125b9ae0 | original; library code rewritten since |
| `models/boxlite/model.py` | `mowl/models/boxlite/model.py` @ 4d9807af | original; library code rewritten since |
| `models/boxlite/program.py` | `mowl/models/boxlite/program.py` @ 125b9ae0 | original; library code rewritten since |
| `models/boxlite/solved_check.py` | `mowl/models/boxlite/checker.py` @ 125b9ae0 | original; library code rewritten since |
| `models/boxsel/model.py` | `mowl/models/boxsel/model.py` @ c20dff13 | adapted: pass-through option `load_role_axioms` of the base class added (24898c8f, default False = original behaviour) |
| `models/boxsquaredel/model.py` | `mowl/models/boxsquaredel/model.py` @ 05e7796a | unchanged since |
| `models/canonical_mixture/model.py` | `mowl/models/canonical_mixture/model.py` @ d68438d2 | original; library code rewritten since |
| `models/cate/categorical/edge.py` | `mowl/projection/categorical/edge.py` @ b3cd4edd | unchanged since |
| `models/cate/categorical/lattice.py` | `mowl/projection/categorical/lattice.py` @ 4b019000 | unchanged since |
| `models/cate/categorical/model.py` | `mowl/projection/categorical/model.py` @ e3a6701b | unchanged since |
| `models/cate/categorical/saturation.py` | `mowl/projection/categorical/saturation.py` @ 6a44a0ca | unchanged since |
| `models/cate/categorical/utils.py` | `mowl/projection/categorical/utils.py` @ b3cd4edd | unchanged since |
| `models/cate/model.py` | `mowl/models/cate/model.py` @ 52950ba3 | adapted: the registration of CatE as a family of `mowl.benchmarks.runner` at import is removed (a baseline does not register itself in the library) |
| `models/concept2box/model.py` | `mowl/models/concept2box/model.py` @ 90d7b556 | original; library code rewritten since |
| `models/cone_qe/model.py` | `mowl/models/cone_qe/model.py` @ a2c7affd | original; library code rewritten since |
| `models/dfalc/model.py` | `mowl/models/dfalc/model.py` @ 8a7ea883 | original; library code rewritten since |
| `models/disk_embeddings/model.py` | `mowl/models/disk_embeddings/model.py` @ aa5bb797 | original; library code rewritten since |
| `models/e2r/model.py` | `mowl/models/e2r/model.py` @ 4d9807af | original; library code rewritten since |
| `models/eike/model.py` | `mowl/models/eike/model.py` @ 5ae8973c | original; library code rewritten since |
| `models/elbe/model.py` | `mowl/models/elbe/model.py` @ ad32a7c1 | original; library code rewritten since |
| `models/elembeddings/model.py` | `mowl/models/elembeddings/model.py` @ a48a3a32 | unchanged since |
| `models/elkem/model.py` | `mowl/models/elkem/model.py` @ 7f1f5b80 | original; library code rewritten since |
| `models/emelpp/model.py` | `mowl/models/emelpp/model.py` @ f0c2ded7 | unchanged since |
| `models/expressive/model.py` | `mowl/models/expressive/model.py` @ 238bae7e | original; library code rewritten since |
| `models/faithel/model.py` | `mowl/models/faithel/model.py` @ 52950ba3 | original; library code rewritten since |
| `models/falcon/model.py` | `mowl/models/falcon/model.py` @ 0bb4f041 | original; library code rewritten since |
| `models/graph_kge/graph_pykeen_model.py` | `mowl/models/graph_kge/graph_pykeen_model.py` @ f2ed11b5 | original; library code rewritten since |
| `models/graph_kge/region_kge.py` | `mowl/models/graph_kge/region_kge.py` @ 238bae7e | original; library file removed since |
| `models/graph_random_walk/random_walk_w2v_model.py` | `mowl/models/graph_random_walk/random_walk_w2v_model.py` @ 4d9807af | adapted: walks through `_walk`, which sets `append=True` on the walker: the library walkers now empty the walk file before a walk (defect fix ae25c313); `append=True` is the original behaviour |
| `models/grounded/model.py` | `mowl/models/grounded/model.py` @ d68438d2 | original (import paths inside functions point to the frozen copies) |
| `models/gumbel_box/model.py` | `mowl/models/gumbel_box/model.py` @ 2871f057 | unchanged since |
| `models/hit/model.py` | `mowl/models/hit/model.py` @ 4d9807af | original; library code rewritten since |
| `models/hyperbolic_cones/model.py` | `mowl/models/hyperbolic_cones/model.py` @ aa5bb797 | unchanged since |
| `models/mcmultibox/model.py` | `mowl/models/mcmultibox/model.py` @ 01ebbc59 | original; library code rewritten since |
| `models/multiboxel/model.py` | `mowl/models/multiboxel/model.py` @ d68438d2 | original; library code rewritten since |
| `models/multielbe/model.py` | `mowl/models/multielbe/model.py` @ d68438d2 | original; library code rewritten since |
| `models/octagon/model.py` | `mowl/models/octagon/model.py` @ 3dd19f80 | original; library code rewritten since |
| `models/ont/model.py` | `mowl/models/ont/model.py` @ 4d9807af | original; library code rewritten since |
| `models/order_base.py` | `mowl/models/order_base.py` @ 38d1a6bc | original; library code rewritten since |
| `models/order_embeddings/model.py` | `mowl/models/order_embeddings/model.py` @ 5fa22b0b | unchanged since |
| `models/prism/model.py` | `mowl/models/prism/model.py` @ d68438d2 | original; library code rewritten since |
| `models/query/expression.py` | `mowl/nn/query/expression.py` @ 90bc8ed8 | unchanged since |
| `models/query/graph.py` | `mowl/models/query/graph.py` @ 90bc8ed8 | original; library code rewritten since |
| `models/query/model.py` | `mowl/models/query/model.py` @ 90bc8ed8 | original; library code rewritten since |
| `models/query/sampling.py` | `mowl/models/query/sampling.py` @ 90bc8ed8 | original; library code rewritten since |
| `models/query2box/model.py` | `mowl/models/query2box/model.py` @ a2c7affd | original; library code rewritten since |
| `models/regd/model.py` | `mowl/models/regd/model.py` @ 27c670d7 | original; library code rewritten since |
| `models/reshuffle/model.py` | `mowl/models/reshuffle/model.py` @ 7d6c2148 | original (import paths inside functions point to the frozen copies) |
| `models/shadow_cones/model.py` | `mowl/models/shadow_cones/model.py` @ b76568ee | original; library code rewritten since |
| `models/syntactic/w2v_model.py` | `mowl/models/syntactic/w2v_model.py` @ 4d9807af | adapted: `generate_corpus` passes `annotation_entity_form="bracketed iri"`: the library corpus functions now write plain IRIs (defect fix 25333501); the bracketed form is the original behaviour |
| `models/taxonomy_base.py` | `mowl/models/order_embeddings/base.py` @ 0fb69333 | original; library code rewritten since |
| `models/transbox/model.py` | `mowl/models/transbox/model.py` @ 7d2590c6 | unchanged since |
| `nn/alc/alcone/losses.py` | `mowl/nn/alc/alcone/losses.py` @ 53bff7d3 | original; library file removed since |
| `nn/alc/alcone/module.py` | `mowl/nn/alc/alcone/module.py` @ 2f6cbce0 | original; library code rewritten since |
| `nn/alc/dfalc/module.py` | `mowl/nn/alc/dfalc/module.py` @ 8a7ea883 | original; library code rewritten since |
| `nn/alc/e2r/losses.py` | `mowl/nn/alc/e2r/losses.py` @ d75bf9eb | original; library code rewritten since |
| `nn/alc/e2r/module.py` | `mowl/nn/alc/e2r/module.py` @ d75bf9eb | original; library code rewritten since |
| `nn/alc/falcon/module.py` | `mowl/nn/alc/falcon/module.py` @ 0bb4f041 | original; library code rewritten since |
| `nn/alc/module.py` | `mowl/nn/alc/module.py` @ ee0dc02c | unchanged since |
| `nn/el/box_lattice/module.py` | `mowl/nn/el/box_lattice/module.py` @ 7cae2e22 | original; library code rewritten since |
| `nn/el/boxel/losses.py` | `mowl/nn/el/boxel/losses.py` @ c20dff13 | original; library file removed since |
| `nn/el/boxel/module.py` | `mowl/nn/el/boxel/module.py` @ 162ff5fd | original; library code rewritten since |
| `nn/el/boxsel/losses.py` | `mowl/nn/el/boxsel/losses.py` @ c20dff13 | original; library file removed since |
| `nn/el/boxsel/module.py` | `mowl/nn/el/boxsel/module.py` @ c20dff13 | original; library code rewritten since |
| `nn/el/boxsquaredel/losses.py` | `mowl/nn/el/boxsquaredel/losses.py` @ 05e7796a | original; library file removed since |
| `nn/el/boxsquaredel/module.py` | `mowl/nn/el/boxsquaredel/module.py` @ 0fb69333 | original; library code rewritten since |
| `nn/el/canonical_mixture/module.py` | `mowl/nn/el/canonical_mixture/module.py` @ d68438d2 | original; library code rewritten since |
| `nn/el/concept2box/losses.py` | `mowl/nn/el/concept2box/losses.py` @ 238bae7e | original; library file removed since |
| `nn/el/concept2box/module.py` | `mowl/nn/el/concept2box/module.py` @ 90d7b556 | original; library code rewritten since |
| `nn/el/eike/module.py` | `mowl/nn/el/eike/module.py` @ c04d9fe7 | original; library code rewritten since |
| `nn/el/elbe/losses.py` | `mowl/nn/el/elbe/losses.py` @ 9203e5d4 | original; library file removed since |
| `nn/el/elbe/module.py` | `mowl/nn/el/elbe/module.py` @ ad32a7c1 | original; library code rewritten since |
| `nn/el/elem/dele_losses.py` | `mowl/nn/el/elem/losses.py` @ a48a3a32 | original (only the DELE loss functions of the source file); library file removed since |
| `nn/el/elem/losses.py` | `mowl/nn/el/elem/losses.py` @ a48a3a32 | original; library file removed since |
| `nn/el/elem/module.py` | `mowl/nn/el/elem/module.py` @ 29c08190 | original; library code rewritten since |
| `nn/el/elkem/losses.py` | `mowl/nn/el/elkem/losses.py` @ 1ad4099a | original; library file removed since |
| `nn/el/elkem/module.py` | `mowl/nn/el/elkem/module.py` @ 85276b76 | original; library code rewritten since |
| `nn/el/elmodule.py` | `mowl/nn/el/elmodule.py` @ 162ff5fd | unchanged since |
| `nn/el/emelpp/losses.py` | `mowl/nn/el/emelpp/losses.py` @ 6a506593 | original; library file removed since |
| `nn/el/emelpp/module.py` | `mowl/nn/el/emelpp/module.py` @ 6a506593 | original; library code rewritten since |
| `nn/el/grounded/losses.py` | `mowl/nn/el/grounded/losses.py` @ 6a68e658 | original; library file removed since |
| `nn/el/grounded/module.py` | `mowl/nn/el/grounded/module.py` @ d68438d2; `mowl/nn/el/structural.py` @ d68438d2 | original; library code rewritten since |
| `nn/el/grounded/queries.py` | `mowl/nn/el/grounded/queries.py` @ 0f19bd8b | original; library code rewritten since |
| `nn/el/grounded/sampling.py` | `mowl/nn/el/grounded/sampling.py` @ 6a68e658 | original; library code rewritten since |
| `nn/el/gumbel_box/module.py` | `mowl/nn/el/gumbel_box/module.py` @ 2871f057 | original; library code rewritten since |
| `nn/el/mcmultibox/module.py` | `mowl/nn/el/mcmultibox/module.py` @ d68438d2 | original; library code rewritten since |
| `nn/el/mcmultibox/volume.py` | `mowl/nn/el/mcmultibox/volume.py` @ a226e197 | original; library file removed since |
| `nn/el/multibox_utils.py` | `mowl/nn/el/multibox_utils.py` @ d68438d2 | unchanged since |
| `nn/el/multiboxel/losses.py` | `mowl/nn/el/multiboxel/losses.py` @ d68438d2 | original; library file removed since |
| `nn/el/multiboxel/module.py` | `mowl/nn/el/multiboxel/module.py` @ 2ac714e6 | original; library code rewritten since |
| `nn/el/multiboxel/pointloss.py` | `mowl/nn/el/pointloss.py` @ d68438d2 | original; library file removed since |
| `nn/el/multiboxel/softagg.py` | `mowl/nn/el/softagg.py` @ d68438d2 | original; library file removed since |
| `nn/el/multiboxel/structural_losses.py` | `mowl/nn/el/structural.py` @ d68438d2 | original; library code rewritten since |
| `nn/el/multielbe/losses.py` | `mowl/nn/el/multielbe/losses.py` @ d68438d2 | original; library file removed since |
| `nn/el/multielbe/module.py` | `mowl/nn/el/multielbe/module.py` @ d68438d2 | original; library code rewritten since |
| `nn/el/octagon/geometry.py` | `mowl/nn/el/octagon/geometry.py` @ 3dd19f80 | original; library code rewritten since |
| `nn/el/octagon/losses.py` | `mowl/nn/el/octagon/losses.py` @ 3dd19f80 | original; library file removed since |
| `nn/el/octagon/module.py` | `mowl/nn/el/octagon/module.py` @ d68438d2 | original; library code rewritten since |
| `nn/el/order_embeddings/base.py` | `mowl/nn/el/order_embeddings/base.py` @ 5fa22b0b | unchanged since |
| `nn/el/order_embeddings/module.py` | `mowl/nn/el/order_embeddings/module.py` @ 5fa22b0b | original; library code rewritten since |
| `nn/el/per_axiom_column.py` | `mowl/nn/el/per_axiom_column.py` @ 0fb69333 | unchanged since |
| `nn/el/prism/losses.py` | `mowl/nn/el/prism/losses.py` @ d68438d2 | original; library file removed since |
| `nn/el/prism/module.py` | `mowl/nn/el/prism/module.py` @ 2ac714e6 | original; library code rewritten since |
| `nn/el/regd/dissimilarity.py` | `mowl/nn/el/regd/dissimilarity.py` @ 97ed2467 | original; library file removed since |
| `nn/el/regd/module.py` | `mowl/nn/el/regd/module.py` @ 27c670d7 | original; library code rewritten since |
| `nn/el/strict_negatives.py` | `mowl/nn/el/strict_negatives.py` @ 162ff5fd | unchanged since |
| `nn/el/structural.py` | `mowl/nn/el/structural.py` @ d68438d2 | original; library code rewritten since |
| `nn/el/transbox/code_losses.py` | `mowl/nn/el/transbox/code_losses.py` @ 34cbef8d | original; library file removed since |
| `nn/el/transbox/losses.py` | `mowl/nn/el/transbox/losses.py` @ 1de90698 | original; library file removed since |
| `nn/el/transbox/module.py` | `mowl/nn/el/transbox/module.py` @ 162ff5fd | original; library code rewritten since |
| `nn/graph/bcbox/losses.py` | `mowl/nn/graph/bcbox/losses.py` @ 2871f057 | original; library file removed since |
| `nn/graph/bcbox/module.py` | `mowl/nn/graph/bcbox/module.py` @ 3489ac20 | original; library code rewritten since |
| `nn/graph/cate/losses.py` | `mowl/nn/graph/cate/losses.py` @ 6a44a0ca | original; library file removed since |
| `nn/graph/cate/module.py` | `mowl/nn/graph/cate/module.py` @ 6a44a0ca | original (import paths inside functions point to the frozen copies) |
| `nn/graph/expressive/module.py` | `mowl/nn/graph/expressive/module.py` @ 238bae7e | original; library file removed since |
| `nn/graph/interactions.py` | `mowl/nn/graph/expressive/module.py` @ 238bae7e; `mowl/nn/graph/reshuffle/module.py` @ 35d9c5e0 | original; library file removed since |
| `nn/graph/reshuffle/module.py` | `mowl/nn/graph/reshuffle/module.py` @ 35d9c5e0 | original; library file removed since |
| `nn/lm/hit.py` | `mowl/nn/lm/hit.py` @ c7559d14 | original; library code rewritten since |
| `nn/lm/hyperbolic.py` | `mowl/nn/lm/hyperbolic.py` @ c7559d14 | original; library code rewritten since |
| `nn/lm/ont.py` | `mowl/nn/lm/ont.py` @ c7559d14 | original; library code rewritten since |
| `nn/lm/text_encoders.py` | `mowl/nn/lm/encoders.py` @ c7559d14 | original; library code rewritten since |
| `nn/numerics.py` | `mowl/nn/numerics.py` @ 2871f057 | unchanged since |
| `nn/order/cones.py` | `mowl/nn/order/cones.py` @ 61fdc45c | original; library file removed since |
| `nn/order/disks.py` | `mowl/nn/order/disks.py` @ aa5bb797 | original; library file removed since |
| `nn/order/manifolds.py` | `mowl/nn/order/manifolds.py` @ 1f12b11b | original; library code rewritten since |
| `nn/order/module.py` | `mowl/nn/order/module.py` @ b76568ee | original; library code rewritten since |
| `nn/query/base.py` | `mowl/nn/query/geometry.py` @ a18af38a | original; library code rewritten since |
| `nn/query/geometry.py` | `mowl/nn/query/geometry.py` @ a18af38a | original; library code rewritten since |
| `training/batching.py` | `mowl/training/batching.py` @ 48cd2112 | original; library code rewritten since |
| `training/constraints.py` | `mowl/training/constraints.py` @ fa438240 | unchanged since |
| `training/grounded.py` | `mowl/training/grounded.py` @ fa438240 | unchanged since |
| `training/lm_reference.py` | `mowl/training/lm_reference.py` @ a361b254 | original; library code rewritten since |
| `training/regime.py` | `mowl/training/regime.py` @ 162ff5fd | original; library code rewritten since |
| `training/schedules.py` | `mowl/training/schedules.py` @ fa438240 | original; library code rewritten since |
| `training/structure.py` | `mowl/training/structure.py` @ fa438240 | unchanged since |
| `kge/model.py` | `mowl/kge/model.py` @ 6cc74711 (as released in v2.1.0, f8b2cb07) | original: `KGEModel` trained by PyKEEN's `SLCWATrainingLoop`; the library version trains in mOWL's loop (round 7) |

## Replaced by the original in round 7

These files had been copies of the library as it was after the recomposition; they are now the
original code (2d0ed505):

- `base_models/elmodel.py` (new): the EL model base with its own training loop, loss sum and
  negative sampling. Before, every baseline EL model subclassed the library's
  `mowl.base_models.elmodel.EmbeddingELModel` and trained through `mowl.training.loop`,
  `weighted_mean_loss` and `corrupt_column_blocks`.
- `models/{hit,ont,cate,bcbox,e2r,mcmultibox,multiboxel,multielbe}/model.py`: the whole original
  model class. Before, each was split into a `base.py` copied from the library's rewritten
  common base (`mowl/models/<method>/common.py`) plus a few methods; `base.py`,
  `hit/text_ranking.py`, `cate/lattice_ranking.py` and `e2r/violations.py` (library code written
  during the recomposition) are removed.
- `training/regime.py`, `training/schedules.py`: the original regime base and schedules (the
  copies had `replaces_validation`, `skips_empty_steps` and a `temperature` attribute in
  `TemperatureAnnealing`, added to the library later). `training/batching.py` and
  `training/lm_reference.py` (new) are the originals the schedules and HiT/OnT import.
- `nn/el/structural.py`: the original `inside_measure` (the copy had the rewritten one: `max`
  instead of `amax`, which splits the gradient differently on ties).
- `geometric/core/temporal_cones.py`, `geometric/core/regions.py`: a helper the library added
  later (`_individual_point`, `PairHull.of_pairs`) removed.
- `nn/el/grounded/{sampling,queries}.py`: the original `isinstance(module, GroundedELModule)` check
  instead of the interface check that let the baseline samplers read library geometries.
