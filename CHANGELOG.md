# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- Added the `GALENJackermeier2024Dataset`, `GOJackermeier2024Dataset` and `AnatomyJackermeier2024Dataset` builtin datasets: the GALEN, GO and ANATOMY ontologies with the 80/10/10 subsumption prediction split distributed with the Box²EL reference implementation, also used by TransBox (WWW 2025). The class IRIs of the Anatomy dataset are opaque generated IRIs, because the reference benchmark ships no class names for ANATOMY [#157][i157].
- Added extraction of the two EL++ role axiom normal forms — role inclusion (`R ⊑ S`) and role chain (`R ∘ T ⊑ S`, which is also how transitive property declarations are represented). New `RoleInclusion` and `RoleChain` axiom classes and an `extract_role_axioms()` function in `mowl.ontology.normalize`; new `RoleAxiomDataset`, `RoleInclusionDataset` and `RoleChainDataset` in `mowl.datasets.el`. Role axioms over non-atomic property expressions (e.g. inverses) are skipped with a warning, as are chains of more than two properties [#157][i157].
- Added a `load_role_axioms` argument to `ELDataset` and `EmbeddingELModel`, defaulting to `False`. When enabled, `get_gci_datasets()` also returns the `role_inclusion` and `role_chain` datasets; when left at the default nothing changes for existing code, including for the shipped ontologies that do contain role axioms (PPI yeast, for instance, has 3 role inclusions and 6 role chains) [#157][i157].
- Added `ELModule.role_inclusion_loss` and `ELModule.role_chain_loss` (not implemented in the base class) and the `ELModule.role_axiom_capable` class flag. Loading the role axioms with a module that does not declare the flag raises `NotImplementedError` at the start of training rather than failing mid-epoch, so `ELEmbeddings`, `ELBE`, `BoxSquaredEL` and `BoxEL` are unaffected [#157][i157].
- Added `mowl.datasets.default_data_root`, which resolves the dataset cache directory, and a `MOWL_DATA_ROOT` environment variable to override it [#150][i150].
- Added a `data_root` argument to every builtin dataset, so the download location stays overridable per dataset [#150][i150].
- Added support for multi-column negative sampling in `EmbeddingELModel`: a GCI's `corrupt_column` now accepts a single column index or a list of column indices, and `index_pool` accepts a single pool name or a list with one pool per corrupted column. For *K* corrupted columns, the returned negative set is the concatenation of the *K* single-column negative sets (one per column, each corrupting only its own column), so each positive sample is paired with *K* negatives. Single-column configurations behave exactly as before. The configuration is validated at the start of `train()` rather than mid-epoch [#156][i156].
- Added the `TransBox` model (`mowl.models.TransBox`) with `TransBoxModule` (`mowl.nn.el.transbox`): concept box embeddings with translational role embeddings for :math:`\mathcal{EL}^{++}` following Yang, Chen and Sattler (WWW 2025). Atomic concepts and roles are embedded as axis-aligned boxes and individuals as points, so complex :math:`\mathcal{EL}^{++}` concepts are embedded by composing the atomic boxes; GCIs of the form `C ⊑ ∃R.D` are trained against the ∃rall-enhanced right-hand side and negated on both sides (two negatives per positive), and the module implements the role axiom losses, so the model accepts `load_role_axioms=True` [#155][i155].
- `mowl.geometric` package for constructive exact geometric models, model checking, and box-representability diagnostics of normalised $\mathcal{EL}^{++}$ ontologies:
  - `NormalisedOntology`, `normalise`, `to_owl`, and normal-form dataclasses `NF1`, `NF2`, `NF3`, `NF4`, `NF6`, `NF7` (`TOP`, `BOTTOM`, `Axiom`): extract normalised $\mathcal{EL}^{++}$ TBox and RBox axioms (splitting role chains of length $> 2$) and record dropped axioms (Baader, Brandt & Lutz, 2005; Kazakov, Krötzsch & Simančík, 2014).
  - `Completion` and `CompletionResult`: worklist-driven ELK saturation rules R1–R7 deriving all entailed concept-name subsumptions, role edges, and $\bot$ propagations (Kazakov, Krötzsch & Simančík, 2014).
  - `CanonicalModel`: finite canonical model $\mathcal{I} \models \mathcal{O}$ with exact concept-name subsumption ($A^{\mathcal{I}} \subseteq B^{\mathcal{I}} \iff \mathcal{O} \models A \sqsubseteq B$) and unsatisfiability ($A^{\mathcal{I}} = \emptyset \iff \mathcal{O} \models A \sqsubseteq \bot$), raising `InconsistentOntology` iff $\mathcal{O} \models \top \sqsubseteq \bot$ (Kazakov, Krötzsch & Simančík, 2014).
  - `IntervalUnion`, `RectUnion`, `Box`, `Face`, `ThickFace`, `PairHull`, `EMPTY_1D`, and `simplex_points`: pure-NumPy region and role-region representations with exact set operations and H-representations (Gutiérrez-Basulto & Schockaert, 2018; Lacerda, Ozaki & Guimarães, 2024).
  - `GeometricInterpretation`, `DictInterpretation`, and `ModelChecker`: exact decision procedures for axiom satisfaction (`NF1`–`NF4`, `NF6`, `NF7`, and $\bot$ forms), non-degeneracy, and strong faithfulness on concept names across interval, polyhedral, and propositional box interpretations (Gutiérrez-Basulto & Schockaert, 2018; Lacerda, Ozaki & Guimarães, 2024; Bourgaux et al., 2024).
  - `ConvexPolyhedralModel`: constructive convex polyhedral model in $\mathbb{R}^{|\Delta^{\mathcal{I}}| - 1}$ guaranteed to be an exact, non-degenerate, strongly faithful model of any consistent normalised $\mathcal{EL}^{++}$ ontology, verified by `ModelChecker` on construction (Gutiérrez-Basulto & Schockaert, 2018; Lacerda, Ozaki & Guimarães, 2024).
  - `IntervalUnionModel`: constructive 1-D interval-union model guaranteed to be an exact, non-degenerate, strongly faithful model of any consistent normalised $\mathcal{EL}^{++}$ ontology, verified by `ModelChecker` on construction (Lacerda, Ozaki & Guimarães, 2024; Federico, Leemhuis & Kutz, 2026).
  - `helly_check`, `HellyReport`, and `HellyCertificate`: Helly-satisfiability diagnostic enumerating minimal jointly unsatisfiable concept sets and emitting sound certificates (non-degenerate certificates exclude every non-degenerate box model; strong certificates exclude conjunction-faithful box models), exact for the strong reading on the propositional fragment when `complete=True`; `box_representable_* = True` means no Helly obstruction was found, not that a box model exists (Eckhoff, 1988; Leemhuis & Kutz, 2025; Leemhuis & Kutz, 2026).
  - `BoxModel` and `BoxModelBuilder`: MILP-based search (`scipy.optimize.milp`) for axis-parallel box models of propositional $\mathcal{EL}_\bot$ ontologies after a Helly pre-check, verifying any found model (including a post-checked time-limited incumbent) with `ModelChecker`, raising `NotBoxRepresentable` on a non-degenerate Helly certificate in both `faithful` modes, and raising `BoxModelNotFound` when inconclusive (Leemhuis & Kutz, 2025).
  - `class_points`, `to_bounding_boxes`, and `boxes_as_tensors` (`mowl.geometric.export`): export polyhedral vertex sets, documented lossy bounding boxes, and `class_to_id`-ordered box arrays (`(lo, hi)` or `(center, offset)`) for seeding `ELBE` and `BoxSquaredEL` (Gutiérrez-Basulto & Schockaert, 2018; Leemhuis & Kutz, 2025).
  - `GeometricError`, `InconsistentOntology`, `UnsupportedAxiom`, `UnsupportedRegion`, `NotBoxRepresentable`, `BoxModelNotFound`, `InternalConsistencyError`, and `MissingDependency`: domain exception hierarchy distinguishing proved non-representability from inconclusive MILP search.
  - `certificate_triangles` (`mowl.geometric.helly`): scalable $O(\sum_v \deg(v)^2)$ finder for non-degenerate singleton Helly certificates whose three pairwise conjunctions are direct `NF2` definitions (sound for all returned triangles; not complete for conjunctions defined through chains of equivalences or higher-arity certificates).
  - `ExactBoxInit` (`mowl.geometric.init`): exact-model box initialisation (`method="projected"` with Gaussian projection and deterministic k-means for $k > 1$, or `method="milp"` via `BoxModelBuilder` with automatic fallback to `"projected"` on `NotBoxRepresentable`/`BoxModelNotFound`) and unweighted $L_2^2$ `anchor_loss` for `BoxSquaredELModule`, `ELBEModule`, and `MultiBoxELModule`. Under `"projected"` mode with $k = 1$, guarantees $\text{box}(C) \subseteq \text{box}(D)$ and zero `gci0` loss at margin $0$ for every entailed atomic subsumption $C \sqsubseteq D$ with a satisfiable subclass $C$ (unsatisfiable subclasses are point boxes at the origin and are not covered; disjointness is not guaranteed; classes absent from the canonical model are skipped as `NaN` rows and ignored by `apply_to_module` and `anchor_loss`).
- Added `MultiBoxEL` (`mowl.models.MultiBoxEL`) and `MultiBoxELModule` (`mowl.nn.el.multiboxel`): multi-component Box$^2$EL embedding model representing each ontology class as a union of $k = \text{n\_components}$ axis-parallel boxes in $\mathbb{R}^d$ with a shared class bump and Box$^2$EL role and individual semantics (`effective_components` reports non-degenerate, non-nested component counts). At $k = 1$ with `component_reg=0.0`, all ten GCI losses (including `gci1` batches with empty intersections) and `regularization_loss` equal `BoxSquaredEL` to within `1e-6`. In `gci1`, a detached argmin over emptiness selects a designated component pair and a detached gate enforces inclusion only on intersecting pairs (plus the designated pair); in `gci1_bot`, zero loss at $\gamma = 0$ implies disjointness across all $k^2$ component pairs (up to boundary contact) but not conversely. `MultiBoxELModule` does not implement the role axiom losses (`role_axiom_capable = False`), so `MultiBoxEL` does not support `load_role_axioms=True` (Federico, Leemhuis & Kutz, 2026).
- Added optional `init_boxes: Optional[ExactBoxInit] = None` and `anchor_weight: float = 0.0` keyword arguments to `BoxSquaredEL` and `ELBE` (and `MultiBoxEL`), applying `init_boxes.apply_to_module` in `init_module()` and adding `anchor_weight * init_boxes.anchor_loss(self.module)` in `get_regularization_loss()` (raising `ValueError` if `anchor_weight < 0`). `apply_to_module(module, class_index_dict=None)` matches initialisation rows (indexed by `dataset.class_to_id`) to module rows by IRI through the model's `class_index_dict`; the rows of the auxiliary classes that normalisation appends to `class_index_dict` (IRIs under `http://mowl.borg/aux/`) keep their random initialisation and are excluded from `anchor_loss`. Without `class_index_dict`, the module must have exactly `len(class_to_id)` class rows. Defaults leave behaviour unchanged.
### Changed
- Changed `RemoteDataset` to cache datasets under `$XDG_CACHE_HOME/mowl/datasets` (falling back to `~/.cache/mowl/datasets`) instead of the current working directory. Datasets are now shared between working directories rather than re-downloaded per directory, and running the tests or an example no longer scatters dataset directories through the checkout. Existing downloads in a working directory are not picked up and will be fetched once into the cache [#150][i150].
- Changed the test suite to resolve dataset files through the cache instead of assuming pytest runs from the repository root.
### Fixed
- Fixed `ELNormalizer` corrupting the normal forms of every ontology with nested class expressions. The jcel normaliser was given a second, fresh `IntegerOntologyObjectFactoryImpl` whose entity counter restarted at the first usable identifier, so the auxiliary classes it introduces to flatten nested expressions received the integer identifiers of named classes of the ontology. On reverse translation each auxiliary class became an unrelated named class, and axioms with auxiliary identifiers beyond the named range were dropped silently. For example, `B ⊓ ∃r.C ⊑ A` became `∃r.C ⊑ B` and `B ⊑ A` instead of `∃r.C ⊑ X` and `B ⊓ X ⊑ A`; on the Pizza ontology the normalised TBox made 35 named classes unsatisfiable instead of 2. **Models trained with earlier versions on ontologies with nested class expressions (e.g. the GDA datasets) were trained on these corrupted normal forms.** The normaliser now shares the translator's factory, names auxiliary classes and object properties `http://mowl.borg/aux/<tag>#AuxiliaryEntity<n>`, where `<tag>` is derived from the SHA-1 of the preprocessed input TBox (names are stable for an ontology, and identical preprocessed TBoxes share the same namespace), and raises instead of dropping an axiom it cannot translate back. Normal forms that are not GCIs (functional object properties, ranges) are skipped with a warning. The on-disk cache is now `<ontology>_mowl_el_normalized_v2.owl`, so existing caches with corrupted normal forms are no longer read and are recomputed.
- Fixed `ELNormalizer` silently replacing inverse roles: jcel represents `r⁻` by an auxiliary object property unrelated to `r`, so `A ⊑ ∃r⁻.B` became `A ⊑ ∃AuxiliaryEntityN.B`. Axioms using `ObjectInverseOf` are now dropped by `preprocess_ontology` like the other unsupported constructs, which are counted (`ELNormalizer.num_dropped_axioms`) and reported in one warning.
- Fixed EL models not having embeddings for the auxiliary classes of the normal forms: `EmbeddingELModel` now normalises the training ontology eagerly, once, at construction, and appends the auxiliary classes and object properties to `class_index_dict` and `object_property_index_dict` after the dataset entities, whose indices are unchanged. Consequently `class_embeddings` and `object_property_embeddings` now include auxiliary keys, and checkpoints saved with earlier versions will not load into models of ontologies with nested class expressions (the embedding tables have more rows). `dataset.classes` and the evaluation classes are not modified, so auxiliary classes are never ranking candidates, and the `"classes"` negative sampling pool contains named classes only (`named_class_index_dict`). `ELDataset` skips axioms that mention entities missing from given index dictionaries (e.g. auxiliary classes of a validation or testing ontology) with one warning giving their number, instead of raising `KeyError`.
- Fixed `ELBE`'s `gci1_loss` returning an `(n, n)` matrix instead of one loss per sample. It added a column of shape `(n, 1)` to a flat vector of shape `(n,)`, which broadcasts, so every GCI1 score was combined with every other sample's score and the loss minimised was not the intended one. Training still ran, because the mean of the matrix is a scalar. The bug dates from 2023 and was invisible to the tests, which only ever passed single-row batches, where `(1, 1)` and `(1,)` broadcast to `(1, 1)`. `ELEmModule` and `BoxSquaredELModule` were checked and are unaffected.
- Fixed `RemoteDataset._download` hanging forever on a connection that goes silent mid-transfer. `requests` has no default timeout, so a stalled socket never raised; the request now sets a connect and read timeout and retries with backoff [#149][i149].
- Fixed an interrupted download leaving a truncated tarball that later runs mistook for a complete one. The transfer now writes to a temporary file and is renamed into place only once it finishes [#149][i149].

## [2.1.0] - 2026-08-16
### Added
- Added the `GDAProjector`, which unfolds nested existential restrictions into edges whose relation is the composition of the roles traversed, for gene-disease association graphs [#142][i142].
- Added the `FALCON` model (`FALCONModel`) for embedding the ALC language, with an `ALCModule` neural base class, a `FALCONSubsumptionEvaluator`, and a `FALCONVisualizer` membership heatmap [#71][i71].
- Added `BoxEL` model (with `BoxELPPI` and `BoxELGDA` examples) wrapping `BoxELModule` [#70][i70].
- Added 2-D geometric EL visualizers for ELEm, ELBE, and Box²EL, plus a role-visualization API showing Box²EL relation pairs.
- Added an `epoch_callback` argument to `train()` and a geometric EL training animation gallery example.
- Exposed `neg_sampling_gcis` in `ELEmbeddings`, `ELBE`, and `BoxSquaredEL`.
- Added tests for all 13 previously untested `OWLAPIAdapter` methods.
### Changed
- Documented the available EL models (`ELEmbeddings`, `ELBE`, `BoxSquaredEL`, `BoxEL`) in the embedding guide.
- Replaced the interactive jshtml animation with an autoplaying GIF in the gallery and docs.
- Bumped CI GitHub Actions (setup-java, setup-python, checkout, upload-artifact, download-artifact, github-script, sigstore, conda setup-miniconda).
- Raised build requirements to `setuptools >=83.0.0` and `setuptools-scm >=10.2.1`.
### Removed
- Removed ontology load caches that polluted shared test fixtures.
### Fixed
- Fixed `BaseRankingEvaluator` scoring the wrong entities when the evaluation pool is a subset of the ontology's classes. The swept column carried pool positions where the evaluation model expected global entity ids, so `GDAEvaluator`, `PPIEvaluator` and `SubsumptionEvaluator` silently produced meaningless ranking metrics [#146][i146].
- Fixed the ALC documentation doctest by dropping empty `testoutput` blocks.
- Repaired the `ELEmbeddings` and `ELBE` gallery examples, which had not been updated for the 2.0.0 API and could not run: `epochs` was passed to `__init__` instead of `train()`, and the ELBE example still imported `ELBoxPPI` from the removed `mowl.models.elboxembeddings` module. Their published evaluation metrics have been regenerated.

## [2.0.0] - 2026-05-12
### Added
- Added caching support for `ELNormalizer`: normalized ontologies can now be saved to disk and loaded on subsequent runs, significantly speeding up repeated normalization of the same ontology.
- Added `ontology_path` and `use_cache` parameters to `ELNormalizer.normalize()` and `ELDataset`.
- `EmbeddingELModel` now automatically uses caching when the dataset provides ontology paths (e.g., `PathDataset`).
- Added `BoxSquaredELGDA` and `BoxSquaredELPPI` model classes.
- Added `evaluate_ppi()` convenience method on PPI models (`ELBEPPI`, `ELEmPPI`).
- Added `neg_capable_gcis` property to `ELModule`; negative sampling now extended to all non-bot GCIs.
- Added tests for EL models (ELBE, ELBEPPI, ELBEGDA, ELEmGDA, BoxSquaredEL).
### Changed
- Refactored EL model training loop: `train(epochs, validate_every=1)` now has a standardised signature across all EL models.
- `eval_gci_name` is now a validated property on `EmbeddingELModel`; it must be set explicitly before calling `train` with validation or `evaluate`.
### Removed
- `ELBoxEmbeddings` alias removed. Use `ELBE` directly.
- `epochs` parameter removed from EL model `__init__`. Pass `epochs` to `train()` instead.
### Fixed
- `ELBEPPI` now automatically sets `PPIEvaluator` in `__init__`, matching `ELEmPPI` behavior.
- Fixed `evaluate_ppi()` in `ELBEPPI` and `ELEmPPI` to pass required `testing_ontology` argument.

## [1.0.2]
### Changed
- Upgraded scipy dependency from 1.15.0 to 1.16.0
- Upgraded JPype1 dependency from 1.5.0 to 1.5.1
- Pinned class-resolver to version 0.5.4 [#97][i97]
### Fixed
- Fix bug handling anonymous individuals in the OWL2VecStarProjector class

## [1.0.1]
### Fixed 
- Fix bug on CI pipeline to publish on PyPI

## [1.0.0]
### Added
- Added `load_corpus` to `mowl.base_models.SyntacticModel`
- Added `.jar` files to enable pip install from GitHub
- Tested with Python 3.12
- Refactored Evaluation module.
- Refactored `model` object and introduced `model.train()` and `model.evaluate()` methods.
- Added `GDADatasetV2` and `GDADatasetV2EL`. Datasets were introduced in [https://github.com/bio-ontology-research-group/gda_analysis](https://github.com/bio-ontology-research-group/gda_analysis)
- Extended experiment set to match [Ontology Embeddings Survey paper](https://arxiv.org/html/2406.10964v1).
- Added mOWL format of GO and FoodOn datasets from [OWL2VecStar repository](https://github.com/KRR-Oxford/OWL2Vec-Star)
- Added PPI Human dataset (String v12.0)
- Added `ontology_extended.owl` version to PPI Yeast and PPI Human datasets. This extensions are required for `CatE`.
- Added `PPIEvaluator`, `PPICatEEvaluator`, `GDAEvaluator` and `SubsumptionEvaluator`

### Changed 
- Removed `.mean()` for GCI losses in BoxSquaredELModule
- Property `evaluation_classes` in `mowl.datasets.builtin.PPIYeastDataset` returns a pair of `OWLClasses` objects instead of a single `OWLClasses` object.
- Updated CatE projection method based on latest version of paper (Nesy 2024 submission) ([#59][i59])
### Deprecated
- `mowl.nn.ELBoxModule` changed name to `mowl.nn.ELBEModule`
### Removed
- End of support for Python 3.8 based on https://endoflife.date/python
### Fixed
- Fix bug in GCI2 score for ELEmbeddings
- Fix bottleneck in ELBE example for PPI.
- Fix bugs in BoxSquaredEL model.
- Fix bug on OWL2VecStarProjector about projecting with literals

### Security

## [0.3.0]
### Added
- Implemented `CategoricalProjector` based on [https://arxiv.org/abs/2305.07163](https://arxiv.org/abs/2305.07163). ([#59][i59])

### Removed
- Removed redundant class `based_models.EmbeddingModel`


## [0.2.1]
### Fixed
Fixed issue related to importing graph-based models due to missing `__init__.py` files. ([#60][i60])

## [0.2.0]

### Added
- [BoxSquaredEL](https://arxiv.org/abs/2301.11118) module added to `mowl.nn`
- Implemented `model.from_pretrained` method. Related to issue [#43][i43]
- Implemented `model.add_axioms` method. Related to issue [#43][i43]
- Added models `RandomWalkPlusW2VModel`, `GraphPlusPyKEENModel`, `SyntacticPlusW2VModel`, 
- Updated dependencies: JPype-1.3.0 --> JPype-1.4.1, pykeen-1.9.0 --> pykeen-1.10.1
- Support for Python 3.8, 3.9, 3.10, 3.11. ([#42][i42])
- 
### Changed
- Bug fixed in corpus generation methods. Issue [#36][i36].
- Updated dependencies to work with Python 3.8 and 3.9. Issue [#42][i42]

## [0.1.1]

### Added
- `Family` dataset: a small ontology containing 12 axioms.
- Unit tests up to 88% of coverage
- `DL2VecProjector` includes assertion axioms with named individuals
- `FastTensorDataLoader` is part of mOWL in the module `mowl.utils.data`
- `DeepGOZero` implementation in mOWL
- Module `mowl.owlapi` with shortcuts to the OWLAPI
- Extended `Dataset` class.
### Security
- Added Patch for bug CVE-2007-4559. Based on pull request [#32][i32]

## [0.1.0]

### Added
- Modules `mowl.ontology.extend` and `mowl.ontology.create` created including `insert_annotations` and `create_from_triples` methods, respectively.
- Package `deprecated` as dependency.
- ELEmbeddingModel abstract class that contains basic functionality for models embedding the EL language.
- Implementation of [ELBoxEmbeddings](https://arxiv.org/abs/2202.14018)
- ELDataset class to work with EL models.
- Module `mowl.nn` where `torch.nn`-based modules will reside. The first module existing there is the abstract module for EL models.
- Module `mowl.models` where implementation of ELEmbeddings and ELBoxEmbeddings reside.
- PyKEEN as dependency
- `GDAHumanELDataset` and `GDAMouseELDataset`, which are reduced versions of `GDAHumanDataset` and `GDAMouseDataset`, respectively. The new datasets can be normalized into the `EL` language using the [jcel](https://julianmendez.github.io/jcel/) library.
- Started implementation of unit tests for `datasets` and `walking` modules.
### Changed
- All builtin datasets can be imported from `mowl.datasets.builtin` module.
- Updated implementation of [ELEmbeddings](https://www.ijcai.org/Proceedings/2019/845)
- Changed method `mowl.datasets.PathDataset.get_evaluation_classes` to property `mowl.datasets.PathDataset.evaluation_classes`
- #25 Walking methods now accept an optional parameter for a list of nodes to filter the random walks.

### Deprecated
- Modules `mowl.datasets.ppi_yeast` and `mowl.datasets.gda`.
- File `mowl.datasets.build_ontology`.
- Class `mowl.embeddings.elembeddings.ELEmbeddings`. Future versions will point to `mowl.models.ELEmbeddings`
- Method `mowl.corpus.base.extract_annotation_corpus`. Future versions will split this method into two: `mowl.corpus.base.extract_annotation_corpus` and `mowl.corpus.base.extract_and_save_annotation_corpus`.
- Method `mowl.datasets.PathDataset.get_evaluation_classes`.

## [0.0.30] - 2022-07-03
### Added
- Added `matplotlib` and `tqdm` as dependencies of the package.

### Fixed
- Walking methods accept optional `outfile` parameter and corpus extraction methods do not append by default.
- Documentation updated and fixed some typos.

[unreleased]: https://github.com/bio-ontology-research-group/mowl/compare/v2.1.0...HEAD
[2.1.0]: https://github.com/bio-ontology-research-group/mowl/compare/v2.0.0...v2.1.0
[2.0.0]: https://github.com/bio-ontology-research-group/mowl/compare/v1.0.3...v2.0.0
[1.0.3]: https://github.com/bio-ontology-research-group/mowl/releases/tag/v1.0.3
[1.0.2]: https://github.com/bio-ontology-research-group/mowl/releases/tag/v1.0.2
[1.0.1]: https://github.com/bio-ontology-research-group/mowl/releases/tag/v1.0.1
[1.0.0]: https://github.com/bio-ontology-research-group/mowl/releases/tag/v1.0.0
[0.3.0]: https://github.com/bio-ontology-research-group/mowl/releases/tag/v0.3.0
[0.2.1]: https://github.com/bio-ontology-research-group/mowl/releases/tag/v0.2.1
[0.2.0]: https://github.com/bio-ontology-research-group/mowl/releases/tag/v0.2.0
[0.1.1]: https://github.com/bio-ontology-research-group/mowl/releases/tag/v0.1.1
[0.1.0]: https://github.com/bio-ontology-research-group/mowl/releases/tag/v0.1.0
[0.0.30]: https://github.com/bio-ontology-research-group/mowl/releases/tag/v0.0.30


[i32]: https://github.com/bio-ontology-research-group/mowl/issues/32
[i36]: https://github.com/bio-ontology-research-group/mowl/issues/36
[i42]: https://github.com/bio-ontology-research-group/mowl/issues/42
[i43]: https://github.com/bio-ontology-research-group/mowl/issues/43
[i59]: https://github.com/bio-ontology-research-group/mowl/issues/59
[i60]: https://github.com/bio-ontology-research-group/mowl/issues/60
[i70]: https://github.com/bio-ontology-research-group/mowl/issues/70
[i71]: https://github.com/bio-ontology-research-group/mowl/issues/71
[i142]: https://github.com/bio-ontology-research-group/mowl/issues/142
[i146]: https://github.com/bio-ontology-research-group/mowl/issues/146
[i149]: https://github.com/bio-ontology-research-group/mowl/issues/149
[i150]: https://github.com/bio-ontology-research-group/mowl/issues/150
[i155]: https://github.com/bio-ontology-research-group/mowl/issues/155
[i156]: https://github.com/bio-ontology-research-group/mowl/issues/156
[i157]: https://github.com/bio-ontology-research-group/mowl/issues/157
[i97]: https://github.com/bio-ontology-research-group/mowl/issues/97

