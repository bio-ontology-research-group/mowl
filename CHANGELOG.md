# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- `mowl.geometric` package for constructive exact geometric models, model checking, and box-representability diagnostics of normalised $\mathcal{EL}^{++}$ ontologies:
  - `NormalisedOntology`, `normalise`, `to_owl`, and normal-form dataclasses `NF1`, `NF2`, `NF3`, `NF4`, `NF6`, `NF7` (`TOP`, `BOTTOM`, `Axiom`): extract normalised $\mathcal{EL}^{++}$ TBox and RBox axioms (splitting role chains of length $> 2$) and record dropped axioms (Baader, Brandt & Lutz, 2005; Kazakov, Krötzsch & Simančík, 2014).
  - `Completion` and `CompletionResult`: worklist-driven ELK saturation rules R1–R7 deriving all entailed concept-name subsumptions, role edges, and $\bot$ propagations (Kazakov, Krötzsch & Simančík, 2014).
  - `CanonicalModel`: finite canonical model $\mathcal{I} \models \mathcal{O}$ with exact concept-name subsumption ($A^{\mathcal{I}} \subseteq B^{\mathcal{I}} \iff \mathcal{O} \models A \sqsubseteq B$) and unsatisfiability ($A^{\mathcal{I}} = \emptyset \iff \mathcal{O} \models A \sqsubseteq \bot$), raising `InconsistentOntology` iff $\mathcal{O} \models \top \sqsubseteq \bot$ (Kazakov, Krötzsch & Simančík, 2014).
  - `IntervalUnion`, `RectUnion`, `Box`, `Face`, `ThickFace`, `PairHull`, `EMPTY_1D`, and `simplex_points`: pure-NumPy region and role-region representations with exact set operations and H-representations (Gutiérrez-Basulto & Schockaert, 2018; Lacerda, Ozaki & Guimarães, 2024).
  - `GeometricInterpretation`, `DictInterpretation`, and `ModelChecker`: exact decision procedures for axiom satisfaction (`NF1`–`NF4`, `NF6`, `NF7`, and $\bot$ forms), non-degeneracy, and strong faithfulness on concept names across interval, polyhedral, and propositional box interpretations (Gutiérrez-Basulto & Schockaert, 2018; Lacerda, Ozaki & Guimarães, 2024; Bourgaux et al., 2024).
  - `ConvexPolyhedralModel`: constructive convex polyhedral model in $\mathbb{R}^{|\Delta^{\mathcal{I}}| - 1}$ guaranteed to be an exact, non-degenerate, strongly faithful model of any consistent normalised $\mathcal{EL}^{++}$ ontology, verified by `ModelChecker` on construction (Gutiérrez-Basulto & Schockaert, 2018; Lacerda, Ozaki & Guimarães, 2024).
  - `IntervalUnionModel`: constructive 1-D interval-union model guaranteed to be an exact, non-degenerate, strongly faithful model of any consistent normalised $\mathcal{EL}^{++}$ ontology, verified by `ModelChecker` on construction (Lacerda, Ozaki & Guimarães, 2024; Federico, Leemhuis & Kutz, 2026).
  - `helly_check`, `HellyReport`, and `HellyCertificate`: Helly-satisfiability diagnostic enumerating minimal jointly unsatisfiable concept sets and emitting sound non-box-representability certificates, exact for the strong reading on the propositional fragment when `complete=True` (Eckhoff, 1988; Leemhuis & Kutz, 2025; Leemhuis & Kutz, 2026).
  - `BoxModel` and `BoxModelBuilder`: MILP-based search (`scipy.optimize.milp`) for axis-parallel box models of propositional $\mathcal{EL}_\bot$ ontologies after a Helly pre-check, verifying any found model with `ModelChecker`, raising `NotBoxRepresentable` on a Helly certificate, and raising `BoxModelNotFound` when inconclusive (Leemhuis & Kutz, 2025).
  - `class_points`, `to_bounding_boxes`, and `boxes_as_tensors` (`mowl.geometric.export`): export polyhedral vertex sets, documented lossy bounding boxes, and `class_to_id`-ordered box arrays (`(lo, hi)` or `(center, offset)`) for seeding `ELBoxEmbeddings` and `BoxSquaredEL` (Gutiérrez-Basulto & Schockaert, 2018; Leemhuis & Kutz, 2025).
  - `GeometricError`, `InconsistentOntology`, `UnsupportedAxiom`, `UnsupportedRegion`, `NotBoxRepresentable`, `BoxModelNotFound`, `InternalConsistencyError`, and `MissingDependency`: domain exception hierarchy distinguishing proved non-representability from inconclusive MILP search.
### Changed 
### Deprecated
### Removed
### Fixed
### Security

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

[unreleased]: https://github.com/bio-ontology-research-group/mowl/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/bio-ontology-research-group/mowl/releases/tag/v0.2.1
[0.2.0]: https://github.com/bio-ontology-research-group/mowl/releases/tag/v0.2.0
[0.1.1]: https://github.com/bio-ontology-research-group/mowl/releases/tag/v0.1.1
[0.1.0]: https://github.com/bio-ontology-research-group/mowl/releases/tag/v0.1.0
[0.0.30]: https://github.com/bio-ontology-research-group/mowl/releases/tag/v0.0.30


[i32]: https://github.com/bio-ontology-research-group/mowl/issues/32
[i36]: https://github.com/bio-ontology-research-group/mowl/issues/36
[i42]: https://github.com/bio-ontology-research-group/mowl/issues/42
[i43]: https://github.com/bio-ontology-research-group/mowl/issues/43
[i60]: https://github.com/bio-ontology-research-group/mowl/issues/60
