# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- Added `.jar` files to enable pip install from GitHub
- Tested with Python 3.12
### Changed 
- Removed `.mean()` for GCI losses in BoxSquaredELModule
- Property `evaluation_classes` in `mowl.datasets.builtin.PPIYeastDataset` returns a pair of `OWLClasses` objects instead of a single `OWLClasses` object.
### Deprecated
- `mowl.nn.ELBoxModule` changed name to `mowl.nn.ELBEModule`
### Removed
### Fixed
- Fix bug in GCI2 score for ELEmbeddings
- Fix bottleneck in ELBE example for PPI.
- Fix bugs in BoxSquaredEL model.

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

[unreleased]: https://github.com/bio-ontology-research-group/mowl/compare/v0.3.0...HEAD
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

