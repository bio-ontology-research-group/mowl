# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[unreleased]: https://github.com/bio-ontology-research-group/mowl/compare/v0.0.30...HEAD
[0.0.30]: https://github.com/bio-ontology-research-group/mowl/releases/tag/v0.0.30
