# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Modules `mowl.ontology.extend` and `mowl.ontology.create` created including `insert_annotations` and `create_from_triples` methods, respectively.
- Package `deprecated` as dependency.
### Changed
- All builtin datasets can be imported from `mowl.datasets.builtin` module.
### Deprecated
- Modules `mowl.datasets.ppi_yeast` and `mowl.datasets.gda`.
- File `mowl.datasets.build_ontology`.
### Removed
### Fixed
### Security

## [0.0.30] - 2022-07-03
### Added
- Added `matplotlib` and `tqdm` as dependencies of the package.

### Fixed
- Walking methods accept optional `outfile` parameter and corpus extraction methods do not append by default.
- Documentation updated and fixed some typos.

[unreleased]: https://github.com/bio-ontology-research-group/mowl/compare/v0.0.30...HEAD
[0.0.30]: https://github.com/bio-ontology-research-group/mowl/releases/tag/v0.0.30
