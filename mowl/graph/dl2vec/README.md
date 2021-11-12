# Dl2Vec: [Predictingl candidate genes from phenotypes, functions and anatomical site of expression](https://academic.oup.com/bioinformatics/article/37/6/853/5922810)

## Original implementation: <https://github.com/bio-ontology-research-group/DL2Vec>


## Bugs fixed from the original implementation:

* Not including inferences of the type: $$A \equiv B \sqcap C = \vdash A \equiv B $$
* Not including inferences of the type: $A \equiv B \sqcap \exists R. C \vdash A \sqsubseteq \exists R. C $
* Not including `disjoint_with` axioms.
* Not including axioms with GO classes as relations. In case of `GO`: `GO:0007303`, `GO:0005248`.
