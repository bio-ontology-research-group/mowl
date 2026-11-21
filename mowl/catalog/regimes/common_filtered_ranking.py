"""Regime cards: the ranking of the joint mOWL benchmarks and the scoring head for
vector-only methods."""

from mowl.catalog.card import Reference, RegimeCard, RegimeKind
from mowl.catalog.regimes._regime_refs import CHEN2020B

BORDES2013 = Reference(
    "bordes2013",
    "Bordes, A., Usunier, N., Garcia-Duran, A., Weston, J., Yakhnenko, O. (2013). "
    "Translating Embeddings for Modeling Multi-relational Data. In Advances in Neural "
    "Information Processing Systems 26.",
    "https://papers.nips.cc/paper/2013/hash/1cecc7a77928ca8133fa24680a88d2f9-Abstract.html")

REGIMES = (
    RegimeCard(
        key="common-filtered-ranking",
        assumption="closed world over stated axioms",
        task="subsumption-prediction",
        name="Filtered ranking of the joint mOWL benchmarks",
        kind=RegimeKind.EVALUATION,
        summary=(
            "For every test axiom C ⊑ D or C ⊑ ∃r.D, ranks D among all classes of the "
            "ontology (C removed), after removing the other classes known to be correct "
            "for the query (training, validation and test axioms). Tied candidates share "
            "the average rank. Filtered MRR, Hits@1/10/100, mean and median rank, AUC. "
            "The score is whatever the model declares, so every model family is ranked "
            "the same way."),
        components=("test set", "candidate set", "filtering", "ties", "metrics", "scores"),
        applies_to=("Any model that declares axiom_scoring "
                    "(mowl.evaluation.model_scoring): normal-form EL models, CatE, ALC "
                    "models, E2R, binary code boxes, knowledge graph embeddings of a "
                    "projection, and vector-only models through the pair classifier head"),
        references=(BORDES2013,),
        standalone=False,
        implementation="mowl.evaluation.filtered_ranking.common_ranking",
        limitations=(
            "The score differs between model families (a geometric loss, an energy, a "
            "classifier logit); only ranks are compared, never score values.",
            "A held-out test axiom may be entailed by the training ontology; the regime "
            "does not separate these from axioms that are not.",
        ),
        notes=(
            "The filtered setting is that of Bordes et al. (2013, Section 4.2). Average "
            "ranks for ties are used instead of the position-based ties of several "
            "reference codes, which reward a constant score. One regime serves both "
            "targets: with a test set of held-out axioms it measures inductive "
            "prediction, with a test set of entailed axioms deductive inference; the "
            "benchmark states which."),
    ),
    RegimeCard(
        key="pair-classifier-head",
        assumption="closed world over stated axioms",
        name="Pair classifier on class vectors",
        kind=RegimeKind.EVALUATION,
        summary=(
            "A small network on the vectors of two classes, trained on the stated axioms "
            "of the training ontology against randomly corrupted ones; its logit scores "
            "candidate axioms for a model that gives vectors only."),
        components=("scores",),
        applies_to=("Models with class_embeddings and no axiom score of their own: random "
                    "walk and corpus models with Word2Vec; usable with any model"),
        references=(CHEN2020B,),
        standalone=False,
        introduced_with=("owl2vec_star",),
        used_by=("dl2vec", "onto2vec", "opa2vec", "taxonomy_walks"),
        implementation="mowl.evaluation.model_scoring.PairClassifierHead",
        limitations=(
            "The result measures the vectors together with the head: its size, negatives "
            "and training length are part of the evaluation.",
        ),
        notes=(
            "OWL2Vec* (Chen et al.) evaluates subsumption prediction by training a "
            "classifier on the concatenated vectors of the two classes. The features here "
            "are [x, y, x*y, |x-y|] on unit-length vectors; one network per axiom form "
            "and role; trained on the training ontology only."),
    ),
)
