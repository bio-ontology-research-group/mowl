"""Graph-based methods: an ontology projection followed by a graph embedding.

mOWL separates the projection (``mowl.projection``), the walk generator
(``mowl.walking``) and the embedding (Word2Vec or a PyKEEN model). Cards are given per
published method, with ``implementation`` pointing at the model class a user instantiates
and the projector named in ``notes``. The categorical projector (CatE) has its own card.
"""

from mowl.catalog.card import Family, MethodCard, Origin, Status
from mowl.catalog.entries import _refs as R

_RW = "mowl.models.graph_random_walk.random_walk_w2v_model.RandomWalkPlusW2VModel"
_KGE = "mowl.models.graph_kge.graph_pykeen_model.GraphPlusPyKEENModel"

_NO_SEMANTICS = (
    "Graph projections keep only part of the axioms and the embedding is trained on "
    "co-occurrence or triple plausibility, not on model-theoretic semantics; the survey "
    "of Chen et al. (2025, Section II, Table I) describes such methods as only partially "
    "faithful to the graph structure, with low interpretability.")

OWL2VEC_STAR = MethodCard(
    key="owl2vec_star",
    expressivity="OWL 2 DL",
    tasks=(),
    head_tasks=("subsumption",),
    factory="mowl.benchmarks.families.owl2vec_star",
    name="OWL2Vec*",
    family=Family.GRAPH,
    region="random walks + Word2Vec",
    summary=(
        "Projects the ontology to an RDF-style graph with the OWL2Vec* rules (subclass, "
        "existential and universal restrictions, domain/range, assertions), generates "
        "random walks and trains Word2Vec on them. The original method also adds a "
        "lexical corpus from annotations."),
    logic="OWL 2 (any OWL ontology; only the axiom patterns of the projection rules are used)",
    concepts="vector",
    roles="vector (relation nodes in walks)",
    individuals="vector",
    status=Status.LEARNABLE,
    origin=Origin.REIMPLEMENTATION,
    implementation=_RW,
    limitations=(_NO_SEMANTICS,),
    references=(R.CHEN2020B, R.MIKOLOV2013, R.CHEN2025),
    code_url="https://github.com/KRR-Oxford/OWL2Vec-Star",
    notes=(
        "Use RandomWalkPlusW2VModel with mowl.projection.OWL2VecStarProjector and a "
        "walker from mowl.walking (DeepWalk or Node2Vec). mOWL reproduces the structural "
        "projection; the lexical (annotation) document of the original is not built "
        "automatically (include_literals=True adds annotation triples to the graph, and "
        "mowl.corpus can extract annotation corpora). The projector can also be combined "
        "with GraphPlusPyKEENModel (OWL2Vec*-TransE style)."),
)

DL2VEC = MethodCard(
    key="dl2vec",
    expressivity="OWL 2 DL",
    tasks=(),
    head_tasks=("subsumption",),
    factory="mowl.benchmarks.families.dl2vec",
    name="DL2Vec",
    family=Family.GRAPH,
    region="random walks + Word2Vec",
    summary=(
        "Projects class axioms (subclass, equivalence, existential and universal "
        "restrictions, including nested conjunctions) to a graph with edges labelled by "
        "relations, generates random walks and trains Word2Vec on them. Designed for "
        "phenotype-based gene prioritisation."),
    logic="OWL 2 class axioms (projection rules of DL2Vec)",
    concepts="vector",
    roles="vector (relation nodes in walks)",
    individuals="vector",
    status=Status.LEARNABLE,
    origin=Origin.REIMPLEMENTATION,
    implementation=_RW,
    limitations=(_NO_SEMANTICS,),
    references=(R.CHEN2020, R.MIKOLOV2013),
    notes=(
        "Use RandomWalkPlusW2VModel with mowl.projection.DL2VecProjector and a walker "
        "from mowl.walking. The projector can equally be combined with "
        "GraphPlusPyKEENModel."),
)

TAXONOMY_WALKS = MethodCard(
    key="taxonomy_walks",
    expressivity="taxonomy",
    tasks=(),
    head_tasks=("subsumption",),
    factory="mowl.benchmarks.families.taxonomy_walks",
    name="Taxonomy projection + random walks",
    family=Family.GRAPH,
    region="random walks + Word2Vec",
    summary=(
        "Projects only subclass axioms A ⊑ B (TaxonomyProjector), optionally also "
        "A ⊑ ∃R.B as an R-edge (TaxonomyWithRelationsProjector), then embeds the graph "
        "with DeepWalk or node2vec walks and Word2Vec. A simple baseline."),
    logic="Subclass hierarchy (optionally with A ⊑ ∃R.B)",
    concepts="vector",
    roles="vector (relation nodes in walks)",
    status=Status.LEARNABLE,
    origin=Origin.ADAPTATION,
    implementation=_RW,
    limitations=(_NO_SEMANTICS,),
    references=(R.PEROZZI2014, R.GROVER2016, R.MIKOLOV2013),
    notes=(
        "Not a single published method: the combination of mOWL's taxonomy projectors "
        "with DeepWalk/node2vec walks. Use RandomWalkPlusW2VModel with "
        "mowl.projection.TaxonomyProjector or TaxonomyWithRelationsProjector."),
)

GRAPH_KGE = MethodCard(
    key="graph_kge",
    # What the widest projectors read; a narrower projector uses less of the ontology.
    expressivity="OWL 2 DL",
    # The score of the edge an axiom is projected to: C -subclassof-> D for C ⊑ D.
    tasks=("subsumption",),
    factory="mowl.benchmarks.families.graph_kge",
    name="Projection + knowledge graph embedding",
    family=Family.GRAPH,
    region="knowledge graph embedding (PyKEEN)",
    summary=(
        "Projects the ontology with any mOWL projector and trains a PyKEEN knowledge graph "
        "embedding model (TransE, DistMult, ComplEx, BoxE, ...) on the resulting triples."),
    logic="Depends on the projector",
    concepts="entity embedding of the chosen KGE model",
    roles="relation embedding of the chosen KGE model",
    individuals="entity embedding of the chosen KGE model",
    status=Status.LEARNABLE,
    origin=Origin.ADAPTATION,
    implementation=_KGE,
    limitations=(
        _NO_SEMANTICS,
        "Any guarantees of the chosen KGE model (e.g. BoxE's rule-capturing theorems) "
        "refer to the projected triples, not to the ontology's DL semantics.",
    ),
    references=(R.ALI2021, R.CHEN2025),
    notes=(
        "Use GraphPlusPyKEENModel with any projector from mowl.projection and a PyKEEN "
        "ERModel. OWL2Vec*-TransE in Chen et al. (2025) is an instance."),
)

GDA_PROJECTION = MethodCard(
    key="gda_projection",
    name="GDA projection",
    family=Family.GRAPH,
    region="knowledge graph embedding (PyKEEN)",
    summary=(
        "Extends the OWL2Vec* projection with one rule: an existential restriction whose "
        "filler is not atomic is unfolded through intersections and nested existentials "
        "until a named class is reached, giving one edge labelled by the composed role "
        "path. Used for inductive gene-disease association prediction with KGE models."),
    logic="OWL 2 class axioms (OWL2Vec* rules plus nested existentials)",
    concepts="entity embedding of the chosen KGE model",
    roles="relation embedding (one per composed role path)",
    individuals="entity embedding of the chosen KGE model",
    status=Status.LEARNABLE,
    origin=Origin.REIMPLEMENTATION,
    implementation=_KGE,
    limitations=(
        _NO_SEMANTICS,
        "The composed relation name is not injective: different role paths whose local "
        "names concatenate to the same string give the same relation.",
    ),
    references=(R.ZHAPA2026, R.CHEN2020B),
    notes=(
        "Use GraphPlusPyKEENModel with mowl.projection.GDAProjector. The publication is "
        "in preparation."),
)

CARDS = (OWL2VEC_STAR, DL2VEC, TAXONOMY_WALKS, GRAPH_KGE, GDA_PROJECTION)
