"""Region-based knowledge graph and query embeddings (BoxE, ExpressivE, RESHUFFLE,
Concept2Box, Query2Box, BetaE, ConE and AConE).

These methods are defined for knowledge graphs (triples, rule patterns, logical queries
over a KG), not for description logic ontologies. Each card states how the method is
connected to ontologies in mOWL and which axiom forms it supports.
"""

from mowl.catalog.card import (Family, Guarantee, GuaranteeKind as K, MethodCard, Origin,
                               Reference, Status)
from mowl.catalog.entries import _refs as R
from mowl.logic import Construct, Expressivity

_QUERY_NOTES = (
    "ADAPTATION of the data and the task, reimplementation of the geometry: the paper is "
    "about queries over a knowledge graph. mOWL reads the ontology as a graph (entities: "
    "named classes and individuals; edges: A ⊑ B and a:B as 'isa', A ⊑ ∃r.B and r(a,b) as "
    "r, each with its inverse; isa closed transitively by default; role edges are NOT "
    "propagated down the hierarchy) and reads a class expression as a query on it through "
    "the shared layer mowl.nn.query.expression (named class = everything below it, ∃r.C = "
    "projection along r backwards, ⊓ = intersection, ⊔ = union by disjunctive normal "
    "form, ¬ = negation, ∀r.C = ¬∃r.¬C). Answers are entities of the graph: closed world "
    "over the entities, not entailment. Read for training: named left-hand side with named "
    "classes and ∃r.B conjuncts on the right, equivalences of those, class and role "
    "assertions; the rest is skipped (counted in model.skipped_axioms), so the declared "
    "expressivity is that of the part read; queries themselves may be ALC class "
    "expressions where the geometry has the operators. Own training loop (query sampling "
    "and the negative sampling loss of the reference codes) shared by the four query "
    "methods in mowl.models.query; it is not a TrainingRegime (those are tied to EL normal "
    "forms). Axiom scores for the joint rankings: gci0 and class_assertion are 'the entity "
    "is an answer of the class', gci2 'the entity is an answer of ∃r.D'; gci1 and gci3 "
    "are not scored. Evaluation regime 'query-answering' through score_query (the "
    "negation structures are not sampled by mowl.evaluation.sample_queries). Default "
    "sizes (dimension, hidden width, layers) are small; the papers' are larger. ")

_QUERY_FIELDS = dict(
    status=Status.LEARNABLE, origin=Origin.ADAPTATION,
    expressivity="EL with ABox",
    tasks=("subsumption", "instance-checking", "query-answering"),
    axioms=("subclass_of", "equivalent_classes", "class_assertion",
            "object_property_assertion"))

_KG_ONLY = ("Defined for knowledge graphs, not for DL ontologies: there is no "
            "interpretation of TBox axioms (GCIs over complex concepts).")

BOXE = MethodCard(
    key="boxe",
    assumption="closed world over stated axioms",
    training=("logsigmoid-negative-sampling", "either-side-corruption",),
    evaluation=("kg-filtered-link-prediction", "common-filtered-ranking",),
    name="BoxE",
    family=Family.GEOMETRIC,
    region="box",
    summary=(
        "Entities are points with translational bumps; each relation has one box per "
        "argument position (head, tail). A triple holds when each entity, bumped by the "
        "other, lies in the corresponding box. Unary relations (types) are boxes."),
    logic="Knowledge graphs with rule patterns (inverse, symmetry, hierarchy, intersection, "
          "mutual exclusion, anti-symmetry); higher-arity facts",
    concepts="box (types as unary relations)",
    roles="one box per argument position",
    individuals="point with bump",
    status=Status.LEARNABLE,
    origin=Origin.ADAPTATION,
    expressivity="EL with ABox",
    tasks=("subsumption",),
    implementation="mowl.models.boxe.model.BoxE",
    guarantees=(
        Guarantee(
            K.EXPRESSIVITY,
            "Fully expressive with dimension d equal to the number of entities to the power "
            "n − 1 times the number of relations (n the maximal arity).",
            "boxe2020, Theorem 5.1"),
        Guarantee(
            K.EXPRESSIVITY,
            "Can capture any finite consistent set of inverse, symmetry, hierarchy, "
            "intersection, mutual exclusion and anti-symmetry rules; the positive fragment "
            "can be injected.",
            "boxe2020, Theorems 5.3 and 5.4"),
        Guarantee(
            K.SOUNDNESS,
            "Sound under the embedding semantics; fully ABox- and TBox-expressive for "
            "patterns without composition.",
            "bourgaux2024, Table 4"),
        Guarantee(
            K.LIMITATION,
            "Not complete for languages with role disjointness: {r ⊑ ¬s} with "
            "{r(a,b), s(a,c), r(d,c), s(d,b)} is satisfiable but has no BoxE-model.",
            "bourgaux2024, Example 4"),
    ),
    intersection_closed=True,
    helly="2 (axis-parallel boxes; eckhoff1988)",
    related=("graph_kge",),
    notes=(
        "Not reimplemented: this is PyKEEN's BoxE (pykeen.models.BoxE) trained through "
        "GraphPlusPyKEENModel with PyKEEN's negative sampling and self-adversarial loss. "
        "Ontology link: the triples are the edges of a mowl.projection projector (default OWL2VecStarProjector: C subclassof D, C r D for C ⊑ ∃r.D, a type C for C(a), r(a, b)); axioms the projector does not read are ignored. Scores are available for C ⊑ D (gci0, the subclassof edge) and C ⊑ ∃r.D (gci2) only; class and role assertions are trained on but not scored. The paper's unary relations (types as boxes) become binary "
        "type edges and classes are entities, so every guarantee above refers to the "
        "projected triples, not to the description logic semantics. Higher-arity facts are "
        "not used."),
    references=(
        Reference("boxe2020",
                  "Abboud, R., Ceylan, İ. İ., Lukasiewicz, T., Salvatori, T. (2020). BoxE: A "
                  "Box Embedding Model for Knowledge Base Completion. In Proceedings of "
                  "NeurIPS 2020.",
                  "https://arxiv.org/abs/2007.06267"),
        R.BOURGAUX2024),
)

EXPRESSIVE = MethodCard(
    key="expressive",
    assumption="closed world over stated axioms",
    training=("logsigmoid-negative-sampling", "either-side-corruption",),
    evaluation=("kg-filtered-link-prediction", "common-filtered-ranking",),
    name="ExpressivE",
    family=Family.GEOMETRIC,
    region="hyper-parallelogram (in the virtual triple space)",
    summary=(
        "Entity pairs are points in a 2d-dimensional virtual triple space and each relation "
        "is a hyper-parallelogram there, so rule patterns, including composition, become "
        "spatial relations between regions."),
    logic="Knowledge graphs with rule patterns (including composition)",
    concepts="not represented",
    roles="hyper-parallelogram in R^(2d)",
    individuals="point",
    status=Status.LEARNABLE,
    origin=Origin.ADAPTATION,
    expressivity="EL with ABox",
    tasks=("subsumption",),
    implementation="mowl.models.expressive.model.ExpressivE",
    guarantees=(
        Guarantee(
            K.EXPRESSIVITY,
            "Can capture any graph over the given entities and relations (full "
            "expressiveness) for sufficient dimension.",
            "pavlovic2023, Theorem 5.1"),
        Guarantee(
            K.EXPRESSIVITY,
            "Captures symmetry, anti-symmetry, inversion, hierarchy, intersection, mutual "
            "exclusion and compositional definition patterns.",
            "pavlovic2023, Theorems 5.2-5.4",
            verified_in_mowl="tests/nn/test_expressive_module.py (symmetry, hierarchy, "
                             "mutual exclusion on hand-built parallelograms)"),
        Guarantee(
            K.LIMITATION,
            "Sound but not complete for languages with mutual exclusion, hence not fully "
            "KB-expressive: the convexity argument of Example 2 applies to its "
            "parallelograms.",
            "bourgaux2024, Example 11 and Table 4"),
    ),
    references=(
        Reference("pavlovic2023",
                  "Pavlovic, A., Sallinger, E. (2023). ExpressivE: A Spatio-Functional "
                  "Embedding For Knowledge Graph Completion. In Proceedings of ICLR 2023.",
                  "https://arxiv.org/abs/2206.04192"),
        R.BOURGAUX2024),
    code_url="https://github.com/aleksvap/expressive",
    related=("graph_kge", "boxe"),
    notes=(
        "Reimplemented as a PyKEEN ERModel (mowl.nn.graph.expressive) after the authors' "
        "code (tanh maps, minimal denominator Dmin, distance of Equation 3, self-adversarial "
        "loss with margin 3 and temperature 2) and trained through GraphPlusPyKEENModel; "
        "only the base model, not the ablations. Ontology link: the triples are the edges of a mowl.projection projector (default OWL2VecStarProjector: C subclassof D, C r D for C ⊑ ∃r.D, a type C for C(a), r(a, b)); axioms the projector does not read are ignored. Scores are available for C ⊑ D (gci0, the subclassof edge) and C ⊑ ∃r.D (gci2) only; class and role assertions are trained on but not scored. Composition "
        "patterns hold for the projected triples; role chains of the ontology are not "
        "read."),
)

RESHUFFLE = MethodCard(
    key="reshuffle",
    assumption="closed world over stated axioms",
    training=("pairwise-margin-ranking", "either-side-corruption",),
    evaluation=("kg-filtered-link-prediction", "inductive-sampled-ranking",
                "common-filtered-ranking",),
    name="RESHUFFLE",
    family=Family.GEOMETRIC,
    region="box after a coordinate reshuffle",
    summary=(
        "Region-based KG embedding in which entity embeddings are compared by ordering "
        "constraints after permuting coordinates, computed by a GNN that acts as a "
        "differentiable rule base. Captures bounded inference with closed path rules "
        "faithfully."),
    logic="Knowledge graphs with closed path rules",
    concepts="not represented",
    roles="region over reshuffled coordinates",
    individuals="vector (GNN output)",
    status=Status.LEARNABLE,
    origin=Origin.ADAPTATION,
    expressivity="EL with ABox",
    tasks=("subsumption",),
    implementation="mowl.models.reshuffle.model.Reshuffle",
    guarantees=(
        Guarantee(
            K.SOUNDNESS,
            "If the entity embedding captures every triple of an eq-complete KG G, the "
            "model captures every triple entailed by the closed path rules P and G.",
            "pavlovic2025, Proposition 1"),
        Guarantee(
            K.WEAK_FAITHFULNESS,
            "For the GNN-learned entity representations, triples not entailed by P ∪ G "
            "are not captured once the number of GNN layers is large enough.",
            "pavlovic2025, Proposition 2",
            condition="enough GNN layers"),
    ),
    references=(
        Reference("pavlovic2025",
                  "Pavlovic, A., Sallinger, E., Schockaert, S. (2025). Faithful "
                  "Differentiable Reasoning with Reshuffled Region-based Embeddings. In "
                  "Proceedings of KR 2025, 489-499.",
                  "https://doi.org/10.24963/kr.2025/48"),),
    code_url="https://github.com/AleksVap/RESHUFFLE",
    related=("graph_kge", "boxe", "expressive"),
    notes=(
        "Implemented from the authors' code and checked against the paper (arXiv:2406.09529, "
        "Sections 4-6 and 7): random binary "
        "entity features, soft reshuffling relation matrices (column softmax, last row "
        "dropped), max-aggregation message passing shared over the layers, score "
        "-||ReLU(x_h R_r - x_t)||_2, margin ranking loss with 100 negatives. Differences: "
        "the random features are drawn once per model, not at every forward pass; training "
        "is on the projected ontology graph; by default inverse relations are separate "
        "parameters used for message passing only; the authors' code (and the recipes) also "
        "train the inverse triples as positives and predict heads through the inverse matrix "
        "(PyKEEN create_inverse_triples=True), which is Reshuffle(inverse_triples=True). "
        "Without it the head side of a link prediction is scored with the forward matrix, "
        "which favours entities with small feature matrices and ranks heads far worse. Inductive link prediction, "
        "the setting of the paper: Reshuffle.inductive_costs(graph, rows) scores triples "
        "over a graph with unseen entities, encoding it with fresh random features, and is "
        "what the evaluation regime inductive-sampled-ranking calls (benchmarks "
        "kg-inductive-*, with the published Hits@10 of Tables 2 and 3 and the settings of "
        "the authors' code as recipes). The authors stop training early on Hits@10 of the "
        "validation triples (every 10 epochs, patience 10, relative delta 0.01, best weights "
        "restored): Reshuffle(early_stopping={...}). Not reproduced exactly: the authors draw "
        "the test negatives without a seed and new random features at every batch, mOWL "
        "draws both once per evaluation seed. The "
        "propositions above are the paper's; tests/nn/test_reshuffle_module.py checks "
        "Propositions 1, 2 and 7 on a hand-built rule graph (Equation 7) for the "
        "matrices and the message passing of Equation 6, not for learned matrices. The "
        "paper's text initialises features with probability 0.5 of a 1, the code (the "
        "default here) 0.3085 (argument density); RESHUFFLE_nL and RESHUFFLE2 of the paper "
        "are aggregation='max_self_noLoop' and row_function='square'. Ontology link: the triples are the edges of a mowl.projection projector (default OWL2VecStarProjector: C subclassof D, C r D for C ⊑ ∃r.D, a type C for C(a), r(a, b)); axioms the projector does not read are ignored. Scores are available for C ⊑ D (gci0, the subclassof edge) and C ⊑ ∃r.D (gci2) only; class and role assertions are trained on but not scored. The GNN encodes the "
        "whole graph at every step, which suits small and medium graphs."),
)

QUERY2BOX = MethodCard(
    key="query2box",
    assumption="closed world over stated axioms",
    training=("logsigmoid-negative-sampling",),
    evaluation=("query-answering", "common-filtered-ranking",),
    name="Query2Box",
    family=Family.GEOMETRIC,
    region="box",
    summary=(
        "Embeds conjunctive logical queries over a KG as boxes and entities as points; "
        "projection and intersection operators map boxes to boxes, and an answer is an "
        "entity inside the query box. Disjunctions are handled by rewriting to DNF."),
    logic="Existential positive first-order queries over knowledge graphs",
    concepts="query box",
    roles="box translation (projection operator)",
    individuals="point",
    implementation="mowl.models.query2box.model.Query2Box",
    module="mowl.nn.query.geometry.Query2BoxGeometry",
    notes=_QUERY_NOTES + (
        "Query2Box has no negation: a class expression with ¬ or ∀ raises UnsupportedOperator. "
        "Geometry as in the reference code of the BetaE/Query2Box authors (KGReasoning, "
        "geo='box'): attention-averaged centers, min-offset with sigmoid gate, offsets activated "
        "by relu (option softplus/none). Not recorded as a guarantee because mOWL has no "
        "test for it: Theorem 1 of ren2020q2b (with M conjunctive queries with disjoint "
        "answers, answering every EPFO query by a distance threshold needs VC dimension "
        "at least M, so unions are answered by DNF with the best branch, which mOWL does). "
        "Paper and code agree on the distance (checked against the LaTeX source): "
        "d_out + alpha * d_in with alpha = center_weight, d_in the distance of the centre "
        "to the entity clamped into the box. "),
    **_QUERY_FIELDS,
    intersection_closed=True,
    helly="2 (axis-parallel boxes; eckhoff1988)",
    references=(
        Reference("ren2020q2b",
                  "Ren, H., Hu, W., Leskovec, J. (2020). Query2box: Reasoning over Knowledge "
                  "Graphs in Vector Space using Box Embeddings. In Proceedings of ICLR 2020.",
                  "https://arxiv.org/abs/2002.05969"),),
)

BETAE = MethodCard(
    key="betae",
    assumption="closed world over stated axioms",
    training=("logsigmoid-negative-sampling",),
    evaluation=("query-answering", "common-filtered-ranking",),
    name="BetaE",
    family=Family.ALGEBRAIC,
    region="product of Beta distributions",
    summary=(
        "Embeds entities and first-order queries (with negation) over a KG as products of "
        "Beta distributions; intersection and negation are probabilistic operators on "
        "the distribution parameters."),
    logic="First-order queries with conjunction, disjunction and negation over KGs",
    concepts="Beta distribution embedding of a query",
    roles="neural projection operator",
    individuals="Beta distribution embedding",
    implementation="mowl.models.betae.model.BetaE",
    module="mowl.nn.query.geometry.BetaEGeometry",
    notes=_QUERY_NOTES + (
        "Geometry as in the authors' code (KGReasoning, geo='beta'): parameters clamped to [0.05,"
        " 1e9], KL(entity || query), negation 1/parameter, DNF for unions (not the De Morgan "
        "variant). Default MLP is 1 hidden layer of 128 units (code: 2 layers of 1600). "),
    **_QUERY_FIELDS,
    guarantees=(
        Guarantee(
            K.EXPRESSIVITY,
            "Negation is an involution, N(N(S)) = S, and intersection is idempotent, "
            "I({S, ..., S}) = S.",
            "ren2020betae, Proposition 1",
            verified_in_mowl="tests/nn/test_query_geometry.py::"
                             "test_betae_negation_is_an_involution_and_intersection_idempotent"),
    ),
    references=(
        Reference("ren2020betae",
                  "Ren, H., Leskovec, J. (2020). Beta Embeddings for Multi-Hop Logical "
                  "Reasoning in Knowledge Graphs. In Proceedings of NeurIPS 2020.",
                  "https://arxiv.org/abs/2010.11465"),),
)

CONE_QE = MethodCard(
    key="cone_qe",
    assumption="closed world over stated axioms",
    training=("logsigmoid-negative-sampling",),
    evaluation=("query-answering", "common-filtered-ranking",),
    name="ConE (cone query embeddings)",
    family=Family.GEOMETRIC,
    region="cone (product of 2-D sector-cones)",
    summary=(
        "Embeds first-order queries over a KG as Cartesian products of two-dimensional "
        "sector-cones, which supports negation through the closure-complement of a cone."),
    logic="First-order queries with conjunction, disjunction and negation over KGs",
    concepts="product of 2-D sector-cones",
    roles="neural projection operator",
    individuals="cone (degenerate)",
    implementation="mowl.models.cone_qe.model.ConE",
    module="mowl.nn.query.geometry.ConEGeometry",
    notes=_QUERY_NOTES + (
        "Geometry as in the authors' code (QE-ConE): tanh-scaled axes, projection MLP, "
        "SemanticAverage and CardMin intersection, complement negation, distance with learnable "
        "modulus. Differences: the intersection axis uses atan2 instead of the code's clamp of "
        "|x| < 1e-3 and quadrant patching, and negation is not done in place. "
        "Learning rates above about 5e-3 made the loss collapse to log 2 on a "
        "14-entity test ontology (the code uses 1e-4); default MLP 1 layer "
        "of 128 (code: 2 of 1600). "),
    **_QUERY_FIELDS,
    guarantees=(
        Guarantee(
            K.CLOSURE,
            "Sector-cones are closed under closure-complement; their union and "
            "intersection are again cones.",
            "zhang2021cone, Section 3.1 (after Definition 3)",
            verified_in_mowl="tests/nn/test_query_geometry.py::"
                             "test_cone_negation_is_the_complement_and_an_involution"),
    ),
    references=(
        Reference("zhang2021cone",
                  "Zhang, Z., Wang, J., Chen, J., Ji, S., Wu, F. (2021). ConE: Cone "
                  "Embeddings for Multi-Hop Reasoning over Knowledge Graphs. In Proceedings "
                  "of NeurIPS 2021.",
                  "https://arxiv.org/abs/2110.13715"),),
)

ACONE = MethodCard(
    key="acone",
    assumption="closed world over stated axioms",
    training=("logsigmoid-negative-sampling",),
    evaluation=("query-answering", "common-filtered-ranking",),
    name="AConE",
    family=Family.GEOMETRIC,
    region="cone in complex space (multicone)",
    summary=(
        "A query embedding method that associates KG queries with SROI⁻ concepts: concepts "
        "are cones in complex vector space and relations are rotations, so the learned "
        "embedding can be read as SROI⁻ axioms."),
    logic="SROI⁻ concepts learned from a knowledge graph (no TBox input)",
    concepts="cone (multicone) in complex space",
    roles="rotation (with aperture scaling)",
    individuals="complex vector",
    implementation="mowl.models.acone.model.AConE",
    module="mowl.nn.query.geometry.AConEGeometry",
    notes=_QUERY_NOTES + (
        "Geometry as in the paper (Eq. 2, rotations; aperture-additive) and the authors' code "
        "(models.py: the projection is the complex product of axis and aperture with the "
        "relation's angles; the unused MLP of the code is not built). Intersection, negation and "
        "score are ConE's, as in the code. The aperture-multiplicative variant and the multicone "
        "semantics of the union are not implemented (unions: DNF with best branch, as in the "
        "code). The paper's claims that learned relations satisfy axioms (role inclusion, "
        "composition) are not checked by mOWL: only the algebra of the operators is tested; the "
        "paper's SROI⁻ axiom extraction from a learned model is not implemented. Observed on a "
        "14-entity test ontology: with rotation-only projection, paths of several isa steps "
        "(2p, 3p) were fitted worse than single steps in a short run (not investigated "
        "further). "),
    **_QUERY_FIELDS,
    guarantees=(
        Guarantee(
            K.EXPRESSIVITY,
            "∃R.(C ⊔ D) ≡ ∃R.C ⊔ ∃R.D holds for rotations with γ = 1 and δ ≥ 0 "
            "(Proposition 3), and R ⊑ S holds for rotations with the same θ if γ1 ≤ γ2 and "
            "δ1 ≤ δ2 (Proposition 5). The other statements (Propositions 2 and 4, Corollaries "
            "6-7: commutation, composition, transitivity) are not recorded: mOWL has no test "
            "for them.",
            "he2024acone, Propositions 3 and 5",
            condition="for the aperture-additive rotation, any parameters (Prop. 5: δ1 ≤ δ2)",
            verified_in_mowl="tests/nn/test_query_geometry.py::"
                             "test_acone_proposition_5_role_inclusion_and_"
                             "proposition_3_union_distribution"),
    ),
    references=(
        Reference("he2024acone",
                  "He, Y., Hernandez, D., Nayyeri, M., Xiong, B., Zhu, Y., Kharlamov, E., "
                  "Staab, S. (2024). Generating SROI⁻ Ontologies via Knowledge Graph Query "
                  "Embedding Learning. arXiv:2407.09212.",
                  "https://arxiv.org/abs/2407.09212"),),
    code_url="https://github.com/RoyaHe/AConE",
)

CONCEPT2BOX = MethodCard(
    key="concept2box",
    assumption="closed world over stated axioms",
    training=("pairwise-margin-ranking", "either-side-corruption",),
    evaluation=("kg-filtered-link-prediction", "common-filtered-ranking",),
    name="Concept2Box",
    family=Family.GEOMETRIC,
    region="box",
    summary=(
        "Jointly embeds a two-view KG: concepts of the ontology view are boxes and entities "
        "of the instance view are points, linked by a vector-to-box distance."),
    logic="Two-view knowledge graph (concept hierarchy plus instance triples)",
    concepts="box",
    roles="KGE relation embedding",
    individuals="point",
    status=Status.LEARNABLE,
    origin=Origin.ADAPTATION,
    implementation="mowl.models.concept2box.model.Concept2Box",
    module="mowl.nn.el.concept2box.module.Concept2BoxModule",
    expressivity=Expressivity({Construct.TOP, Construct.EXISTENTIAL},
                              label="EL without conjunction").with_abox(),
    tasks=("subsumption", "instance-checking", "role-assertion"),
    axioms=("gci0", "gci2", "class_assertion", "object_property_assertion"),
    intersection_closed=True,
    helly="2 (axis-parallel boxes; eckhoff1988)",
    related=("transbox", "boxel"),
    notes=(
        "Ontology link: the two views of the paper are the TBox and the ABox. Ontology "
        "view: C ⊑ D (gci0, the triple (D, subclass, C)) and C ⊑ ∃r.D (gci2, the triple "
        "(C, r, D)) with the conditional probability of Gumbel boxes; instance view: role "
        "assertions r(a, b) with a translational (TransE) score; cross-view links: class "
        "assertions C(a) with the vector-to-box distance. Not used: conjunctions on the "
        "left (gci1), existentials on the left (gci3), bottom, role axioms (a warning "
        "names them). Adaptations: the BERT encoder of the concept and relation texts is "
        "replaced by learned embeddings of the concept and the relation joined by the "
        "two-layer MLP, with no text; the pairwise hinge losses of the instance and cross "
        "views are split into a positive and a negative part so that the shared EL "
        "training loop applies (negatives by corruption); the cross-view box is the box of "
        "the concept in the context of a null relation. The expected volumes use the Gumbel "
        "box closed forms were checked against Dasgupta et al. (2020, Lemma 1, Proposition 1 "
        "and the softplus approximation) and Equations 1-3 of the Concept2Box paper, and "
        "tested against the Bessel expression and a Monte Carlo estimate. The paper "
        "alternates updates of the three losses; mOWL sums them (Equation 11). Margins "
        "(0.3, 0.15) and learning rate (0.005) are the paper's; dimensions are smaller. "
        "No reference code was consulted, and no published number is reproduced."),
    references=(
        Reference("huang2023",
                  "Huang, Z., Wang, D., Huang, B., Zhang, C., Shang, J., Liang, Y., Wang, Z., "
                  "Li, X., Faloutsos, C., Sun, Y., Wang, W. (2023). Concept2Box: Joint "
                  "Geometric Embeddings for Learning Two-View Knowledge Graphs. "
                  "arXiv:2307.01933.",
                  "https://arxiv.org/abs/2307.01933"),),
)

CARDS = (BOXE, EXPRESSIVE, RESHUFFLE, QUERY2BOX, BETAE, CONE_QE, ACONE, CONCEPT2BOX)
