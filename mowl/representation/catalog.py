"""The parts of every catalogued method (:data:`REPRESENTATIONS`, by method key).

Kept in one place so that the methods can be compared part by part; the catalogue cards
read it (:attr:`mowl.catalog.MethodCard.representation`). Each entry was written from the
implementation in mOWL, not from the paper alone: where mOWL adapts a method (for example
octagons as class regions), the entry describes the adaptation.
"""

from mowl.representation.vocabulary import (
    View as V, Space as S, ClassModel as C, RelationModel as R, IndividualModel as I,
    Measure as M, Construction as K, Parametrisation as P, Composable as Y, Representation,
    OperatorFamily as F, Projection as Pj, Corpus as Co, Walker as Wk, Encoder as En,
    Reading as Rd, Scorer as Sc)


def _el(classes, relations, individuals, measure, conjunction=K.EXACT, existential=K.EXACT,
        composable=Y.YES, space=S.EUCLIDEAN, view=V.NORMAL_FORMS, **kwargs):
    """A region method trained on normal-form axioms."""
    return Representation(view=view, space=space, classes=classes,
                          relations=relations, individuals=individuals, measure=measure,
                          conjunction=conjunction, existential=existential,
                          composable=composable, **kwargs)


def _hierarchy(classes, measure, space=S.EUCLIDEAN, view=V.GRAPH, composable=Y.YES, **kwargs):
    """A method that embeds a hierarchy: no relations, no compound classes."""
    return Representation(view=view, space=space, classes=classes, relations=R.NONE,
                          individuals=I.NONE, measure=measure, composable=composable,
                          **kwargs)


def _tokens(view):
    """Every name is a point; only similarity is defined."""
    return Representation(view=view, space=S.EUCLIDEAN, classes=C.POINT, relations=R.TOKEN,
                          individuals=I.POINT, measure=M.SIMILARITY, composable=Y.YES,
                          notes="The method is its view of the ontology followed by a word "
                                "embedding model; the two are already separate in mOWL.")


def _sequences(view, projection=(), walker=(), corpus=()):
    """A method that learns from sequences (:mod:`mowl.representation.sequence`): every name
    is a point read from the encoder; the measure is a similarity, candidate axioms are
    ranked with a classifier on pairs of vectors (or the similarity)."""
    return Representation(view=view, space=S.EUCLIDEAN, classes=C.POINT, relations=R.TOKEN,
                          individuals=I.POINT, measure=M.SIMILARITY, composable=Y.YES,
                          projection=projection, walker=walker, corpus=corpus,
                          encoder=(En.WORD2VEC_SKIPGRAM, En.WORD2VEC_CBOW,
                                   En.FASTTEXT_SKIPGRAM, En.FASTTEXT_CBOW),
                          reading=Rd.TOKEN, scorer=(Sc.PAIR_CLASSIFIER, Sc.SIMILARITY),
                          notes="A view of the ontology, sequences made from it, an encoder "
                                "of the sequences, the vector of each entity's token and a "
                                "scorer of candidate axioms; each part can be exchanged "
                                "(mowl.models.ComposedSequenceModel).")


def _exact(classes, relations, individuals, space=S.EUCLIDEAN, view=V.CANONICAL_MODEL,
           conjunction=K.EXACT, existential=K.EXACT, negation=K.NONE, disjunction=K.NONE,
           **kwargs):
    """A construction: nothing is learned, statements hold or do not."""
    return Representation(view=view, space=space, classes=classes, relations=relations,
                          individuals=individuals, measure=M.EXACT, conjunction=conjunction,
                          existential=existential, negation=negation,
                          disjunction=disjunction, parametrisation=P.SOLVER,
                          composable=Y.NO, **kwargs)


def _query(classes, relations, individuals, measure, negation, space, composable=Y.YES,
           **kwargs):
    return Representation(view=V.QUERIES, space=space, classes=classes, relations=relations,
                          individuals=individuals, measure=measure, conjunction=K.LEARNED,
                          existential=K.EXACT if relations is not R.NETWORK else K.LEARNED,
                          negation=negation, disjunction=K.NONE, composable=composable,
                          **kwargs)


def _kg(relations, individuals=I.POINT, parametrisation=P.TABLE, **kwargs):
    """A knowledge graph model applied to a graph projection: classes and individuals are
    entities alike."""
    return Representation(view=V.GRAPH, space=S.EUCLIDEAN, classes=C.POINT,
                          relations=relations, individuals=individuals,
                          measure=M.TRIPLE_SCORE, parametrisation=parametrisation,
                          composable=Y.YES, **kwargs)


#: Method key -> :class:`~mowl.representation.Representation`.
REPRESENTATIONS = {
    # ---- region methods on normal-form axioms -------------------------------------------
    "elem": _el(C.BALL, R.TRANSLATION, I.REGION, M.CENTRE_DISTANCE, conjunction=K.APPROXIMATE,
                notes="Two balls do not intersect in a ball: the conjunction only appears "
                      "as a loss on the three centres and radii."),
    "emelpp": _el(C.BALL, R.TRANSLATION, I.REGION, M.CENTRE_DISTANCE,
                  conjunction=K.APPROXIMATE,
                  notes="ELEmbeddings with the norm term inside every axiom loss and with "
                        "losses for role inclusions and role chains on the shift vectors."),
    "emelvar": _el(C.BALL, R.TRANSLATION_TOLERANCE, I.REGION, M.CENTRE_DISTANCE,
                   conjunction=K.APPROXIMATE),
    "elbe": _el(C.BOX, R.TRANSLATION, I.REGION, M.BOUNDS),
    "boxel": _el(C.BOX, R.AFFINE, I.REGION, M.SOFT_VOLUME_RATIO),
    "boxsel": _el(C.BOX, R.AFFINE, I.REGION, (M.SOFT_VOLUME_RATIO, M.VOLUME_RATIO),
                  notes="BoxEL's parts; the method adds a statement type (bounds on a "
                        "conditional probability), scored on volumes."),
    "box2el": _el(C.BOX, R.HEAD_TAIL_BOXES, I.NOMINAL, M.BOUNDS, composable=Y.MOSTLY,
                  notes="Every class also has a bump vector that only the relation model "
                        "uses: the class model and the relation model share a parameter."),
    "transbox": _el(C.BOX, R.TRANSLATION_BOX, I.POINT, M.BOUNDS,
                    notes="Role chains are sums of the relation boxes, role inclusions are "
                          "inclusions of them."),
    "elkem": _el(C.BOX, R.AFFINE_BOXES, I.POINT, M.BOUNDS),
    "octagon": _el(C.OCTAGON, R.TRANSLATION, I.POINT, M.BOUNDS,
                   notes="mOWL's adaptation: the paper uses octagons as relations of a "
                         "knowledge graph model."),
    "prism": _el(C.PRISM, R.TRANSLATION, I.POINT, M.BOUNDS),
    "multielbe": _el(C.MULTI_BOX, R.TRANSLATION, I.REGION, M.BOUNDS,
                     conjunction=K.APPROXIMATE,
                     notes="ELBE with the class model replaced; equal to ELBE with one "
                           "component. The intersection of two unions has more components "
                           "than either, so the conjunction keeps the best pair."),
    "multiboxel": _el(C.MULTI_BOX, R.HEAD_TAIL_BOXES, I.NOMINAL, M.BOUNDS,
                      composable=Y.MOSTLY, conjunction=K.APPROXIMATE,
                      notes="Box2EL with the class model replaced; the bump vector is "
                            "shared as in Box2EL."),
    "mcmultibox": _el(C.MULTI_BOX, R.PAIR_REGION, I.REGION, M.SAMPLED_VOLUME,
                      composable=Y.PARTLY, conjunction=K.APPROXIMATE,
                      notes="Relations are unions of boxes over pairs; every score is a "
                            "volume estimated on sampled points, which ties the parts to "
                            "the sample."),
    "regd": _el(C.BOX, R.TRANSLATION, I.REGION, M.BOUNDARY_DEPTH,
                notes="ELBE with the measure replaced."),
    "regd_elem": _el(C.BALL, R.TRANSLATION, I.REGION, M.BOUNDARY_DEPTH,
                     conjunction=K.APPROXIMATE,
                     notes="ELEmbeddings with the measure replaced."),
    "boxlite": Representation(
        view=V.ONTOLOGY, space=S.EUCLIDEAN, classes=C.BOX, relations=R.HEAD_TAIL_BOXES,
        individuals=I.POINT_WITH_BUMP, measure=M.EXACT, conjunction=K.EXACT,
        existential=K.EXACT, negation=K.EXACT, parametrisation=P.SOLVER, composable=Y.NO,
        notes="The boxes are the solution of a convex programme; negation is the convex "
              "complement box of DL-Lite."),
    "grounded": _el((C.MULTI_BOX,), R.PAIR_REGION, I.CANONICAL_ELEMENT, M.BOUNDS,
                    composable=Y.NO, view=(V.NORMAL_FORMS, V.CANONICAL_MODEL),
                    notes="Elements of the canonical model are positioned first and "
                          "classes and relations are regions around them; the parts are "
                          "designed together."),
    "canonical_mixture": Representation(
        view=V.CANONICAL_MODEL, space=S.SIMPLEX, classes=C.FIXED, relations=R.FIXED,
        individuals=(I.CANONICAL_ELEMENT, I.DISTRIBUTION), measure=M.EXACT,
        conjunction=K.EXACT, existential=K.EXACT, parametrisation=(P.REASONER, P.TABLE),
        composable=Y.PARTLY,
        notes="Only the individual model is learned: an individual is a distribution over "
              "the elements of the canonical model."),
    "faithel": Representation(
        view=V.CANONICAL_MODEL, space=S.SIMPLEX, classes=C.CONVEX, relations=R.PAIR_REGION,
        individuals=I.POINT, measure=M.POINT_TO_REGION, conjunction=K.EXACT,
        existential=K.EXACT, parametrisation=(P.SOLVER, P.TABLE), composable=Y.PARTLY,
        notes="Classes and relations are fixed convex regions of a strongly faithful "
              "model; only the individual model is learned."),
    "concept2box": Representation(
        view=(V.NORMAL_FORMS, V.GRAPH), space=S.EUCLIDEAN, classes=(C.SOFT_BOX, C.LINKED),
        relations=(R.NETWORK, R.TRANSLATION), individuals=I.POINT,
        measure=(M.SOFT_VOLUME_RATIO, M.POINT_TO_REGION), conjunction=K.EXACT,
        existential=K.LEARNED, composable=Y.PARTLY,
        notes="Two linked parts: a box model of the classes, whose relations transform "
              "boxes, and a translation model of the individuals; class membership links "
              "them by the distance of a point to a box."),
    "eike": Representation(
        view=(V.NORMAL_FORMS, V.TEXT), space=S.EUCLIDEAN, classes=(C.ELLIPSOID, C.POINT, C.LINKED),
        relations=R.TRANSLATION, individuals=I.POINT,
        measure=(M.CENTRE_DISTANCE, M.SIMILARITY), parametrisation=(P.TABLE, P.TEXT_ENCODER),
        composable=Y.PARTLY,
        notes="Two spaces: ellipsoids for what a class contains and vectors, optionally "
              "from a language model, for what it means. Relations only hold between "
              "individuals."),
    "composed": _el((C.BALL, C.BOX), R.TRANSLATION, I.REGION, (M.CENTRE_DISTANCE, M.BOUNDS),
                    notes="Whatever parts are chosen; the entry lists the parts that can "
                          "be chosen."),
    # ---- hierarchies ---------------------------------------------------------------------
    "order_embeddings": _hierarchy(C.ORDER_CONE, M.ORDER_VIOLATION),
    "cate": Representation(
        view=V.GRAPH, space=S.EUCLIDEAN, classes=C.ORDER_CONE, relations=R.NONE,
        individuals=I.NOMINAL, measure=M.ORDER_VIOLATION, conjunction=K.REIFIED,
        existential=K.REIFIED, negation=K.REIFIED, disjunction=K.REIFIED, composable=Y.YES,
        notes="Order embeddings of the lattice of class expressions that the categorical "
              "projection builds: the view carries the logic, and every compound class "
              "is an element with an embedding of its own."),
    "box_lattice": _hierarchy(C.BOX, M.VOLUME_RATIO, space=S.UNIT_CUBE),
    "gumbel_box": _hierarchy((C.BOX, C.SOFT_BOX), M.SOFT_VOLUME_RATIO, space=S.UNIT_CUBE,
                             notes="The box lattice model with the measure replaced: the "
                                   "classes are the same boxes (the geometry labels them "
                                   "soft box), and the smoothing of the boundaries is in "
                                   "the measure."),
    "bcbox": _hierarchy((C.BOX, C.SOFT_BOX), M.SOFT_VOLUME_RATIO, composable=Y.MOSTLY,
                        notes="Boxes as in the box lattice model, with the smoothed measure "
                              "of Gumbel boxes. Binary codes switch coordinates off per "
                              "class or globally; they are stored with the boxes and read by "
                              "the measure, so the two parts share them."),
    "hyperbolic_cones": _hierarchy(C.GEODESIC_CONE, M.ANGLE, space=S.HYPERBOLIC),
    "shadow_cones": _hierarchy(C.POINT, M.ANGLE, space=S.HYPERBOLIC,
                               notes="A class is a point; the shadow cone that it casts is "
                                     "computed by the measure. Hyperbolic entailment cones, "
                                     "in contrast, give a class a cone of its own."),
    "disk_embeddings": _hierarchy(C.DISK, M.PROTRUSION,
                                  space=(S.HYPERBOLIC, S.SPHERE, S.EUCLIDEAN)),
    "hit": _hierarchy(C.POINT, M.DISTANCE_DEPTH, space=S.HYPERBOLIC, view=V.TEXT,
                      parametrisation=P.TEXT_ENCODER),
    "ont": Representation(
        view=V.TEXT, space=S.HYPERBOLIC, classes=C.POINT, relations=R.ROTATION,
        individuals=I.NONE, measure=M.DISTANCE_DEPTH, conjunction=K.REIFIED,
        existential=K.LEARNED, parametrisation=P.TEXT_ENCODER, composable=Y.PARTLY,
        notes="HiT with a relation model: a rotation (with a scaling) of the hyperbolic "
              "space sends the point of a class to the point of its existential "
              "restriction; its angles are computed from the name of the relation. "
              "Compound classes are also embedded from their verbalisation."),
    # ---- full class expressions ----------------------------------------------------------
    "falcon": Representation(
        view=V.CLASS_EXPRESSIONS, space=S.FINITE_DOMAIN, classes=C.FUZZY_SET,
        relations=R.NETWORK, individuals=I.DOMAIN_ELEMENT, measure=M.FUZZY_IMPLICATION,
        conjunction=K.FUZZY, existential=K.FUZZY, negation=K.FUZZY, disjunction=K.FUZZY,
        operators=(F.PRODUCT, F.GOEDEL, F.LUKASIEWICZ), composable=Y.YES,
        notes="The domain is the named individuals plus sampled anonymous ones; "
              "membership degrees come from networks over their vectors, and a role is "
              "scored on every pair of domain elements. A combination of parts: the "
              "operator family can be exchanged, and so can the relation model."),
    "dfalc": Representation(
        view=V.CLASS_EXPRESSIONS, space=S.FINITE_DOMAIN, classes=C.FUZZY_SET,
        relations=R.FUZZY_RELATION, individuals=I.DOMAIN_ELEMENT,
        measure=M.FUZZY_IMPLICATION, conjunction=K.FUZZY, existential=K.FUZZY,
        negation=K.FUZZY, disjunction=K.FUZZY, composable=Y.YES,
        operators=(F.GOEDEL, F.PRODUCT, F.LUKASIEWICZ),
        notes="Truth tables over a fixed finite domain; fixing the domain is a closed "
              "domain assumption."),
    "alcone": Representation(
        view=V.CLASS_EXPRESSIONS, space=S.EUCLIDEAN, classes=C.AXIS_CONE, relations=R.FUZZY_RELATION,
        individuals=I.REGION, measure=M.FUZZY_IMPLICATION, conjunction=K.FUZZY,
        existential=K.FUZZY, negation=K.FUZZY, disjunction=K.FUZZY, composable=Y.YES,
        operators=(F.LOGIT_GOEDEL, F.GOEDEL, F.PRODUCT, F.LUKASIEWICZ),
        notes="A cone is the set of its half-axes; a relation is a table of logits over "
              "pairs of half-axes. The operators are Goedel operators on the logits of "
              "the half-axes, exact after discretisation."),
    "e2r": Representation(
        view=V.CLASS_EXPRESSIONS, space=S.SUBSPACES, classes=C.SUBSPACE,
        relations=R.SUBSPACE, individuals=I.POINT, measure=M.POINT_TO_REGION,
        conjunction=K.EXACT, existential=K.NONE, negation=K.EXACT, disjunction=K.EXACT,
        composable=Y.PARTLY,
        notes="Quantum logic: negation is the orthogonal complement, disjunction the span."),
    "composed-alc": Representation(
        view=V.CLASS_EXPRESSIONS, space=S.FINITE_DOMAIN, classes=(C.FUZZY_SET, C.AXIS_CONE),
        relations=(R.NETWORK, R.FUZZY_RELATION), individuals=(I.DOMAIN_ELEMENT, I.REGION),
        measure=M.FUZZY_IMPLICATION, conjunction=K.FUZZY, existential=K.FUZZY,
        negation=K.FUZZY, disjunction=K.FUZZY,
        operators=(F.PRODUCT, F.GOEDEL, F.LUKASIEWICZ, F.LOGIT_GOEDEL), composable=Y.YES,
        notes="Whatever parts are chosen; the entry lists the parts that can be chosen."),
    # ---- query answering -----------------------------------------------------------------
    "query2box": _query(C.BOX, R.TRANSLATION_BOX, I.POINT,
                        (M.POINT_TO_REGION, M.BOUNDS), K.NONE, S.EUCLIDEAN,
                        notes="The relation model is that of TransBox (a relation shifts "
                              "and widens a box); the conjunction is a learned attention "
                              "over the boxes, not their intersection (the exact "
                              "intersection is an option). The distance of an answer to "
                              "the query box is a point-to-region distance; two query "
                              "regions are compared by bounds."),
    "betae": _query(C.DISTRIBUTION, R.NETWORK, I.DISTRIBUTION, M.DIVERGENCE, K.EXACT,
                    S.DISTRIBUTIONS, composable=Y.PARTLY,
                    notes="The conjunction is a learned attention over the distributions, "
                          "not an intersection; the negation (inverting the parameters) is "
                          "exact."),
    "cone_qe": _query(C.SECTOR_CONE, R.NETWORK, I.REGION, M.ANGLE, K.APPROXIMATE,
                      S.ANGLES, composable=Y.PARTLY,
                      notes="The conjunction is a learned attention over the cones; the "
                            "negation is the complement of every sector, which is not the "
                            "complement of the cone (a product of sectors): it misses the "
                            "points outside in some coordinates and inside in others."),
    "acone": _query(C.SECTOR_CONE, R.ROTATION, I.REGION, M.ANGLE, K.APPROXIMATE, S.ANGLES,
                    notes="ConE with the relation model replaced. The conjunction is "
                          "learned as in ConE; the negation is ConE's complement of every "
                          "sector (approximate, see cone_qe)."),
    # ---- knowledge graph models on a graph projection ------------------------------------
    "graph_kge": Representation(
        view=V.GRAPH, space=S.EUCLIDEAN, classes=C.POINT,
        relations=(R.TRANSLATION, R.ROTATION, R.MATRIX), individuals=I.POINT,
        measure=M.TRIPLE_SCORE, composable=Y.YES,
        notes="Any projection with any knowledge graph model of PyKEEN, which has its own "
              "split into representations and an interaction."),
    "boxe": _kg(R.HEAD_TAIL_BOXES, individuals=I.POINT_WITH_BUMP),
    "expressive": _kg(R.PAIR_REGION),
    "reshuffle": _kg(R.MATRIX, parametrisation=P.GRAPH_ENCODER,
                     notes="Entity embeddings are computed by message passing from fixed "
                           "random features, so unseen entities get an embedding."),
    # ---- walks and sentences -------------------------------------------------------------
    "dl2vec": _sequences((V.GRAPH, V.CORPUS), projection=Pj.DL2VEC,
                         walker=(Wk.DEEPWALK, Wk.NODE2VEC)),
    "owl2vec_star": _sequences((V.GRAPH, V.CORPUS, V.TEXT), projection=Pj.OWL2VEC_STAR,
                               walker=(Wk.DEEPWALK, Wk.NODE2VEC), corpus=Co.ANNOTATIONS),
    "taxonomy_walks": _sequences((V.GRAPH, V.CORPUS), projection=(
        Pj.TAXONOMY, Pj.TAXONOMY_WITH_RELATIONS), walker=(Wk.DEEPWALK, Wk.NODE2VEC)),
    "gda_projection": _tokens((V.GRAPH, V.CORPUS)),
    "onto2vec": _sequences(V.CORPUS, corpus=Co.AXIOMS),
    "opa2vec": _sequences((V.CORPUS, V.TEXT), corpus=(Co.AXIOMS, Co.ANNOTATIONS)),
    "composed_sequences": _sequences(
        (V.GRAPH, V.CORPUS, V.TEXT),
        projection=(Pj.TAXONOMY, Pj.TAXONOMY_WITH_RELATIONS, Pj.DL2VEC, Pj.OWL2VEC_STAR,
                    Pj.GDA), walker=(Wk.DEEPWALK, Wk.NODE2VEC),
        corpus=(Co.AXIOMS, Co.ANNOTATIONS)),
    # ---- constructions -------------------------------------------------------------------
    "canonical_model": _exact(C.FIXED, R.FIXED, I.CANONICAL_ELEMENT, space=S.FINITE_DOMAIN,
                              view=V.ONTOLOGY),
    "least_model": _exact(C.FIXED, R.FIXED, I.CANONICAL_ELEMENT, space=S.FINITE_DOMAIN,
                          view=V.ONTOLOGY),
    "box_milp": _exact(C.BOX, R.NONE, I.NONE, view=V.ONTOLOGY, existential=K.NONE),
    "interval_union": _exact(C.INTERVAL_UNION, R.PAIR_REGION, I.POINT),
    "convex_polyhedral": _exact(C.CONVEX, R.PAIR_REGION, I.POINT, space=S.SIMPLEX),
    "convex_regions": _exact(C.CONVEX, R.PAIR_REGION, I.POINT, space=S.SIMPLEX),
    "alcone-exact": _exact(C.AXIS_CONE, R.MATRIX, I.REGION, negation=K.EXACT,
                           disjunction=K.EXACT),
    "temporal_cones": _exact(C.AXIS_CONE, R.NONE, I.REGION, negation=K.EXACT,
                             disjunction=K.EXACT, existential=K.NONE),
    "attributed_dl": _exact(C.CONVEX, R.PAIR_REGION, I.POINT, space=S.SIMPLEX),
}
