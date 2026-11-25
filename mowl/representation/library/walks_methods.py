"""The methods that learn from sequences, as combinations of parts.

========================  ===================================================================
Method                    Combination
========================  ===================================================================
``dl2vec``                DL2Vec projection, DeepWalk walks, word2vec skip-gram, token
                          vectors, a pair classifier for ranking
``owl2vec_star``          OWL2Vec* projection, DeepWalk walks (with ``lexical=True`` mixed
                          with annotation sentences), word2vec skip-gram, ...
``taxonomy_walks``        taxonomy projection, DeepWalk walks, word2vec skip-gram, ...
``onto2vec``              axiom sentences, word2vec skip-gram, ...
``opa2vec``               axiom sentences followed by annotation sentences, word2vec
                          skip-gram, ...
========================  ===================================================================

All five are :func:`compose_sequences` with other arguments; every part can be swapped
through the arguments (``walker="node2vec"``, ``corpus=("axioms",)`` added to the walks,
``sg=0``, ``encoder="fasttext"``, ``scorer="similarity"``, ``entailment=Entailment("elk")``)
or by passing a part object. The defaults are the ones of the comparison runs of
:mod:`mowl.benchmarks.families`, not the settings of a paper.

The functions have the signature of :func:`~mowl.representation.register_method`
(``nb_classes, nb_relations, nb_individuals, embed_dim, ...``) and are registered with
``trained_on="sequences"``; the sizes are not used (the vocabulary comes from the
sequences). :class:`~mowl.models.ComposedSequenceModel` trains them.
"""


from mowl.representation.methods import register_method
from mowl.representation.sequence import SequenceGeometry, OntologyView, SequenceSource
from mowl.representation.library.walks_views import (
    projection_view, ProjectionView, AxiomCorpus, AnnotationCorpus, Walks, Sentences, Mixed,
    make_walker)
from mowl.representation.library.walks_encoders import (
    ENCODERS, READINGS, PAIR_MEASURES, SCORERS)

__all__ = ["compose_sequences", "dl2vec", "owl2vec_star", "taxonomy_walks", "onto2vec",
           "opa2vec", "SEQUENCE_METHODS"]


def _named(value, registry, kind, **options):
    if isinstance(value, str):
        if value not in registry:
            raise ValueError(f"Unknown {kind} {value!r}; available: {sorted(registry)}.")
        return registry[value](**options)
    return value


def _corpus_view(kind, entailment, annotation_entity_form):
    if isinstance(kind, OntologyView):
        return kind
    if kind == "axioms":
        return AxiomCorpus(entailment=entailment)
    if kind == "annotations":
        return AnnotationCorpus(entailment=entailment, entity_form=annotation_entity_form)
    raise ValueError(f"Unknown corpus {kind!r}; use 'axioms', 'annotations' or a view.")


#: The defaults of the options that only a named walker or encoder reads.
_WALK_DEFAULTS = dict(num_walks=10, walk_length=20, alpha=0.1, p=1.0, q=1.0, walks_file=None)
_ENCODER_DEFAULTS = dict(sg=1, window=5, min_count=1, restart=False)


def _unused(owner, why, **options):
    """Options that were given (not ``None``) to a part that does not read them: an error,
    not silence."""
    given = sorted(k for k, v in options.items() if v is not None)
    if given:
        raise TypeError(f"{given} given, but {owner} {why}.")


def compose_sequences(nb_classes=0, nb_relations=0, nb_individuals=0, embed_dim=None,
                      projection=None, projection_options=None, walker=None,
                      num_walks=None, walk_length=None, alpha=None, p=None, q=None,
                      walks_file=None, fresh_walks=True, corpus=(),
                      annotation_entity_form="iri", entailment=None, encoder="word2vec",
                      sg=None, window=None, min_count=None, workers=None, seed=None,
                      word2vec_options=None, restart=None, reading="token",
                      measure="dot sigmoid", scorer="pair classifier",
                      scorer_options=None, source=None):
    """A sequence method from freely chosen parts.

    View and sequences:

    :param projection: A projection name (:data:`~.walks_views.PROJECTIONS`, or
        ``"taxonomy"``, ``"taxonomy_rels"``, ``"dl2vec"``, ``"owl2vec_star"``, ``"gda"``),
        a projector of :mod:`mowl.projection`, a graph view, or ``None`` for no walks.
    :param projection_options: Options of the projector (``bidirectional_taxonomy``, ...).
    :param walker: ``"deepwalk"``, ``"node2vec"`` or a :class:`mowl.walking.WalkingModel`
        (then ``num_walks``, ``walk_length``, ``alpha``, ``p``, ``q``, ``walks_file`` are
        not used).
    :param fresh_walks: See :class:`~.walks_views.Walks` (``fresh``).
    :param corpus: Corpus views whose sentences follow the walks: ``"axioms"``,
        ``"annotations"`` or view objects.
    :param annotation_entity_form: See :class:`~.walks_views.AnnotationCorpus`.
    :param entailment: An :class:`~mowl.representation.sequence.Entailment` applied to every
        view (the axioms a reasoner entails are read too).
    :param source: A :class:`~mowl.representation.sequence.SequenceSource` to use instead of
        all of the above.

    Encoder (:data:`~.walks_encoders.ENCODERS`): ``encoder`` is ``"word2vec"``,
    ``"fasttext"`` or an encoder object; ``embed_dim``, ``sg``, ``window``, ``min_count``,
    ``workers``, ``seed`` and ``word2vec_options`` (any other option of gensim) configure a
    named one, ``restart`` see :class:`~.walks_encoders.Word2VecEncoder`. (Defaults: the
    option is ``None``, which means ``embed_dim=100``, ``sg=1``, ``window=5``,
    ``min_count=1``, ``workers=4``, ``seed=0``, ``restart=False``, ``walker="deepwalk"``,
    ``num_walks=10``, ``walk_length=20``, ``alpha=0.1``, ``p=q=1``.) An option that the
    chosen parts do not read (``window`` with an encoder object, ``walker`` without a
    projection, ``scorer_options`` with a scorer object) raises ``TypeError``.

    Reading, measure, scorer: names in :data:`~.walks_encoders.READINGS`,
    :data:`~.walks_encoders.PAIR_MEASURES`, :data:`~.walks_encoders.SCORERS` (with
    ``scorer_options`` for a named scorer) or objects.
    """
    walk_options = dict(num_walks=num_walks, walk_length=walk_length, alpha=alpha, p=p, q=q,
                        walks_file=walks_file)
    if source is not None:
        _unused("a source was given", "which replaces them", projection=projection,
                projection_options=projection_options, walker=walker, **walk_options)
        if corpus:
            raise TypeError("corpus given, but a source was given, which replaces it.")
    elif projection is None:
        _unused("there is no projection to walk on", "(a walker needs one)",
                walker=walker, projection_options=projection_options)
        _unused("this method has no walks (no projection)", "that they would configure",
                **walk_options)
    if scorer_options and not isinstance(scorer, str):
        raise TypeError("scorer_options given, but the scorer is an object.")
    if source is None:
        sources = []
        if projection is not None:
            if walker is not None and not isinstance(walker, str):
                _unused("the walker is an object, which has its own settings",
                        "(they are options of a named walker)", **walk_options)
            if walker is None:
                walker = "deepwalk"
            if isinstance(projection, str):
                view = projection_view(projection, entailment=entailment,
                                       **(projection_options or {}))
            elif isinstance(projection, OntologyView):
                view = projection
            else:
                view = ProjectionView(projection, entailment=entailment)
            if isinstance(walker, str):
                walk = {**_WALK_DEFAULTS, **{k: v for k, v in walk_options.items()
                                             if v is not None}}
                walker = make_walker(walker, num_walks=walk["num_walks"],
                                     walk_length=walk["walk_length"], alpha=walk["alpha"],
                                     p=walk["p"], q=walk["q"],
                                     workers=4 if workers is None else workers,
                                     seed=0 if seed is None else seed,
                                     outfile=walk["walks_file"])
            sources.append(Walks(view, walker, fresh=fresh_walks))
        for kind in ((corpus,) if isinstance(corpus, (str, OntologyView)) else corpus):
            sources.append(Sentences(_corpus_view(kind, entailment, annotation_entity_form)))
        if not sources:
            raise ValueError("A sequence method needs a projection to walk on or a corpus.")
        source = sources[0] if len(sources) == 1 else Mixed(*sources)
    elif not isinstance(source, SequenceSource):
        raise TypeError("source must be a SequenceSource.")
    if isinstance(encoder, str):
        if encoder not in ENCODERS:
            raise ValueError(f"Unknown encoder {encoder!r}; available: {sorted(ENCODERS)}.")
        opts = {**_ENCODER_DEFAULTS, **{k: v for k, v in dict(
            sg=sg, window=window, min_count=min_count, restart=restart).items()
            if v is not None}}
        encoder = ENCODERS[encoder](vector_size=100 if embed_dim is None else embed_dim,
                                    workers=4 if workers is None else workers,
                                    seed=0 if seed is None else seed, **opts,
                                    **(word2vec_options or {}))
    else:
        if not (source is None and projection is not None and isinstance(walker, str)):
            _unused("neither a named encoder nor a named walker reads them", "",
                    workers=workers, seed=seed)
        _unused("the encoder is an object, which has its own settings",
                "(they are options of a named encoder)", embed_dim=embed_dim, sg=sg,
                window=window, min_count=min_count, restart=restart,
                word2vec_options=word2vec_options or None)
    return SequenceGeometry(source, encoder,
                            _named(reading, READINGS, "reading"),
                            _named(measure, PAIR_MEASURES, "measure"),
                            _named(scorer, SCORERS, "scorer", **(scorer_options or {})))


@register_method("dl2vec", trained_on="sequences")
def dl2vec(nb_classes=0, nb_relations=0, nb_individuals=0, embed_dim=None,
           bidirectional_taxonomy=True, **options):
    """DL2Vec (Chen et al. 2020): the DL2Vec projection of the class axioms, random walks,
    word2vec skip-gram. Options: those of :func:`compose_sequences`."""
    projection = options.pop("projection", "dl2vec")
    return compose_sequences(embed_dim=embed_dim, projection=projection, projection_options={
        "bidirectional_taxonomy": bidirectional_taxonomy}, **options)


@register_method("owl2vec_star", trained_on="sequences")
def owl2vec_star(nb_classes=0, nb_relations=0, nb_individuals=0, embed_dim=None,
                 bidirectional_taxonomy=True, include_literals=False, lexical=False,
                 **options):
    """OWL2Vec* (Chen et al. 2021): the OWL2Vec* projection, random walks, word2vec
    skip-gram.

    :param lexical: Also the lexical document of the original: the annotation sentences
        after the walks (off by default, as in :mod:`mowl.benchmarks.families`).
    :param include_literals: Annotation triples as edges of the graph.
    """
    projection = options.pop("projection", "owl2vec_star")
    corpus = tuple(options.pop("corpus", ()))
    if lexical and "annotations" not in corpus:
        corpus += ("annotations",)
    return compose_sequences(embed_dim=embed_dim, projection=projection,
                             projection_options={
                                 "bidirectional_taxonomy": bidirectional_taxonomy,
                                 "include_literals": include_literals},
                             corpus=corpus, **options)


@register_method("taxonomy_walks", trained_on="sequences")
def taxonomy_walks(nb_classes=0, nb_relations=0, nb_individuals=0, embed_dim=None,
                   bidirectional_taxonomy=True, relations=False, **options):
    """Random walks on the class hierarchy (DeepWalk or node2vec), word2vec skip-gram.

    :param relations: Also an ``r`` edge for every :math:`A \\sqsubseteq \\exists r.B`
        (the taxonomy-with-relations projection).
    """
    name = options.pop("projection", "taxonomy_rels" if relations else "taxonomy")
    return compose_sequences(embed_dim=embed_dim, projection=name, projection_options={
        "bidirectional_taxonomy": bidirectional_taxonomy}, **options)


@register_method("onto2vec", trained_on="sequences")
def onto2vec(nb_classes=0, nb_relations=0, nb_individuals=0, embed_dim=None, **options):
    """Onto2Vec (Smaili et al. 2018): every axiom a sentence, word2vec skip-gram. To read
    the entailed axioms too, pass ``entailment=Entailment(...)``."""
    options.setdefault("corpus", ("axioms",))
    return compose_sequences(embed_dim=embed_dim, **options)


@register_method("opa2vec", trained_on="sequences")
def opa2vec(nb_classes=0, nb_relations=0, nb_individuals=0, embed_dim=None, **options):
    """OPA2Vec (Smaili et al. 2019): the axiom sentences of Onto2Vec followed by the
    annotation sentences (labels, definitions, synonyms), word2vec skip-gram. mOWL does not
    pre-train on an external text corpus."""
    options.setdefault("corpus", ("axioms", "annotations"))
    return compose_sequences(embed_dim=embed_dim, **options)


#: The methods of this module by catalogue key.
SEQUENCE_METHODS = {"dl2vec": dl2vec, "owl2vec_star": owl2vec_star,
                    "taxonomy_walks": taxonomy_walks, "onto2vec": onto2vec,
                    "opa2vec": opa2vec}
