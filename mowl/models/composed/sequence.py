"""A model that learns from sequences, put together from parts."""

import os

import torch as th

import mowl.error.messages as msg
from mowl.base_models.model import Model
from mowl.representation import methods as M
from mowl.representation.sequence import SequenceGeometry
from mowl.training.schedules import linear_decay_interval

__all__ = ["ComposedSequenceModel", "SequenceScores"]


class SequenceScores(th.nn.Module):
    """The ``evaluation_model`` of a sequence model: ``forward(rows)`` is the measure of the
    geometry (its ``inside``) between the first and the last column of ``rows`` (class
    indices, two or three columns), one cost per row."""

    def __init__(self, geometry):
        super().__init__()
        self.geometry = geometry

    def forward(self, data, *args, **kwargs):
        if data.shape[1] == 2:
            x, y = data[:, 0], data[:, 1]
        elif data.shape[1] == 3:
            x, y = data[:, 0], data[:, 2]
        else:
            raise ValueError("Data must have 2 or 3 columns")
        g = self.geometry
        return g.inside(g.classes(x), g.classes(y)).reshape(-1).float().to(data.device)


class ComposedSequenceModel(Model):
    """A model that learns from walks or sentences, put together from parts
    (:mod:`mowl.representation.sequence`): a view of the ontology, a source of sequences, an
    encoder, an entity reading, a measure and a scorer.

    Either name a published combination (a key of :data:`mowl.representation.METHODS`
    registered with ``trained_on="sequences"``; ``method_options`` go to its function),

    .. testcode::

       from mowl.datasets.builtin import FamilyDataset
       from mowl.models import ComposedSequenceModel

       model = ComposedSequenceModel(FamilyDataset(), method="dl2vec", walker="node2vec",
                                     embed_dim=16, workers=1)

    or pass the parts (``source``, ``encoder``, ``reading``, ``measure``, ``scorer``); a
    part passed together with ``method`` replaces that part of the method.

    :param method: Key of the combination. Without a method, a source and an encoder, the
        model has the parts of ``"dl2vec"`` (and ``method_options`` are its options).
    :param source, encoder, reading, measure, scorer: A part (an object, taken as it is, or a
        name, which is an option of the method) that replaces that part of ``method``.
    :param view: The projection of the ontology that the walks run on (a name, a projector
        or a view; ``projection`` of the method) and ``walker``: ``"deepwalk"``,
        ``"node2vec"`` or a walking model. Only a method with walks has them; for the others
        they raise ``TypeError``.
    :param device: Where the scorer computes (the encoders of gensim run on the CPU).

    Training (:meth:`train`) writes the sequences the first time (and again after
    :meth:`add_axioms`), reads their vocabulary into the encoder and trains it. An encoder
    that has a vocabulary keeps its vectors when it is trained again (see ``restart`` of
    :class:`~mowl.representation.library.walks_encoders.Word2VecEncoder`). With a training
    regime (:attr:`training_regime`; the hooks ``on_epoch_start``, ``on_epoch_end``,
    ``should_stop``, ``on_train_end``) the encoder is trained one pass at a time with the
    learning rate falling over all passes, and a regime can make new sequences between
    passes (:class:`mowl.training.sequences.RefreshSequences`).

    Evaluation regimes rank candidate axioms with the scorer of the geometry
    (``axiom_scoring = "method"``, :meth:`axiom_costs`).
    """

    axiom_scoring = "method"
    #: The forms the default scorer (a pair classifier) can score.
    scored_forms = ("gci0", "gci2")
    training_hooks = frozenset({"on_epoch_start", "on_epoch_end", "should_stop",
                                "on_train_end"})

    def __init__(self, dataset, method=None, source=None, encoder=None, reading=None,
                 measure=None, scorer=None, model_filepath=None, device="cpu",
                 _base_options=None, view=None, walker=None, **method_options):
        super().__init__(dataset, model_filepath=model_filepath, **(_base_options or {}))
        if method is None and source is None and encoder is None:
            method = "dl2vec"       # no parts chosen: the parts of DL2Vec
        if view is not None:
            method_options["projection"] = view
        if walker is not None:
            method_options["walker"] = walker
        if method is not None:
            # a part given by name or as an object is an option of the method, which
            # builds the geometry with it; a replaced source takes the place of the walks
            if source is not None:
                clash = sorted({"projection", "projection_options", "walker", "num_walks",
                                "walk_length", "alpha", "p", "q", "walks_file", "corpus",
                                "fresh_walks", "entailment", "annotation_entity_form"}
                               & set(method_options))
                if clash:
                    raise TypeError(f"{clash} describe the walks and sentences of the "
                                    f"method, which source= replaces.")
            for name, part in (("encoder", encoder), ("reading", reading),
                               ("measure", measure), ("scorer", scorer)):
                if part is not None:
                    method_options[name] = part
            encoder = reading = measure = scorer = None
            if method not in M.METHODS or M.METHOD_KINDS.get(method) != "sequences":
                raise ValueError(
                    f"{method!r} is not a method that learns from sequences; those are "
                    f"{sorted(k for k, v in M.METHOD_KINDS.items() if v == 'sequences')}.")
            geometry = self._build(method, method_options)
            if source is not None:
                geometry.source = source
        else:
            if method_options:
                raise TypeError(f"Unexpected arguments {sorted(method_options)}.")
            if view is not None or walker is not None:
                raise TypeError("view and walker are parts of a method: pass method=...")
            from mowl.representation.library.walks_encoders import (TokenVectors, DotSigmoid,
                                                                    PairClassifier)
            from mowl.representation.library.walks_views import Walks
            geometry = SequenceGeometry(source if source is not None else Walks(), encoder,
                                        reading if reading is not None else TokenVectors(),
                                        measure if measure is not None else DotSigmoid(),
                                        scorer if scorer is not None else PairClassifier())
        self.method = method
        self.module = geometry.bind(self)
        self.device = th.device(device) if isinstance(device, str) else device
        self.update_w2v_model = False
        self.axioms_added = False
        self._written = False
        #: The training loss of every call of the encoder (``None`` if it reports none).
        self.losses = []

    def _build(self, method, options):
        """The geometry of ``method``; an option it has no part for raises ``TypeError``
        that names the accepted ones."""
        import inspect
        from mowl.representation.library import walks_methods
        build = M.METHODS[method]
        accepted = set(inspect.signature(build).parameters)
        if any(p.kind is inspect.Parameter.VAR_KEYWORD
               for p in inspect.signature(build).parameters.values()):
            accepted |= set(inspect.signature(walks_methods.compose_sequences).parameters)
        accepted -= {"nb_classes", "nb_relations", "nb_individuals", "options"}
        unknown = sorted(set(options) - accepted)
        if unknown:
            raise TypeError(f"The method {method!r} takes no {unknown}; accepted: "
                            f"{sorted(accepted)} (and the parts source, encoder, reading, "
                            f"measure, scorer).")
        return build(len(self.class_index_dict), len(self.object_property_index_dict),
                     len(self.individual_index_dict), **options)

    # -- parts --------------------------------------------------------------------------
    @property
    def source(self):
        return self.module.source

    @property
    def encoder(self):
        return self.module.encoder

    @encoder.setter
    def encoder(self, value):
        self.module.encoder = value
        self.module.refresh()

    @property
    def corpus_path(self):
        """The file with the sequences the encoder is trained on."""
        return self.module.source.path

    @property
    def w2v_model(self):
        """The gensim model of the encoder (``None`` if there is no encoder)."""
        encoder = self.module.encoder
        return None if encoder is None else getattr(encoder, "model", None)

    @w2v_model.setter
    def w2v_model(self, model):
        from mowl.representation.library.walks_encoders import Word2VecEncoder
        self.encoder = None if model is None else Word2VecEncoder(model=model)

    @property
    def embed_dim(self):
        return self.module.dimension

    @embed_dim.setter
    def embed_dim(self, value):
        # kept for code that sets it after set_w2v_model; the encoder decides the size
        pass

    def set_w2v_model(self, *args, **kwargs):
        """Use a new :class:`gensim.models.Word2Vec` with these arguments as the encoder
        (:class:`~mowl.representation.library.walks_encoders.Word2VecEncoder`; it also
        takes ``restart``)."""
        from mowl.representation.library.walks_encoders import Word2VecEncoder
        self.encoder = Word2VecEncoder(*args, **kwargs)

    @property
    def scoring_head(self):
        """The scorer of the geometry (what ``scoring="pair_classifier"`` of the evaluation
        regimes calls)."""
        return self.module.scorer

    @scoring_head.setter
    def scoring_head(self, head):
        from mowl.evaluation.model_scoring import PairClassifierHead
        from mowl.representation.library.walks_encoders import PairClassifier
        self.module.scorer = PairClassifier(head=head) \
            if isinstance(head, PairClassifierHead) else head

    # -- embeddings ---------------------------------------------------------------------
    def _check_encoder(self):
        if self.module.encoder is None:
            raise AttributeError(msg.W2V_MODEL_NOT_SET)
        if not self.module.encoder.ready:
            raise AttributeError(msg.RANDOM_WALK_MODEL_EMBEDDINGS_NOT_FOUND)

    def _embeddings(self, names):
        self._check_encoder()
        encoder = self.module.encoder
        return {name: encoder.vector(name) for name in names if encoder.has(name)}

    @property
    def class_embeddings(self):
        return self._embeddings(self.dataset.classes.as_str)

    @property
    def object_property_embeddings(self):
        return self._embeddings(self.dataset.object_properties.as_str)

    @property
    def individual_embeddings(self):
        return self._embeddings(self.dataset.individuals.as_str)

    @property
    def evaluation_model(self):
        """A module whose ``forward(rows)`` is the measure of the geometry between the first
        and the last column (:class:`SequenceScores`)."""
        return SequenceScores(self.module)

    def axiom_costs(self, gci_name, rows):
        """Costs of the candidate axioms ``rows`` (the scorer of the geometry)."""
        return self.module.score(gci_name, rows)

    # -- training -----------------------------------------------------------------------
    def _check_parts(self):
        if self.module.encoder is None:
            raise AttributeError(msg.W2V_MODEL_NOT_SET)

    def _ensure_sequences(self):
        if not self._written or self.axioms_added:
            self.module.source.write(self.dataset)
            self._written = True
            self.axioms_added = False

    def refresh_sequences(self, seed=None):
        """Make the sequences anew (with ``seed`` for their random parts, if given) and add
        new tokens to the encoder; its vectors stay."""
        if seed is not None:
            self.module.source.reseed(seed)
        self.module.source.write(self.dataset)
        self._written = True
        if self.module.encoder.ready:
            self.module.encoder.prepare(self.corpus_path, update=True)

    def train(self, epochs=None):
        """Train the encoder on the sequences.

        :param epochs: Passes over the sequences; default: the ``epochs`` of the encoder.
        """
        self._check_parts()
        self._ensure_sequences()
        encoder = self.module.encoder
        encoder.prepare(self.corpus_path, update=self.update_w2v_model)
        if epochs is None:
            epochs = encoder.epochs
        regime = self.training_regime
        if regime is None:
            self.losses.append(encoder.fit(self.corpus_path, epochs))
        else:
            self._train_with_regime(regime, epochs)
        self.module.refresh()

    def _train_with_regime(self, regime, epochs):
        encoder = self.module.encoder
        rates = getattr(encoder, "learning_rates", None)
        for epoch in range(epochs):
            regime.on_epoch_start(epoch, epochs)
            start = end = None
            if rates is not None:
                start, end = linear_decay_interval(*rates, epoch, epochs)
            loss = encoder.fit(self.corpus_path, 1, start_alpha=start, end_alpha=end)
            self.losses.append(loss)
            regime.on_epoch_end(epoch, {"total": float("nan") if loss is None else loss})
            if regime.should_stop(epoch):
                break
        regime.on_train_end()

    def add_axioms(self, *axioms):
        """Add ``axioms`` to the dataset, make the sequences through the new entities and
        add their tokens to the encoder; the vectors there are stay. The next
        :meth:`train` makes all sequences anew."""
        entities = set()
        for axiom in axioms:
            entities |= set(axiom.getClassesInSignature())
            entities |= set(axiom.getObjectPropertiesInSignature())
            entities |= set(axiom.getIndividualsInSignature())
        self.dataset.add_axioms(*axioms)
        self.module.source.extend(self.dataset, [str(e.toStringID()) for e in entities])
        self.update_w2v_model = True
        if self.module.encoder is not None and self.module.encoder.ready:
            self.module.encoder.prepare(self.corpus_path, update=True)
        self.axioms_added = True
        self.module.refresh()

    def from_pretrained(self, model):
        """Use the gensim model saved at ``model`` (a path) as the encoder; training goes on
        from its vectors."""
        if not isinstance(model, str):
            raise TypeError("Parameter model must be a string pointing to the Word2Vec model "
                            "file.")
        if not os.path.exists(model):
            raise FileNotFoundError("Pretrained model path does not exist")
        self._is_pretrained = True
        from gensim.models import Word2Vec
        from mowl.representation.library.walks_encoders import Word2VecEncoder
        restart = getattr(self.module.encoder, "restart", False)
        self.encoder = Word2VecEncoder(model=Word2Vec.load(model), restart=restart)

    @property
    def as_geometry(self):
        """The module, which is a :class:`~mowl.representation.sequence.SequenceGeometry`."""
        return self.module
