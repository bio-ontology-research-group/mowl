"""A model for ALC axioms put together from parts."""

import inspect

import torch as th
from tqdm import trange

from mowl.base_models.alcmodel import EmbeddingALCModel
from mowl.representation import methods as M
from mowl.representation.library.logic_alc import ExpressionGeometry

_THING = "http://www.w3.org/2002/07/owl#Thing"
_NOTHING = "http://www.w3.org/2002/07/owl#Nothing"


class ComposedALC(EmbeddingALCModel):
    """A model for :math:`\\mathcal{ALC}` axioms, put together from parts: a class model, a
    relation model, a measure and an objective (:mod:`mowl.representation`). It trains on the
    axioms of the ontology whatever the depth of their class expressions
    (:class:`~mowl.datasets.alc.ALCDataset`): subclass, equivalence and disjointness axioms
    between class expressions with :math:`\\sqcap, \\sqcup, \\neg, \\exists, \\forall`, class
    assertions and role assertions. :class:`~mowl.models.ComposedEL` trains the same kind of
    parts on the normal forms of :math:`\\mathcal{EL}` only.

    Either name a published combination,

    .. code-block:: python

       model = ComposedALC(dataset, method="dfalc", embed_dim=20)

    or choose the parts:

    .. code-block:: python

       model = ComposedALC(dataset, classes="fuzzy set (product)", relations="fuzzy relation",
                           measure="conflict", objective=FuzzyObjective(), embed_dim=20)

    :param method: Key of :data:`mowl.representation.METHODS` whose geometry reads ALC axioms
        (``"dfalc"``, ``"falcon"``, ``"alcone"``); ``method_options`` are passed to its
        function.
    :param classes: Name of a class model (:data:`mowl.representation.CLASS_MODELS`) or a
        function ``(size, embed_dim)``.
    :param relations: Name of a relation model, or such a function.
    :param measure: Name of a measure, or a measure object; ``None``: the default of the
        class model.
    :param objective: An :class:`~mowl.representation.library.logic_alc.ExpressionReading`
        objective, such as :class:`~mowl.representation.library.logic_fuzzy.FuzzyObjective`
        (default: ``FuzzyObjective(margin)``).
    :param regularizers: Penalties (:class:`mowl.training.Regularizer`) added to the loss.
    :param schedule: A :class:`~mowl.representation.library.logic_alc.TrainingSchedule`
        (what the loop does around the optimiser step), or ``None``: the one of ``method``.
    :param embed_dim: Dimension of the embeddings (the size of the domain for a geometry whose
        classes are fuzzy sets).
    :param alpha: Weight of the TBox loss term (see :class:`EmbeddingALCModel`).
    :param beta: Weight of the concept-assertion loss term; the role-assertion term is
        weighted ``1 - alpha - beta``.
    :param anon_e: Number of anonymous entities sampled each epoch for a geometry that has a
        domain with a context (FALCON).
    :param margin: Margin of the default objective, or of ``method``.
    """

    #: The training loop is the model's own: a training regime is not consulted.
    training_hooks = frozenset()

    def __init__(self, dataset, method=None, classes="fuzzy set", relations="fuzzy relation",
                 measure=None, objective=None, regularizers=(), schedule=None, embed_dim=50,
                 batch_size=256, alpha=0.3, beta=0.3, anon_e=0, margin=0.0,
                 learning_rate=0.001, model_filepath=None, device="cpu", **method_options):
        super().__init__(dataset, embed_dim=embed_dim, batch_size=batch_size, alpha=alpha,
                         beta=beta, anon_e=anon_e, learning_rate=learning_rate,
                         model_filepath=model_filepath, device=device)
        if method is not None and method not in M.METHODS:
            raise ValueError(f"Unknown method {method!r}; available: {sorted(M.METHODS)}.")
        if method is None and method_options:
            raise TypeError(f"Unexpected arguments {sorted(method_options)}.")
        self.method = method
        self.classes, self.relations, self.measure = classes, relations, measure
        self.objective = objective
        self.regularizers = tuple(regularizers)
        self.schedule = schedule
        self.margin = margin
        self.method_options = method_options
        self.init_module()

    # -- the module ------------------------------------------------------------------------
    def _sizes(self):
        return (len(self.dataset.classes), len(self.dataset.object_properties),
                len(self.dataset.individuals))

    def _build(self):
        """The geometry of the chosen parts."""
        classes, relations, individuals = self._sizes()
        if self.method is not None:
            build = M.METHODS[self.method]
            options = dict(self.method_options)
            parameters = inspect.signature(build).parameters
            if "margin" in parameters:
                options["margin"] = self.margin
            cid = self.class_index_dict
            for name, iri in (("top_index", _THING), ("bottom_index", _NOTHING)):
                if name in parameters and name not in options:
                    options[name] = cid.get(iri)
            module = build(classes, relations, max(individuals, 1), embed_dim=self.embed_dim,
                           **options)
            if self.regularizers:
                module.regularizers = module.regularizers + self.regularizers
            return module
        from mowl.representation.library.logic_fuzzy import DomainElements, FuzzyObjective
        class_factory = M._part(self.classes, M.CLASS_MODELS, "class model")
        class_model = class_factory(classes, self.embed_dim)
        relation_model = None
        if self.relations is not None and relations:
            relation_model = M._part(self.relations, M.RELATION_MODELS, "relation model")(
                relations, self.embed_dim)
        measure = self.measure
        if measure is None:
            measure = M.DEFAULT_MEASURE[class_model.region]()
        elif isinstance(measure, str):
            measure = M._part(measure, M.MEASURES, "measure")()
        domain = getattr(class_model, "domain_size", None)
        if domain is not None:
            individual_model = DomainElements(max(individuals, 1), domain,
                                              family=class_model.family)
        else:
            individual_model = class_factory(max(individuals, 1), self.embed_dim)
        objective = self.objective if self.objective is not None \
            else FuzzyObjective(self.margin)
        return ExpressionGeometry(class_model, relation_model, measure, objective,
                                  individuals=individual_model,
                                  regularizers=self.regularizers, schedule=self.schedule)

    def init_module(self):
        module = self._build()
        if not isinstance(module, ExpressionGeometry):
            raise TypeError(
                f"The geometry of {self.method!r} ({type(module).__name__}) does not read ALC "
                f"axioms (it is not an ExpressionGeometry): ComposedALC cannot train it. "
                f"Methods it trains: dfalc, falcon, alcone.")
        if self.schedule is not None and hasattr(module, "schedule"):
            module.schedule = self.schedule
        self.module = module.to(self.device)
        self._attach_assertions()

    def _build_assertion_dicts(self):
        """``(tail, role) -> heads`` and ``(head, role) -> tails`` of the asserted role
        assertions of the ontology, for the negative sampling of assertion losses (a row of
        the ALC dataset is ``[object, role, subject]``, here head is its first column)."""
        heads, tails = {}, {}
        triples = self.training_dataset.get_obj_prop_assertion_data()
        if triples is None:
            return heads, tails
        for item in triples:
            h, r, t = item[0]
            h, r, t = h.item(), r.item(), t.item()
            heads.setdefault((t, r), []).append(h)
            tails.setdefault((h, r), []).append(t)
        return heads, tails

    def _attach_assertions(self):
        attach = getattr(getattr(self.module, "objective", None), "attach_assertions", None)
        if attach is not None and getattr(self.module.objective, "abox", "") == "sampled":
            heads, tails = self._build_assertion_dicts()
            attach(heads, tails, max(len(self.dataset.individuals), 1))

    # -- anonymous entities --------------------------------------------------------------------
    def sample_anonymous_entities(self):
        """The anonymous entities of this epoch, or ``None`` for a geometry without a domain
        that is completed by a sample (``anon_e == 0``)."""
        if self.anon_e == 0 or getattr(self.module, "domain", None) is None:
            return None
        return super().sample_anonymous_entities()

    def _entity_context(self, anon_e_emb):
        if anon_e_emb is None:
            return None
        return super()._entity_context(anon_e_emb)

    # -- scores ----------------------------------------------------------------------------------
    def _subsumption_costs(self, sub, sup, x, e_emb):
        """Cost of ``sub ⊑ sup`` per row: the ``inside`` of the measure with no margin, as in
        the training loss of the module."""
        from mowl.representation.library.logic_expressions import pattern_region
        module = self.module
        domain = module._context(e_emb)
        try:
            a, column = pattern_region(module, sub, x, 0)
            b, _ = pattern_region(module, sup, x, column)
            return module.measure.inside(a, b, 0.0).reshape(len(x), -1).sum(-1)
        finally:
            if domain is not None:
                domain.context = None

    # -- training -------------------------------------------------------------------------------
    def _epoch_loss(self, dataloaders, e_emb):
        total = super()._epoch_loss(dataloaders, e_emb)
        if self.module.regularizers:
            reg = self.module.regularization_loss()
            total = reg if total is None else total + reg
        return total

    def train(self, epochs, validate_every=1):
        """Trains the model: one optimiser step per epoch on the alpha/beta-weighted loss of
        the ALC axioms and the regularisers, with the hooks of the training schedule of the
        geometry (:class:`~mowl.representation.library.logic_alc.TrainingSchedule`).

        :param epochs: Number of training epochs.
        :param validate_every: Run validation (and checkpoint the best model) every this many
            epochs. Defaults to ``1``.
        """
        if self.module is None:
            self.init_module()
        self.module = self.module.to(self.device)
        schedule = getattr(self.module, "schedule", None)

        optimizer = th.optim.Adam(self.module.parameters(), lr=self.learning_rate)
        best_loss = float("inf")
        if schedule is not None:
            schedule.on_train_start(self.module, epochs)

        for epoch in trange(epochs):
            self.module.train()
            if schedule is not None:
                schedule.on_epoch_start(self.module, epoch, epochs)
            anon_e_emb = self.sample_anonymous_entities()
            e_emb = self._entity_context(anon_e_emb)

            loss = self._epoch_loss(self.training_dataloaders, e_emb)
            if loss is None:
                raise ValueError("No trainable axioms were found in the training ontology.")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if schedule is not None:
                schedule.after_step(self.module, epoch, epochs)
            project = getattr(self.module, "project", None)
            if project is not None:
                with th.no_grad():
                    project()

            valid_loss = None
            if self.dataset.validation and (epoch + 1) % validate_every == 0:
                if schedule is not None:
                    schedule.on_validation(self.module)
                with th.no_grad():
                    self.module.eval()
                    anon_e_emb = self.sample_anonymous_entities()
                    e_emb = self._entity_context(anon_e_emb)
                    valid = self._epoch_loss(self.validation_dataloaders, e_emb)
                    valid_loss = valid.item() if valid is not None else None

                if valid_loss is not None and valid_loss < best_loss:
                    best_loss = valid_loss
                    if self.model_filepath is not None:
                        th.save(self.module.state_dict(), self.model_filepath)

        if schedule is not None:
            schedule.on_train_end(self.module)
        if self.model_filepath is not None and best_loss == float("inf"):
            th.save(self.module.state_dict(), self.model_filepath)

    @property
    def as_geometry(self):
        """The module, which is a :class:`~mowl.representation.Geometry`."""
        return self.module
