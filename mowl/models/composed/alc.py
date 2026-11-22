"""A model for ALC axioms put together from parts."""

import inspect

import torch as th

from mowl.base_models.alcmodel import EmbeddingALCModel
from mowl.evaluation.validation import reached_parameters
from mowl.representation import methods as M
from mowl.representation.regions import ConstructorUnavailable
from mowl.representation.swap import NO_PART, build_method, current_parts
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
    :param classes: Name of a class model (:data:`mowl.representation.CLASS_MODELS`), a
        function ``(size, embed_dim)`` or a class model. With ``method`` it replaces the
        class model of the method (so do the other parts below;
        :func:`mowl.representation.swap.build_method`).
    :param relations: Name of a relation model, such a function or a relation model;
        ``"none"``: none.
    :param individuals: As ``classes``, for the individuals.
    :param objective_options: With ``method``: options of the method's own objective that
        are replaced (:meth:`~mowl.representation.Objective.replace`).
    :param measure: Name of a measure, or a measure object; ``None``: the default of the
        class model.
    :param objective: An :class:`~mowl.representation.library.logic_alc.ExpressionReading`
        objective, such as :class:`~mowl.representation.library.logic_fuzzy.FuzzyObjective`
        (default: ``FuzzyObjective(margin)``). Its ``weighting``
        (:class:`~mowl.representation.library.logic_alc.AxiomKindWeighting`) combines the
        costs of the axiom patterns into the loss; an objective without one gets the
        weighting of ``alpha`` and ``beta``.
    :param regularizers: Penalties (:class:`mowl.training.Regularizer`) added to the loss.
    :param schedule: A :class:`~mowl.representation.library.logic_alc.TrainingSchedule`
        (a training regime of the geometry, such as the annealing of Al-cone), or ``None``:
        the one of ``method``.
    :param embed_dim: Dimension of the embeddings (the size of the domain for a geometry whose
        classes are fuzzy sets).
    :param alpha: Weight of the TBox loss term (see
        :class:`~mowl.representation.library.logic_alc.AxiomKindWeighting`); ``None``: the one
        of the weighting of ``objective``, or ``0.3``. Giving it together with an objective
        that has a weighting raises :class:`TypeError`.
    :param beta: Weight of the concept-assertion loss term; the role-assertion term is
        weighted ``1 - alpha - beta``. ``None`` as for ``alpha``.
    :param anon_e: Number of anonymous entities sampled each epoch for a geometry that has a
        domain with a context (FALCON).
    :param margin: Margin of the default objective, or of ``method``.

    Training is the shared loop of :mod:`mowl.training.loop`
    (:meth:`~mowl.base_models.EmbeddingALCModel.train`); the schedule of the geometry is its
    training regime. A combination of parts (a ``method`` with replaced parts, or parts
    without a method) is checked when the model is built: a combination that cannot evaluate
    the axioms of the ontology, or a replaced part that no axiom reads, raises
    :class:`~mowl.representation.ConstructorUnavailable` with the reason
    (:meth:`check_combination`).
    """

    def __init__(self, dataset, method=None, classes=None, relations=None,
                 measure=None, objective=None, regularizers=(), schedule=None, embed_dim=50,
                 individuals=None, objective_options=None,
                 batch_size=256, alpha=None, beta=None, anon_e=0, margin=0.0,
                 learning_rate=0.001, model_filepath=None, device="cpu", **method_options):
        self._given_weights = (alpha, beta)
        super().__init__(dataset, embed_dim=embed_dim, batch_size=batch_size,
                         alpha=0.3 if alpha is None else alpha,
                         beta=0.3 if beta is None else beta, anon_e=anon_e,
                         learning_rate=learning_rate, model_filepath=model_filepath,
                         device=device)
        if method is not None and method not in M.METHODS:
            raise ValueError(f"Unknown method {method!r}; available: {sorted(M.METHODS)}.")
        if method is None and method_options:
            raise TypeError(f"Unexpected arguments {sorted(method_options)}.")
        if method is None and objective_options:
            raise TypeError("objective_options are options of the objective of a method; "
                            "without one, pass objective=FuzzyObjective(...).")
        if method is None:
            classes = "fuzzy set" if classes is None else classes
            relations = "fuzzy relation" if relations is None else relations
        self.method = method
        self.classes, self.relations, self.measure = classes, relations, measure
        self.individuals, self.objective_options = individuals, objective_options
        self.objective = objective
        self.regularizers = tuple(regularizers)
        self.schedule = schedule
        self.margin = margin
        self.method_options = method_options
        self.init_module()
        self._attach_weighting()
        if self.swapped_parts or method is None:
            self.check_combination()

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
            return build_method(
                self.method, classes, relations, max(individuals, 1),
                embed_dim=self.embed_dim, classes=self.classes, relations=self.relations,
                individuals=self.individuals, measure=self.measure, objective=self.objective,
                objective_options=self.objective_options, regularizers=self.regularizers,
                **options)
        from mowl.representation.library.logic_fuzzy import DomainElements, FuzzyObjective
        class_factory = M._part(self.classes, M.CLASS_MODELS, "class model")
        class_model = class_factory(classes, self.embed_dim)
        relation_model = None
        if self.relations is not None and relations and self.relations != NO_PART:
            relation_model = M._part(self.relations, M.RELATION_MODELS, "relation model")(
                relations, self.embed_dim)
        measure = self.measure
        if measure is None:
            measure = M.DEFAULT_MEASURE[class_model.region]()
        elif isinstance(measure, str):
            measure = M._part(measure, M.MEASURES, "measure")()
        domain = getattr(class_model, "domain_size", None)
        if self.individuals is not None and self.individuals != NO_PART:
            individual_model = M._part(self.individuals, M.CLASS_MODELS, "individual model")
            if not isinstance(individual_model, th.nn.Module):
                individual_model = individual_model(max(individuals, 1), self.embed_dim)
        elif domain is not None:
            individual_model = DomainElements(max(individuals, 1), domain,
                                              family=class_model.family)
        else:
            individual_model = class_factory(max(individuals, 1), self.embed_dim)
        objective = self.objective if self.objective is not None \
            else FuzzyObjective(self.margin)
        return ExpressionGeometry(class_model, relation_model, measure, objective,
                                  individuals=individual_model,
                                  regularizers=self.regularizers, schedule=self.schedule)

    @property
    def swapped_parts(self):
        """The part arguments given together with ``method``."""
        given = {"classes": self.classes, "relations": self.relations,
                 "individuals": self.individuals, "measure": self.measure,
                 "objective": self.objective, "objective_options": self.objective_options,
                 "regularizers": self.regularizers or None}
        return {k: v for k, v in given.items() if v is not None} if self.method else {}

    def init_module(self):
        module = self._build()
        if not isinstance(module, ExpressionGeometry):
            raise ConstructorUnavailable(
                f"The geometry of {self.method!r} ({type(module).__name__}) does not read ALC "
                f"axioms (it is not an ExpressionGeometry): ComposedALC cannot train it. "
                f"Methods it trains: dfalc, falcon, alcone; subspace logic ('e2r') is "
                f"trained by mowl.models.ComposedSubspace.")
        if self.schedule is not None and hasattr(module, "schedule"):
            module.schedule = self.schedule
        self.module = module.to(self.device)
        self._attach_assertions()
        # the regimes of the new geometry are attached when it trains
        self.__dict__.pop("_training_regime", None)
        self._attach_weighting()

    @property
    def weighting(self):
        """The objective part that combines the costs of the axiom patterns into a loss
        (``module.objective.weighting``; the objective gets the one of ``alpha`` and
        ``beta`` if it has none)."""
        objective = getattr(self.module, "objective", None)
        if objective is not None and getattr(objective, "weighting", False) is None:
            self._attach_weighting()
        return super().weighting

    def _attach_weighting(self):
        """The objective of the geometry gets the weighting of ``alpha`` and ``beta`` unless
        it has one; ``alpha``/``beta`` are then those of its weighting."""
        objective = getattr(self.module, "objective", None)
        if objective is None or not hasattr(objective, "axiom_cost"):
            return
        given = getattr(self, "_given_weights", (None, None))
        if getattr(objective, "weighting", None) is None:
            objective.weighting = self.own_weighting
        elif given != (None, None) and objective.weighting is not self.own_weighting:
            raise TypeError(
                f"The weights of the loss terms are given twice: alpha/beta {given} and the "
                f"weighting of the objective ({objective.weighting!r}); give one of them.")
        weighting = objective.weighting
        self.alpha, self.beta = weighting.tbox, weighting.class_assertions

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
        the training loss of the module
        (:meth:`~mowl.representation.library.logic_alc.ExpressionGeometry.inclusion_costs`)."""
        return self.module.inclusion_costs(sub, sup, x, e_emb)

    def score_subsumption(self, sub, sup):
        """Score of :math:`A \\sqsubseteq B` for class IRIs ``sub`` and ``sup``: minus the
        cost of the inclusion with margin 0 (``0`` iff the geometry satisfies it)."""
        cid = self.class_index_dict
        with th.no_grad():
            cost = self.module.class_inclusion_costs(
                th.tensor([cid[sub]], device=self.device),
                th.tensor([cid[sup]], device=self.device))
        return -float(cost)

    # -- training -------------------------------------------------------------------------------
    def _epoch_loss(self, dataloaders, e_emb):
        """The weighted loss of ``{axiom: batches}`` plus the regularisers of the geometry."""
        total = super()._epoch_loss(dataloaders, e_emb)
        if self.module.regularizers:
            reg = self.module.regularization_loss()
            total = reg if total is None else total + reg
        return total

    def check_combination(self):
        """Refuse a combination of parts that cannot train on the ontology, when the model is
        built rather than in the first epoch.

        Every axiom pattern of the training data is evaluated once on its first row (with
        the random state kept, so that the run that follows is the one it would have been),
        then the cost of a subsumption is scored (what the evaluation ranks). A part that
        cannot evaluate an axiom says why (:class:`~mowl.representation.ConstructorUnavailable`
        from the part, or :meth:`ExpressionGeometry.refuses
        <mowl.representation.library.logic_alc.ExpressionGeometry.refuses>`: parts that do
        not evaluate on the geometry's domain, what the objective requires). A replaced class,
        relation or individual model whose parameters no axiom reaches would change nothing
        and is refused as well, and the schedule of the geometry is attached as its training
        regime (a schedule the geometry cannot take raises :class:`TypeError`).

        :raises ConstructorUnavailable: With the method, the replaced parts and the reason.
        """
        given = self.swapped_parts if self.method is not None else {
            k: v for k, v in (("classes", self.classes), ("relations", self.relations),
                              ("individuals", self.individuals), ("measure", self.measure),
                              ("objective", self.objective)) if v is not None}
        name = repr(self.method) if self.method is not None else "The combination"
        objective = getattr(self.module, "objective", None)
        if "objective" in given and not hasattr(objective, "axiom_cost"):
            raise ConstructorUnavailable(
                f"{name} cannot take an objective that is not an ALC objective: the loss of "
                f"{type(self.module).__name__} is the cost of an axiom pattern "
                f"(objective.axiom_cost), which {type(objective).__name__} does not have.")
        # the regimes of the geometry (its schedule) are attached now: one that cannot be
        # used with the geometry raises TypeError here, not in the first epoch
        self.training_regime
        try:
            reached = self._probe()
        except ConstructorUnavailable as error:
            raise ConstructorUnavailable(
                f"{name} cannot take {sorted(given)}: {error} (the parts: "
                f"{self.module.parts}).") from error
        parts = current_parts(self.module)
        for kind in ("classes", "relations", "individuals"):
            part = parts[kind]
            if kind not in given or part is None:
                continue
            own = [id(p) for p in part.parameters()]
            if own and not reached.intersection(own):
                raise ConstructorUnavailable(
                    f"{name} cannot take {kind}={given[kind]!r}: the {kind} part "
                    f"{type(part).__name__} is not read by the loss of any axiom of the "
                    f"ontology (no gradient reaches its parameters), so replacing it would "
                    f"change nothing (the parts: {self.module.parts}).")

    def _probe(self):
        """Evaluate every trained axiom pattern on one row and one subsumption score; return
        the ids of the parameters the gradient reaches."""
        reason = self.module.refuses()
        if reason:
            raise ConstructorUnavailable(reason)
        self._load_datasets()
        weighting = self.weighting

        def loss():
            context = self._entity_context(self.sample_anonymous_entities())
            first_rows = {axiom: [(data.tensors[0][:1],)]
                          for axiom, data in self._training_datasets.items() if len(data)}
            return weighting.loss(self.module, first_rows, context, self.device)

        def score():
            self.axiom_costs("gci0", th.zeros(1, 2, dtype=th.long))
            self.__dict__.pop("_axiom_scoring_context", None)

        return reached_parameters(self.module, loss, after=score)

    @property
    def as_geometry(self):
        """The module, which is a :class:`~mowl.representation.Geometry`."""
        return self.module
