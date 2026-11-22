"""Query embedding models (Query2Box, BetaE, ConE, AConE) on ontologies, from parts.

One model class for the four methods and for every other combination of their parts. A
method is a named combination (:func:`mowl.representation.library.query_methods.query2box`
and its siblings, registered in :data:`mowl.representation.METHODS` with
``trained_on="queries"``): which entity model, relation model, intersection operator,
negation operator and distance make up the geometry (``model.module``, a
:class:`~mowl.representation.library.query_methods.ComposedQueryGeometry`). The model adds
the other parts of a training and evaluation regime: the objective
(:class:`~mowl.representation.library.query_objective.QueryObjective`), the query sampler
(:mod:`mowl.models.query.sampler`), the training regime hooks
(:mod:`mowl.training.regime`) and the evaluation regimes
(:class:`mowl.evaluation.LogicalQueryAnswering`
through :meth:`QueryEmbeddingModel.score_query`, and the rank-based regimes through
:meth:`QueryEmbeddingModel.axiom_costs`). The class expression to query translation is
:mod:`mowl.nn.query.expression`.
"""

import math

import numpy as np
import torch as th

from mowl.base_models.model import Model
from mowl.models.query import sampling
from mowl.models.query.graph import ISA_INV, extract_graph, role_relation
from mowl.models.query.sampler import ReleasedQueries, SampledQueries
from mowl.nn.query.base import UnsupportedOperator
from mowl.nn.query.expression import (Vocabulary, class_expression_to_query, from_nested,
                                      ids, signature, to_dnf)
from mowl.representation.library.query_objective import QueryObjective
from mowl.representation.methods import METHODS

from mowl.training.regime import SCHEDULE_HOOKS  # noqa: E402


def _check_options(method, builder, given):
    """An argument the builder has no part or option for raises ``TypeError`` that names the
    accepted ones."""
    import inspect
    parameters = inspect.signature(builder).parameters
    skipped = {"nb_classes", "nb_relations", "nb_individuals", "embed_dim", "gamma", "tables",
               "num_classes", "extension"}
    unknown = sorted(set(given) - set(parameters))
    if unknown:
        raise TypeError(f"The method {method!r} takes no {unknown}; accepted: "
                        f"{sorted(set(parameters) - skipped)}.")


class QueryEmbeddingModel(Model):
    """A query embedding model of an ontology: :class:`~mowl.models.Query2Box`,
    :class:`~mowl.models.BetaE`, :class:`~mowl.models.ConE` and :class:`~mowl.models.AConE`
    are this class with a method named, and any other combination of parts is this class
    with the parts named.

    **The ontology as a graph.** Entities are the named classes and the named individuals;
    edges are ``A ⊑ B`` and ``a : B`` (relation ``isa``), ``A ⊑ ∃r.B`` and ``r(a, b)``
    (relation ``r``), each with its inverse (see :func:`~mowl.models.query.graph.extract_graph`).
    This is an adaptation: the methods were published for knowledge graphs. A class
    expression is a query on this graph (:attr:`vocabulary`): a named class is
    "everything below it" (``isa`` inverse from the class), ``∃r.C`` projects the answers
    of ``C`` backwards along ``r``, ``⊓``, ``⊔``, ``¬`` (and ``∀r.C = ¬∃r.¬C``) are the
    operators. Its answers are the entities that fall under it; named classes can be
    answers, so ``A ⊑ B`` is "``A`` is an answer of ``B``".

    **Training** minimises the negative sampling loss of
    :class:`~mowl.representation.library.query_objective.QueryObjective`
    (``-log σ(logit(answer)) - mean log σ(-logit(negative))``, all four reference codes) with
    uniformly drawn negative entities. Where the queries come from is the ``queries``
    argument (the query sampler, a training regime in the sense of mOWL: the model does not
    depend on it):

    * ``"sampled"`` (default): queries of the structures 1p, 2p, 3p, 2i, 3i, ip, pi (and the
      negation structures 2in, 3in, inp, pin, pni for geometries with negation) are sampled
      from the graph, backwards from a random answer, in every step
      (:class:`~mowl.models.query.sampler.SampledQueries`).
    * ``"released"``: the training queries of a released query dataset (the dataset must
      have ``training_queries(entity_id, relation_id, names)``, e.g.
      :class:`~mowl.datasets.builtin.query_sets.QueryAnsweringDataset`), as in the four
      papers (:class:`~mowl.models.query.sampler.ReleasedQueries`: batches, weights and the
      order of updates of the reference code).

    :param dataset: mOWL dataset (the training ontology is read).
    :param embed_dim: Embedding dimension.
    :param gamma: Margin of the score.
    :param learning_rate: Adam learning rate.
    :param batch_size: Queries per optimisation step.
    :param negatives: Negative entities per query.
    :param structures: Names of the training structures (default: all the geometry
        supports and the graph has).
    :param hierarchy_closure: Close ``isa`` transitively (and reflexively on classes) when
        reading the ontology.
    :param extension: Whether a named class in a class expression query means everything
        below it (default) or only the class entity itself.
    :param seed: Seed of the query sampler.
    :param queries: ``"sampled"`` or ``"released"`` (see above).
    :param steps_per_epoch: Steps (see above) per epoch; :meth:`train` uses it unless it is
        given there. The papers count steps (300,000 to 450,000): epochs times this.
    :param batch_per_structure: With ``queries="sampled"``: ``batch_size`` queries of every
        structure per step (Query2Box, Appendix) instead of ``batch_size`` in total.
    :param method: The combination of parts: a key of :data:`mowl.representation.METHODS`
        trained on queries (``"query2box"``, ``"betae"``, ``"cone_qe"``, ``"acone"``) or a
        function with the signature of those builders. The named classes set it.
    :param objective: A :class:`~mowl.representation.library.query_objective.QueryObjective`
        (default: the negative sampling loss of the papers).
    :param geometry_kwargs: Arguments of the builder of the method, which are the parts and
        their options: ``intersection``, ``distance``, ``negation``, ``center_weight``,
        ``activation``, ``hidden`` and so on (see the builders in
        :mod:`mowl.representation.library.query_methods`). An argument that the method has
        no part for raises ``TypeError``.
    """

    axiom_scoring = "method"
    scored_forms = ("gci0", "gci2", "class_assertion")
    #: The objective (sampled queries, negative-sampling loss) is the method's own; regimes
    #: that set the optimiser, a schedule or a stopping rule can be attached.
    training_hooks = SCHEDULE_HOOKS
    #: The key of the named combination (a :data:`mowl.representation.METHODS` entry).
    method = None
    #: The module class that the builder of the method returns (named classes only).
    geometry_class = None
    default_gamma = 24.0
    #: Options of the builder that this named method sets to its published or earlier
    #: reading when the caller does not choose.
    geometry_defaults = {}

    def __init__(self, dataset, embed_dim=32, gamma=None, learning_rate=1e-3, batch_size=64,
                 negatives=16, structures=None, hierarchy_closure=True, extension=True,
                 seed=0, queries="sampled", steps_per_epoch=10, batch_per_structure=False,
                 model_filepath=None, device="cpu", method=None, objective=None,
                 **geometry_kwargs):
        super().__init__(dataset, model_filepath=model_filepath)
        method = self.method if method is None else method
        if method is None:
            raise ValueError("Name the combination: method='query2box', 'betae', 'cone_qe', "
                             "'acone' or a builder function.")
        builder = self._builder(method)
        self.embed_dim = embed_dim
        self.gamma = self.default_gamma if gamma is None else gamma
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.negatives = negatives
        self.extension = extension
        if queries not in ("sampled", "released"):
            raise ValueError("queries must be 'sampled' or 'released'.")
        self.query_source = queries
        self.steps_per_epoch = int(steps_per_epoch)
        self.batch_per_structure = bool(batch_per_structure)
        self.device = device
        self.objective = QueryObjective() if objective is None else objective
        self.rng = np.random.default_rng(seed)
        self._class_ids = dict(super().class_index_dict)
        self._individual_ids = dict(super().individual_index_dict)
        self._role_ids = dict(super().object_property_index_dict)
        self.num_classes = len(self._class_ids)
        self.graph, self.skipped_axioms = extract_graph(
            dataset.ontology, self._class_ids, self._individual_ids, self._role_ids,
            closure=hierarchy_closure)
        self.entity_names = [None] * self.graph.num_entities
        for iri, i in self._class_ids.items():
            self.entity_names[i] = iri
        for iri, i in self._individual_ids.items():
            self.entity_names[self.num_classes + i] = iri
        options = {**self.geometry_defaults, **geometry_kwargs}
        _check_options(method, builder, geometry_kwargs)
        self.module = builder(
            self.num_classes, len(self._role_ids), len(self._individual_ids), embed_dim,
            self.gamma, num_classes=self.num_classes, extension=extension,
            **options).to(device)
        self.vocabulary = Vocabulary(
            class_id=self._class_entity, individual_id=self._individual_entity,
            relation_id=self._relation, isa_inverse=ISA_INV, extension=extension)
        templates = sampling.structures(self.module.supports_negation, structures)
        self.loss_history = []
        if queries == "released":
            self.sampler = ReleasedQueries(
                dataset, self.graph, templates, self.rng, self.entity_id, self.relation_id,
                structures=structures, batch_size=batch_size, negatives=negatives,
                device=device)
        else:
            self.sampler = SampledQueries(
                self.graph, templates, self.rng, batch_size=batch_size, negatives=negatives,
                batch_per_structure=self.batch_per_structure, device=device)

    @staticmethod
    def _builder(method):
        if callable(method):
            return method
        if method not in METHODS:
            raise ValueError(f"Unknown method {method!r}.")
        from mowl.representation.methods import METHOD_KINDS
        if METHOD_KINDS.get(method) != "queries":
            raise ValueError(f"{method!r} is trained on axioms, not on queries; use "
                             f"ComposedEL for it.")
        return METHODS[method]

    @property
    def parts(self):
        """The combination: ``{"method": key, **module.parts}`` and the other parts of the
        regime."""
        return {"method": self.method, **self.module.parts,
                "intersection": type(self.module.intersection).__name__,
                "distance": type(self.module.distance_part).__name__,
                "objective": f"{self.objective.kind} negative sampling",
                "sampler": type(self.sampler).__name__}

    # -- the query sampler ---------------------------------------------------------------
    @property
    def training_structures(self):
        """``{name: template}`` of the structures trained (sampled queries; empty for the
        released ones)."""
        return self.sampler.structures

    @property
    def _pool(self):
        return self.sampler.pool

    @property
    def released(self):
        """The released training queries by structure (``queries="released"``)."""
        return self.sampler.queries

    @property
    def _released_columns(self):
        return self.sampler._columns

    def _released_batch(self, group):
        """One batch of the released sampler (``"path"`` or ``"other"``)."""
        return self.sampler.batch(group)

    # -- ids ----------------------------------------------------------------------------
    def entity_id(self, iri):
        """Entity id of a named class (the id used in :meth:`score_query`)."""
        return self._class_ids[iri]

    def relation_id(self, role_iri, inverse=False):
        """Relation id of a role (``inverse``: the relation from filler to subject)."""
        return role_relation(self._role_ids[role_iri], inverse)

    def _class_entity(self, iri):
        return self._class_ids[iri]

    def _individual_entity(self, iri):
        return self.num_classes + self._individual_ids[iri]

    def _relation(self, iri, inverse):
        return role_relation(self._role_ids[iri], inverse)

    @property
    def triples(self):
        """The training graph as rows ``(head, relation, tail)`` of entity and relation ids
        (with the inverse edges), e.g. ``known`` for
        :func:`mowl.evaluation.sample_queries`."""
        return self.graph.triples

    # -- training -----------------------------------------------------------------------
    def loss(self, groups):
        """The objective on a batch of the sampler (``{name: (signature, rows, positives,
        negatives[, weights])}``)."""
        return self.objective(self.module, groups)

    released_loss = loss

    def get_optimizer(self):
        """The optimiser of :meth:`train` (Adam with ``learning_rate``)."""
        return th.optim.Adam(self.module.parameters(), lr=self.learning_rate)

    def train(self, epochs=10, steps_per_epoch=None, **kwargs):
        """Train with Adam. One epoch is ``steps_per_epoch`` steps (default: the value of
        the constructor), each on ``batch_size`` queries (sampled afresh or drawn from the
        released training queries, see the class). The mean loss per epoch is appended to
        :attr:`loss_history`. If ``model_filepath`` was given the parameters are saved.

        :raises ValueError: If no query structure can be sampled from the graph (an
            ontology without subclass or role axioms).
        """
        steps_per_epoch = self.steps_per_epoch if steps_per_epoch is None else steps_per_epoch
        self.sampler.check()
        regime = self.training_regime
        opt = NotImplemented if regime is None else regime.optimizer()
        if opt is NotImplemented:
            opt = self.get_optimizer()
        self.module.train()
        for epoch in range(epochs):
            if regime is not None:
                regime.on_epoch_start(epoch, epochs)
            total, updates = 0.0, 0
            for _ in range(steps_per_epoch):
                for batch in self.sampler.updates():
                    loss = self.loss(batch)
                    if regime is not None:
                        extra = regime.additional_loss(epoch)
                        if extra is not None:
                            loss = loss + extra
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    total += float(loss)
                    updates += 1
            self.loss_history.append(total / updates)
            if regime is not None:
                regime.on_epoch_end(epoch, {"total": self.loss_history[-1]})
                if regime.should_stop(epoch):
                    break
        if regime is not None:
            regime.on_train_end()
        self.module.eval()
        if abs(self.loss_history[-1] - math.log(2)) < 0.02 and epochs > 0:
            import warnings
            warnings.warn(
                f"{type(self).__name__}: the loss is log 2, i.e. every score is 0 and the "
                f"model has collapsed (for ConE-type geometries the tanh of the projection "
                f"output saturates: all queries get the same cone). Lower learning_rate.",
                UserWarning, stacklevel=2)
        if self._model_filepath is not None:
            th.save(self.module.state_dict(), self._model_filepath)

    # -- queries ------------------------------------------------------------------------
    def query(self, class_expression):
        """The query (:class:`~mowl.nn.query.expression.Node`) of an OWL API class
        expression."""
        return class_expression_to_query(class_expression, self.vocabulary)

    def _logits(self, nodes, entity_ids=None, max_cells=2 ** 24):
        """Scores of queries against entities: ``(len(nodes), N)`` for all entities, or
        ``(len(nodes), K)`` for ``entity_ids`` of that shape. A union is the best branch
        of the disjunctive normal form."""
        n_entities = self.graph.num_entities
        branches = {}
        for q, node in enumerate(nodes):
            for b in to_dnf(node):
                branches.setdefault(signature(b), []).append((q, ids(b)))
        width = n_entities if entity_ids is None else entity_ids.shape[1]
        out = th.full((len(nodes), width), -float("inf"), device=self.device)
        chunk = max(1, max_cells // max(1, width * self.embed_dim))
        for sig, items in branches.items():
            for start in range(0, len(items), chunk):
                part = items[start:start + chunk]
                qs = th.tensor([q for q, _ in part], dtype=th.long, device=self.device)
                rows = th.tensor([r for _, r in part], dtype=th.long, device=self.device)
                state = self.module.embed(sig, rows)
                ent = (th.arange(n_entities, device=self.device).expand(len(part), -1)
                       if entity_ids is None else entity_ids[qs])
                out[qs] = th.maximum(out[qs], self.module.logit(state, ent))
        return out

    @th.no_grad()
    def score_query(self, structure, queries):
        """Scores of a batch of queries of one structure against all entities (higher:
        better), the interface of :class:`mowl.evaluation.LogicalQueryAnswering`.

        :param structure: Nested-tuple structure (:func:`mowl.nn.query.expression.from_nested`).
        :param queries: Instances of ``structure`` with entity and relation ids of this
            model (:attr:`triples`, :attr:`entity_names`).
        :return: ``(len(queries), num_entities)`` tensor.
        """
        nodes = [from_nested(structure, q) for q in queries]
        self._require_operators(nodes)
        self.module.eval()
        return self._logits(nodes).cpu()

    @th.no_grad()
    def answer(self, class_expression, top_k=10):
        """The ``top_k`` entities that best answer a class expression, as
        ``(iri, score)`` pairs, best first."""
        node = self.query(class_expression)
        self._require_operators([node])
        self.module.eval()
        scores = self._logits([node])[0]
        best = th.topk(scores, min(top_k, len(scores)))
        return [(self.entity_names[int(i)], float(s)) for s, i in zip(best.values,
                                                                      best.indices)]

    def _require_operators(self, nodes):
        from mowl.nn.query.expression import contains_negation
        if not self.module.supports_negation and any(contains_negation(n) for n in nodes):
            raise UnsupportedOperator(
                f"{type(self.module).__name__} has no negation operator: the query "
                f"contains ¬ (or ∀).")

    # -- axiom scores for the rank-based regimes ----------------------------------------
    #: Query states built at a time by :meth:`axiom_costs` (bounds its memory: about
    #: ``scoring_chunk * embed_dim * 4`` bytes per state tensor, a few of them at a time).
    scoring_chunk = 2 ** 16

    def _axiom_queries(self, gci_name, rows):
        """``(signature, id_matrix, subjects)`` of candidate axioms, built column-wise on
        the rows' device: every axiom of one form is a query of the same shape (a named
        class, or ``∃r.D``), so no per-row query object is needed."""
        from mowl.nn.query.expression import anchor, project
        rows = rows.long()
        if gci_name == "class_assertion":
            subjects, classes = rows[:, 0] + self.num_classes, rows[:, 1]
        else:
            subjects, classes = rows[:, 0], rows[:, -1]
        cols = [classes]
        node = anchor(0)
        if self.extension:
            cols.append(th.full_like(classes, ISA_INV))
            node = project(ISA_INV, node)
        if gci_name == "gci2":
            cols.append(2 + 2 * rows[:, 1] + 1)            # role_relation(r, inverse=True)
            node = project(0, node)
        return signature(node), th.stack(cols, dim=1), subjects

    @th.no_grad()
    def axiom_costs(self, gci_name, rows):
        """Cost (minus the score; lower is more plausible) of candidate axioms in mOWL's
        normal-form layout with this model's indices.

        ``gci0`` ``[C, D]``: the entity ``C`` is an answer of the query ``D``;
        ``gci2`` ``[C, r, D]``: ``C`` is an answer of ``∃r.D``; ``class_assertion``
        ``[a, C]``: the individual ``a`` is an answer of ``C``.

        The rows are scored on the model's device, :attr:`scoring_chunk` at a time; the
        cost is that of :meth:`axiom_costs_by_query` (the same numbers), without building
        a query object per row (a ranking over all classes scores ``queries x classes``
        rows).
        """
        from mowl.evaluation.model_scoring import UnsupportedAxiomForm
        if gci_name not in self.scored_forms:
            raise UnsupportedAxiomForm(f"{type(self).__name__} has no score for "
                                       f"{gci_name!r} axioms.")
        self.module.eval()
        sig, id_matrix, subjects = self._axiom_queries(gci_name, rows.to(self.device))
        out = th.empty(len(rows), device=self.device)
        step = max(1, int(self.scoring_chunk))
        for start in range(0, len(rows), step):
            stop = start + step
            state = self.module.embed(sig, id_matrix[start:stop])
            out[start:stop] = self.module.logit(state, subjects[start:stop, None])[:, 0]
        return -out

    @th.no_grad()
    def axiom_costs_by_query(self, gci_name, rows):
        """:meth:`axiom_costs` through one query object per row (the general path of
        :meth:`answer`; slow for many rows, kept as the reference of the fast path)."""
        from mowl.evaluation.model_scoring import UnsupportedAxiomForm
        if gci_name not in self.scored_forms:
            raise UnsupportedAxiomForm(f"{type(self).__name__} has no score for "
                                       f"{gci_name!r} axioms.")
        from mowl.nn.query.expression import project
        rows = rows.to("cpu").long()
        nodes, subjects = [], []
        for row in rows.tolist():
            if gci_name == "gci0":
                c, d = row
                nodes.append(self._concept(d))
                subjects.append(c)
            elif gci_name == "gci2":
                c, r, d = row
                nodes.append(project(role_relation(r, True), self._concept(d)))
                subjects.append(c)
            else:
                a, c = row
                nodes.append(self._concept(c))
                subjects.append(self.num_classes + a)
        self.module.eval()
        ent = th.tensor(subjects, dtype=th.long, device=self.device).unsqueeze(1)
        return -self._logits(nodes, ent)[:, 0]

    def _concept(self, class_id):
        from mowl.nn.query.expression import anchor, project
        node = anchor(class_id)
        return project(ISA_INV, node) if self.extension else node

    def score_subsumption(self, sub, sup):
        """Score (higher is better) of ``A ⊑ B`` for class IRIs."""
        row = th.tensor([[self._class_ids[sub], self._class_ids[sup]]])
        return -float(self.axiom_costs("gci0", row)[0])

    # -- embeddings ---------------------------------------------------------------------
    @property
    def evaluation_model(self):
        return self

    @property
    def class_embeddings(self):
        """Class IRI to the vector of the entity (the point, center, axis or Beta
        parameters of the entity)."""
        v = self.module.entity_vectors().cpu().numpy()
        return {iri: v[i] for iri, i in self._class_ids.items()}

    @property
    def individual_embeddings(self):
        v = self.module.entity_vectors().cpu().numpy()
        return {iri: v[self.num_classes + i] for iri, i in self._individual_ids.items()}

    @property
    def object_property_embeddings(self):
        """Role IRI to the vector of the relation that goes from subject to filler."""
        v = self.module.relation_vectors().cpu().numpy()
        return {iri: v[role_relation(i)] for iri, i in self._role_ids.items()}
