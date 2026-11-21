"""Query embedding models (Query2Box, BetaE, ConE, AConE) on ontologies.

One model class for the four methods: they differ in the geometry of the operators
(:mod:`mowl.nn.query.geometry`), which a subclass names in ``geometry_class``. The graph
they are trained on, the sampling of training queries, the loss, the class expression to
query translation (:mod:`mowl.nn.query.expression`), the scoring of axioms for the
rank-based regimes and ``score_query`` for
:class:`mowl.evaluation.LogicalQueryAnswering` are shared.
"""

import math

import numpy as np
import torch as th
import torch.nn.functional as F

from mowl.base_models.model import Model
from mowl.models.query import sampling
from mowl.models.query.graph import ISA_INV, extract_graph, role_relation
from mowl.nn.query.expression import (Vocabulary, class_expression_to_query, from_nested,
                                      ids, signature, to_dnf)
from mowl.nn.query.geometry import UnsupportedOperator

from mowl.training.regime import SCHEDULE_HOOKS  # noqa: E402


class QueryEmbeddingModel(Model):
    """Base class of :class:`~mowl.models.Query2Box`, :class:`~mowl.models.BetaE`,
    :class:`~mowl.models.ConE` and :class:`~mowl.models.AConE`.

    **The ontology as a graph.** Entities are the named classes and the named individuals;
    edges are ``A ⊑ B`` and ``a : B`` (relation ``isa``), ``A ⊑ ∃r.B`` and ``r(a, b)``
    (relation ``r``), each with its inverse (see :func:`~mowl.models.query.graph.extract_graph`).
    This is an adaptation: the methods were published for knowledge graphs. A class
    expression is a query on this graph (:attr:`vocabulary`): a named class is
    "everything below it" (``isa`` inverse from the class), ``∃r.C`` projects the answers
    of ``C`` backwards along ``r``, ``⊓``, ``⊔``, ``¬`` (and ``∀r.C = ¬∃r.¬C``) are the
    operators. Its answers are the entities that fall under it; named classes can be
    answers, so ``A ⊑ B`` is "``A`` is an answer of ``B``".

    **Training** samples queries of the structures 1p, 2p, 3p, 2i, 3i, ip, pi (and the
    negation structures 2in, 3in, inp, pin, pni for geometries with negation) from the
    graph, backwards from a random answer, and minimises
    ``-log σ(logit(answer)) - mean log σ(-logit(negative))`` (all four reference codes),
    with uniformly drawn negative entities.

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
    :param geometry_kwargs: Extra arguments of the geometry.
    """

    axiom_scoring = "method"
    scored_forms = ("gci0", "gci2", "class_assertion")
    #: The objective (sampled queries, negative-sampling loss) is the method's own; regimes
    #: that set the optimiser, a schedule or a stopping rule can be attached.
    training_hooks = SCHEDULE_HOOKS
    #: The :class:`~mowl.nn.query.geometry.QueryGeometry` subclass of the method.
    geometry_class = None
    default_gamma = 24.0

    def __init__(self, dataset, embed_dim=32, gamma=None, learning_rate=1e-3, batch_size=64,
                 negatives=16, structures=None, hierarchy_closure=True, extension=True,
                 seed=0, model_filepath=None, device="cpu", **geometry_kwargs):
        super().__init__(dataset, model_filepath=model_filepath)
        self.embed_dim = embed_dim
        self.gamma = self.default_gamma if gamma is None else gamma
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.negatives = negatives
        self.extension = extension
        self.device = device
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
        self.module = self.geometry_class(
            self.graph.num_entities, self.graph.num_relations, embed_dim, self.gamma,
            **geometry_kwargs).to(device)
        self.vocabulary = Vocabulary(
            class_id=self._class_entity, individual_id=self._individual_entity,
            relation_id=self._relation, isa_inverse=ISA_INV, extension=extension)
        templates = sampling.structures(self.module.supports_negation, structures)
        pool = sorted(self.graph.inc)
        self._pool = pool
        self.training_structures = (sampling.usable_structures(templates, self.graph,
                                                               self.rng, pool)
                                    if pool else {})
        self.loss_history = []

    # -- ids ----------------------------------------------------------------------------
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
    def _batch(self):
        names = list(self.training_structures)
        per = max(1, self.batch_size // len(names))
        groups = {}
        for name in names:
            template = self.training_structures[name]
            rows, pos, negs = [], [], []
            for _ in range(per):
                sample = sampling.sample_query(template, self.graph, self.rng, self._pool)
                if sample is None:
                    continue
                query, found, answer = sample
                rows.append(ids(query))
                pos.append(answer)
                negs.append(sampling.negative_entities(
                    found, self.negatives, self.graph.num_entities, self.rng))
            if rows:
                dev = self.device
                groups[name] = (signature(template),
                                th.tensor(rows, dtype=th.long, device=dev),
                                th.tensor(pos, dtype=th.long, device=dev).unsqueeze(1),
                                th.tensor(np.array(negs), dtype=th.long, device=dev))
        return groups

    def loss(self, groups):
        """The negative sampling loss of a batch from :meth:`_batch`."""
        pos_loss, neg_loss, count = 0.0, 0.0, 0
        for sig, rows, pos, neg in groups.values():
            state = self.module.embed(sig, rows)
            p = self.module.logit(state, pos)
            n = self.module.logit(state, neg)
            pos_loss = pos_loss - F.logsigmoid(p).sum()
            neg_loss = neg_loss - F.logsigmoid(-n).mean(dim=1).sum()
            count += len(rows)
        return (pos_loss + neg_loss) / (2 * count)

    def get_optimizer(self):
        """The optimiser of :meth:`train` (Adam with ``learning_rate``)."""
        return th.optim.Adam(self.module.parameters(), lr=self.learning_rate)

    def train(self, epochs=10, steps_per_epoch=10, **kwargs):
        """Train with Adam. One epoch is ``steps_per_epoch`` optimisation steps, each on
        ``batch_size`` freshly sampled queries. The mean loss per epoch is appended to
        :attr:`loss_history`. If ``model_filepath`` was given the parameters are saved.

        :raises ValueError: If no query structure can be sampled from the graph (an
            ontology without subclass or role axioms).
        """
        if not self.training_structures:
            raise ValueError("No training query can be sampled from the graph of this "
                             "ontology: it has no subclass, role or assertion edges.")
        regime = self.training_regime
        opt = NotImplemented if regime is None else regime.optimizer()
        if opt is NotImplemented:
            opt = self.get_optimizer()
        self.module.train()
        for epoch in range(epochs):
            if regime is not None:
                regime.on_epoch_start(epoch, epochs)
            total = 0.0
            for _ in range(steps_per_epoch):
                loss = self.loss(self._batch())
                if regime is not None:
                    extra = regime.additional_loss(epoch)
                    if extra is not None:
                        loss = loss + extra
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += float(loss)
            self.loss_history.append(total / steps_per_epoch)
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
    @th.no_grad()
    def axiom_costs(self, gci_name, rows):
        """Cost (minus the score; lower is more plausible) of candidate axioms in mOWL's
        normal-form layout with this model's indices.

        ``gci0`` ``[C, D]``: the entity ``C`` is an answer of the query ``D``;
        ``gci2`` ``[C, r, D]``: ``C`` is an answer of ``∃r.D``; ``class_assertion``
        ``[a, C]``: the individual ``a`` is an answer of ``C``.
        """
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
