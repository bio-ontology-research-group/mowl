"""Query embedding models (Query2Box, BetaE, ConE, AConE) on ontologies.

One model class for the four methods: they differ in the geometry of the operators
(:mod:`mowl.nn.query.geometry`), which a subclass names in ``geometry_class``. The graph
they are trained on, the sampling of training queries, the loss, the class expression to
query translation (:mod:`mowl.baselines.models.query.expression`), the scoring of axioms for the
rank-based regimes and ``score_query`` for
:class:`mowl.evaluation.LogicalQueryAnswering` are shared.
"""

import math

import numpy as np
import torch as th
import torch.nn.functional as F

from mowl.base_models.model import Model
from mowl.baselines.models.query import sampling
from mowl.baselines.models.query.graph import ISA_INV, extract_graph, role_relation
from mowl.baselines.models.query.expression import (Vocabulary, class_expression_to_query, from_nested,
                                      ids, signature, to_dnf)
from mowl.baselines.nn.query.base import UnsupportedOperator

from mowl.baselines.training.regime import SCHEDULE_HOOKS  # noqa: E402


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

    **Training** minimises ``-log σ(logit(answer)) - mean log σ(-logit(negative))`` (all
    four reference codes) with uniformly drawn negative entities. Where the queries come
    from is the ``queries`` argument (a training regime in the sense of mOWL: the model does
    not depend on it):

    * ``"sampled"`` (default): queries of the structures 1p, 2p, 3p, 2i, 3i, ip, pi (and the
      negation structures 2in, 3in, inp, pin, pni for geometries with negation) are sampled
      from the graph, backwards from a random answer, in every step.
    * ``"released"``: the training queries of a released query dataset (the dataset must
      have ``training_queries(entity_id, relation_id, names)``, e.g.
      :class:`~mowl.datasets.builtin.query_sets.QueryAnsweringDataset`), as in the four
      papers. Every update draws ``batch_size`` queries uniformly from the pool of
      the path structures (1p, 2p, 3p) or of the other structures, so structures
      are drawn in proportion to their number in the release; the positive is a random
      answer on the training graph, the negatives are uniform non-answers, and every query
      is weighted by ``1/sqrt(number of answers)``. One step is an update on a batch of path
      queries, then one on a batch of the other queries, then again one of path queries
      (a single update if there are no other queries), the loop of the reference code
      (KGReasoning ``main.py``, which ConE and AConE copy).

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
                 seed=0, queries="sampled", steps_per_epoch=10, batch_per_structure=False,
                 model_filepath=None, device="cpu", **geometry_kwargs):
        super().__init__(dataset, model_filepath=model_filepath)
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
        self.loss_history = []
        if queries == "released":
            self._load_released(dataset, templates, structures)
            self._pool, self.training_structures = [], {}
            return
        pool = self.graph.entities_with_incoming()
        self._pool = pool
        self.training_structures = (sampling.usable_structures(templates, self.graph,
                                                               self.rng, pool)
                                    if pool else {})

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
    def _batch(self):
        names = list(self.training_structures)
        per = (self.batch_size if self.batch_per_structure
               else max(1, self.batch_size // len(names)))
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

    def _load_released(self, dataset, templates, structures):
        """The released training queries, grouped as path and other structures."""
        if not hasattr(dataset, "training_queries"):
            raise ValueError("queries='released' needs a dataset with training_queries(), "
                             "e.g. mowl.datasets.builtin.query_sets.QueryAnsweringDataset.")
        pool = dataset.training_queries(self.entity_id, self.relation_id)
        names = [n for n in pool if n in templates]
        if structures is not None:
            names = [n for n in names if n in structures]
        if not names:
            raise ValueError(f"None of the released training structures {sorted(pool)} can "
                             f"be trained by this geometry (it trains {sorted(templates)}).")
        self.released = {n: pool[n] for n in names}
        self.released_templates = {n: templates[n] for n in names}
        self._released_columns = {}
        from mowl.datasets.builtin.query_sets import _kinds
        for n, q in self.released.items():
            kinds = _kinds(q.structure)
            self._released_columns[n] = [i for i, k in enumerate(kinds) if k in ("e", "r")]
        path = [n for n in names if n in ("1p", "2p", "3p")]
        other = [n for n in names if n not in path]
        self._released_groups = {"path": path, "other": other}

    def _released_samples(self, q, local):
        """Positives, negatives and weights for the queries ``local`` of a
        :class:`~mowl.datasets.builtin.query_sets.ReleasedTrainingQueries` (vectorised): a
        uniformly drawn answer, ``negatives`` uniform entities that are not answers (the
        ones that are redrawn up to three times), and ``1/sqrt(number of answers)``."""
        n_entities = self.graph.num_entities
        counts = q.offsets[local + 1] - q.offsets[local]
        starts = q.offsets[local]
        pick = starts + (self.rng.random(len(local)) * counts).astype(np.int64)
        positives = q.answers[pick]
        total = int(counts.sum())
        owner = np.repeat(np.arange(len(local)), counts)
        within = np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
        keys = owner * n_entities + q.answers[np.repeat(starts, counts) + within]
        is_answer = np.zeros(len(local) * n_entities, dtype=bool)
        is_answer[keys] = True
        rows = np.arange(len(local))[:, None]
        negatives = self.rng.integers(n_entities, size=(len(local), self.negatives))
        for _ in range(3):
            bad = is_answer[rows * n_entities + negatives]
            if not bad.any():
                break
            negatives[bad] = self.rng.integers(n_entities, size=int(bad.sum()))
        return positives, negatives, 1.0 / np.sqrt(counts)

    def _released_batch(self, group):
        """``batch_size`` training queries drawn uniformly from the structures of a group
        (``"path"`` or ``"other"``), as ``{name: (signature, rows, positives, negatives,
        weights)}``."""
        names = self._released_groups[group]
        sizes = np.array([len(self.released[n]) for n in names])
        draw = self.rng.integers(int(sizes.sum()), size=self.batch_size)
        bounds = np.concatenate([[0], np.cumsum(sizes)])
        which = np.searchsorted(bounds, draw, side="right") - 1
        out = {}
        dev = self.device
        for j, name in enumerate(names):
            local = draw[which == j] - bounds[j]
            if len(local) == 0:
                continue
            q = self.released[name]
            rows = q.rows[local][:, self._released_columns[name]]
            pos, negs, weights = self._released_samples(q, local)
            out[name] = (signature(self.released_templates[name]),
                         th.tensor(rows, dtype=th.long, device=dev),
                         th.tensor(pos, dtype=th.long, device=dev).unsqueeze(1),
                         th.tensor(negs, dtype=th.long, device=dev),
                         th.tensor(weights, dtype=th.float, device=dev))
        return out

    def released_loss(self, groups):
        """The negative sampling loss of a batch from :meth:`_released_batch`, each query
        weighted by ``1/sqrt(number of answers)`` and the weights normalised over the batch
        (as in the reference code)."""
        pos_loss, neg_loss, total = 0.0, 0.0, 0.0
        for sig, rows, pos, neg, w in groups.values():
            state = self.module.embed(sig, rows)
            p = self.module.logit(state, pos).squeeze(1)
            n = self.module.logit(state, neg)
            pos_loss = pos_loss - (w * F.logsigmoid(p)).sum()
            neg_loss = neg_loss - (w * F.logsigmoid(-n).mean(dim=1)).sum()
            total = total + w.sum()
        return (pos_loss / total + neg_loss / total) / 2

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

    def train(self, epochs=10, steps_per_epoch=None, **kwargs):
        """Train with Adam. One epoch is ``steps_per_epoch`` steps (default: the value of
        the constructor), each on ``batch_size`` queries (sampled afresh or drawn from the
        released training queries, see the class). The mean loss per epoch is appended to
        :attr:`loss_history`. If ``model_filepath`` was given the parameters are saved.

        :raises ValueError: If no query structure can be sampled from the graph (an
            ontology without subclass or role axioms).
        """
        steps_per_epoch = self.steps_per_epoch if steps_per_epoch is None else steps_per_epoch
        released = self.query_source == "released"
        if not released and not self.training_structures:
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
            total, updates = 0.0, 0
            for _ in range(steps_per_epoch):
                if released:
                    pattern = (["path", "other", "path"] if self._released_groups["other"]
                               and self._released_groups["path"]
                               else [g for g in ("path", "other") if self._released_groups[g]])
                    batches = [(g, self._released_batch(g)) for g in pattern]
                else:
                    batches = [(None, None)]
                for group, batch in batches:
                    loss = (self.released_loss(batch) if released
                            else self.loss(self._batch()))
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
        """The query (:class:`~mowl.baselines.models.query.expression.Node`) of an OWL API class
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

        :param structure: Nested-tuple structure (:func:`mowl.baselines.models.query.expression.from_nested`).
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
        from mowl.baselines.models.query.expression import contains_negation
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
        from mowl.baselines.models.query.expression import anchor, project
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
        from mowl.baselines.models.query.expression import project
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
        from mowl.baselines.models.query.expression import anchor, project
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
