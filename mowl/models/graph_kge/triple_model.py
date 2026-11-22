r"""Models trained on the triples of a graph of the ontology.

A knowledge graph embedding of an ontology is a combination of parts, and this module is
where they are put together: the **projection** (:mod:`mowl.projection`) turns the ontology
into triples, the **entity model** and the **relation model** (what a relation is: a
translation, boxes, parallelograms, matrices) with the **interaction** that scores a triple
make up the geometry (:class:`~mowl.representation.library.other_graph.TripleGeometry`), the
**objective** is a loss on the scores of positive and corrupted triples, the **negative
sampler** corrupts them, and the **training regime** (batches, optimiser, hooks of
:mod:`mowl.training`) and the **evaluation regime** (:mod:`mowl.evaluation`, through
:meth:`~TripleGraphModel.axiom_costs`) are the machinery of the library.

:class:`TripleGraphModel` is the training loop and the scoring of axioms that every such model
shares. :class:`GraphKGEModel` is the general model built from the parts of a registered
method (``boxe``, ``expressive``, ``reshuffle``, ``graph_kge``, ``gda_projection``), with any
part swapped; :class:`~mowl.models.BoxE`, :class:`~mowl.models.ExpressivE` and
:class:`~mowl.models.Reshuffle` are the named combinations.
:class:`~mowl.models.GraphPlusPyKEENModel` reads the parts from any PyKEEN model.
"""

import copy
import logging
import os

import numpy as np
import torch as th

import mowl.error.messages as msg
from mowl.base_models import KGEModel
from mowl.projection import Edge
from mowl.representation.library.graph_projection import projection as make_projection
from mowl.representation.library.other_graph import UniformCorruption
from mowl.training.regime import TrainingRegime

logger = logging.getLogger(__name__)

__all__ = ["TripleGraphModel", "GraphKGEModel", "EvaluationModel", "HitsEarlyStopping"]


class EvaluationModel(th.nn.Module):
    """The adapter of the evaluators that score triples (``data`` are triples of graph ids)."""

    def __init__(self, kge_model, device):
        logger.warning("A custom EvaluationModel should be created depending on the task. "
                       "This is a generic one.")
        super().__init__()
        self.kge_model = kge_model
        self.device = device

    def forward(self, data, *args, **kwargs):
        return - self.kge_model.score_hrt(data)


class _Table:
    """A table of the geometry with the call signature of a PyKEEN representation."""

    def __init__(self, table):
        self.table = table

    def __call__(self, indices=None):
        if indices is None:
            return self.table.weight
        return self.table(indices)

    def parameters(self):
        return self.table.parameters()


class _Scorer:
    """The geometry of a model built from parts seen as a knowledge graph embedding model
    (``kge_method``): ``score_hrt`` of triples of ids, the parameters, and the tables of the
    entities and of the relations."""

    def __init__(self, module):
        self.module = module

    def score_hrt(self, triples):
        return -self.module.triple_cost(triples[:, 0], triples[:, 1], triples[:, 2])

    def parameters(self):
        return self.module.parameters()

    def zero_grad(self, set_to_none=True):
        self.module.zero_grad(set_to_none=set_to_none)

    def eval(self):
        self.module.eval()
        return self

    def train(self, mode=True):
        self.module.train(mode)
        return self

    @property
    def interaction(self):
        return self.module.measure

    @property
    def entity_representations(self):
        classes = self.module.class_model
        tables = [classes.centres] + ([classes.bumps] if getattr(classes, "bumps", None)
                                      is not None else [])
        return [_Table(t) for t in tables]

    @property
    def relation_representations(self):
        relations = self.module.relation_model
        names = ("head_base", "head_delta", "head_size", "tail_base", "tail_delta",
                 "tail_size")
        if hasattr(relations, "head_base"):
            return [_Table(getattr(relations, n)) for n in names]
        return [_Table(relations.vectors)]


class TripleGraphModel(KGEModel):
    """A model trained on the triples of the graph of an ontology, in the common interface.

    Subclasses say how the geometry is made (:meth:`_make_module`); the model reads the graph
    with its projector, trains the geometry on its triples with corrupted triples as negatives
    and scores axioms as triples (:meth:`axiom_costs`), whatever the parts are.

    Training hyper-parameters are attributes (``optimizer``, a class of :mod:`torch.optim`;
    ``lr``; ``batch_size``; ``num_negs_per_pos``) and a
    :class:`~mowl.training.TrainingRegime` can be assigned to ``training_regime``: it may
    replace the negatives (``negatives``, ``filter_negatives``), the loss (``loss``), the
    optimiser (``optimizer``), regularise (``regularization``), project (``after_step``), and
    act at the start and end of an epoch, stop (``should_stop``) and finish
    (``on_train_end``). The training data is the set of triples of the graph, form
    ``"triple"``.
    """

    axiom_scoring = "method"
    scored_forms = ("gci0", "gci2", "class_assertion", "object_property_assertion")
    #: Label of the edges the projectors give to ``C ⊑ D``.
    SUBCLASS_RELATION = "http://subclassof"
    #: Label of the edges the projectors give to class assertions ``C(a)``.
    TYPE_RELATION = "http://type"
    #: Whether the triples of the graph have inverse triples (relation ``r`` is then ``2r`` in
    #: the model, its inverse ``2r + 1``; the inverse triples are trained as positives and a
    #: head is predicted through them). A subclass sets it before calling the constructor.
    create_inverse_triples = False
    #: Whether :meth:`train` draws the parameters again before it starts (what PyKEEN's
    #: training loop does unless it continues a run). ``False`` (default): training continues
    #: from the current parameters, so the seed of the model decides its initial values, a
    #: second call goes on where the first stopped and pretrained parameters are kept.
    reinitialise = False
    #: The hooks of :mod:`mowl.training` the training loop consults.
    training_hooks = frozenset(("negatives", "filter_negatives", "loss", "optimizer",
                                "on_epoch_start", "on_epoch_end", "should_stop",
                                "on_train_end", "regularization", "after_step"))

    def __init__(self, *args, device="cpu", **kwargs):
        super().__init__(*args, **kwargs)
        self._triples_factory = None
        self._module = None
        self._kge_method = None
        self._axiom_index_maps = None
        self.optimizer = None
        self.lr = None
        self.batch_size = None
        self._num_negs_per_pos = None
        self.drop_last = False
        self.epochs = None
        self.device = device

    # -- the graph -----------------------------------------------------------------------
    @property
    def triples_factory(self):
        """The graph as a :class:`pykeen.triples.TriplesFactory` (ids of the projection;
        with inverse triples when :attr:`create_inverse_triples`).

        :rtype: :class:`pykeen.triples.TriplesFactory`
        """
        if self._triples_factory is not None:
            return self._triples_factory
        self._triples_factory = Edge.as_pykeen(
            self.edges, entity_to_id=self.graph_node_to_id,
            relation_to_id=self.graph_relation_to_id,
            create_inverse_triples=self.create_inverse_triples)
        self._graph_node_to_id = self._triples_factory.entity_to_id
        self._graph_relation_to_id = self._triples_factory.relation_to_id
        return self._triples_factory

    def training_triples(self):
        """The positive triples ``(E, 3)`` of the graph in model ids (with the inverse
        triples when :attr:`create_inverse_triples`: ``(h, 2r, t)`` and ``(t, 2r + 1, h)``)."""
        triples = self.triples_factory.mapped_triples
        if not self.create_inverse_triples:
            return triples
        forward = triples.clone()
        forward[:, 1] = 2 * triples[:, 1]
        backward = triples.clone()
        backward[:, 1] = 2 * triples[:, 1] + 1
        return th.cat([forward, backward.flip(1)])

    # -- the geometry --------------------------------------------------------------------
    def _make_module(self):
        raise NotImplementedError

    @property
    def module(self):
        """The geometry: a :class:`~mowl.representation.Geometry` of the parts of the model
        (``module.parts`` names them)."""
        if self._module is None:
            self._module = self._make_module()
        return self._module

    @property
    def num_negs_per_pos(self):
        """Corrupted triples per positive triple (``None``: one)."""
        return self._num_negs_per_pos

    @num_negs_per_pos.setter
    def num_negs_per_pos(self, value):
        self._num_negs_per_pos = value
        sampler = getattr(self._module, "sampler", None) if self._module is not None else None
        if isinstance(sampler, UniformCorruption):
            sampler.num_negs_per_pos = 1 if value is None else int(value)

    @property
    def evaluation_model(self):
        if self._evaluation_model is None:
            self._evaluation_model = EvaluationModel(self.kge_method, self.device)
        return self._evaluation_model

    # -- embeddings -----------------------------------------------------------------------
    def _entity_table(self):
        raise NotImplementedError

    def _relation_table(self):
        raise NotImplementedError

    def _embeddings(self, names, ids_of, table):
        if len(names) == 0:
            return dict()
        graph_id = {n: ids_of[n] for n in names if n in ids_of}
        idxs = th.tensor(list(graph_id.values()))
        vectors = table(idxs).cpu().detach().numpy()
        return dict(zip(graph_id.keys(), vectors))

    @property
    def class_embeddings(self):
        if not self._ready():
            raise AttributeError(msg.MODEL_NOT_TRAINED_OR_LOADED)
        return self._embeddings(self.dataset.classes.as_str,
                                self.triples_factory.entity_to_id, self._entity_table())

    @property
    def object_property_embeddings(self):
        if not self._ready():
            raise AttributeError(msg.MODEL_NOT_TRAINED_OR_LOADED)
        relations = self.triples_factory.relation_to_id
        return self._embeddings(list(self.graph_relation_to_id.keys()), relations,
                                self._relation_table())

    @property
    def individual_embeddings(self):
        if not self._ready():
            raise AttributeError(msg.MODEL_NOT_TRAINED_OR_LOADED)
        return self._embeddings(self.dataset.individuals.as_str,
                                self.triples_factory.entity_to_id, self._entity_table())

    def _ready(self):
        return True

    # -- scores of axioms -----------------------------------------------------------------
    def _index_maps(self):
        maps = self._axiom_index_maps
        if maps is None:
            nodes = self.triples_factory.entity_to_id
            relations = self.triples_factory.relation_to_id
            class_map = th.full((max(1, len(self.dataset.classes)),), -1, dtype=th.long)
            for iri, i in self.class_index_dict.items():
                class_map[i] = nodes.get(iri, -1)
            individual_map = th.full((max(1, len(self.dataset.individuals)),), -1,
                                     dtype=th.long)
            for iri, i in self.individual_index_dict.items():
                individual_map[i] = nodes.get(iri, -1)
            role_map = th.full((max(1, len(self.dataset.object_properties)),), -1,
                               dtype=th.long)
            for iri, i in self.object_property_index_dict.items():
                role_map[i] = relations.get(iri, -1)
            maps = self._axiom_index_maps = (class_map, role_map,
                                             relations.get(self.SUBCLASS_RELATION, -1),
                                             individual_map,
                                             relations.get(self.TYPE_RELATION, -1))
        return maps

    @th.no_grad()
    def axiom_costs(self, gci_name, rows):
        """Cost of every candidate axiom in ``rows`` (indices of the dataset's classes,
        object properties and individuals), lower is more plausible: minus the score the
        model gives to the edge the axiom is projected to.

        * ``gci0`` rows ``[C, D]``: the edge ``C -subclassof-> D``;
        * ``gci2`` rows ``[C, r, D]``: the edge ``C -r-> D``;
        * ``class_assertion`` rows ``[a, C]``: the edge ``a -type-> C``;
        * ``object_property_assertion`` rows ``[a, r, b]``: the edge ``a -r-> b``.

        An axiom whose role is not an edge label of the graph (and a class assertion when
        the projection has no type edges) gets an infinite cost. A class or individual that
        occurs in no edge is a node with an untrained vector, as everywhere in this
        model."""
        module = self.module
        if gci_name not in self.scored_forms:
            from mowl.evaluation.model_scoring import UnsupportedAxiomForm
            raise UnsupportedAxiomForm(
                f"{type(self).__name__} has no score for {gci_name!r} axioms.")
        class_map, role_map, subclass, individual_map, type_relation = self._index_maps()
        cpu = rows.cpu()
        if gci_name == "gci0":
            head, tail = class_map[cpu[:, 0]], class_map[cpu[:, 1]]
            relation = th.full_like(head, subclass)
        elif gci_name == "gci2":
            head, tail = class_map[cpu[:, 0]], class_map[cpu[:, 2]]
            relation = role_map[cpu[:, 1]]
        elif gci_name == "class_assertion":
            head, tail = individual_map[cpu[:, 0]], class_map[cpu[:, 1]]
            relation = th.full_like(head, type_relation)
        else:
            head, tail = individual_map[cpu[:, 0]], individual_map[cpu[:, 2]]
            relation = role_map[cpu[:, 1]]
        if self.create_inverse_triples:
            relation = th.where(relation >= 0, 2 * relation, relation)
        known = (head >= 0) & (tail >= 0) & (relation >= 0)
        out = th.full((len(cpu),), float("inf"))
        if known.any():
            triples = th.stack([head[known], relation[known], tail[known]], dim=1)
            module.eval()
            costs = []
            for s in range(0, len(triples), 2 ** 16):
                chunk = triples[s:s + 2 ** 16].to(self.device)
                costs.append(module.triple_cost(chunk[:, 0], chunk[:, 1], chunk[:, 2])
                             .reshape(-1).cpu())
            out[known] = th.cat(costs).float()
        return out.to(rows.device)

    # -- training -------------------------------------------------------------------------
    def _check_training_setup(self):
        if self.optimizer is None:
            raise AttributeError(msg.PYKEEN_OPTIMIZER_NOT_SET)
        if self.lr is None:
            raise AttributeError(msg.PYKEEN_LR_NOT_SET)
        if self.batch_size is None:
            raise AttributeError(msg.PYKEEN_BATCH_SIZE_NOT_SET)

    def train(self, epochs=0):
        """Train on the triples of the graph.

        Every epoch visits the triples in random batches; the corrupted triples of a batch
        (the sampler of the geometry, or the regime's ``negatives``) are scored with the same
        triple score, the objective of the geometry (or the regime's ``loss``) turns the scores
        into the loss, plus the regularisation of the geometry; after the optimiser step the
        geometry's constraints are applied (``module.project()``).

        :param epochs: Number of epochs to train the model.
        :type epochs: int
        """
        module = self.module
        self._check_training_setup()
        regime = self.training_regime
        factory = self.triples_factory
        triples = self.training_triples()
        num_entities, num_relations = factory.num_entities, factory.num_relations

        if self.reinitialise:
            module.reset_parameters()
        optimizer = NotImplemented
        if regime is not None:
            optimizer = regime.optimizer()
        if optimizer is NotImplemented:
            optimizer = self.optimizer(params=module.trainable_parameters(), lr=self.lr)

        def sample(positives):
            negatives = NotImplemented
            if regime is not None:
                negatives = regime.negatives("triple", positives)
            if negatives is NotImplemented:
                negatives = module.sampler(positives, num_entities, num_relations)
            if regime is not None:
                negatives = regime.filter_negatives("triple", positives, negatives)
            return negatives

        batches = th.utils.data.DataLoader(
            dataset=_batches(triples, self.batch_size, self.drop_last, sample),
            batch_size=None, batch_sampler=None)
        if len(batches) == 0:
            raise ValueError("There is no training batch: the graph has no triples.")
        for epoch in range(1, 1 + epochs):
            module.train()
            if regime is not None:
                regime.on_epoch_start(epoch - 1, epochs)
            total = 0.0
            for positives, negatives in batches:
                optimizer.zero_grad(set_to_none=True)
                positives = positives.to(self.device)
                negatives = negatives.to(self.device)
                shape = negatives.shape[:-1]
                positive_scores = -module.triple_cost(positives[:, 0], positives[:, 1],
                                                      positives[:, 2])
                flat = negatives.reshape(-1, 3)
                negative_scores = -module.triple_cost(flat[:, 0], flat[:, 1],
                                                      flat[:, 2]).reshape(*shape)
                loss = NotImplemented
                if regime is not None:
                    loss = regime.loss("triple", positive_scores, negative_scores)
                if loss is NotImplemented:
                    loss = module.loss(positive_scores, negative_scores)
                loss = loss + self._regularization(module, regime)
                if not th.isfinite(loss):
                    raise FloatingPointError("The loss is not finite.")
                loss.backward()
                optimizer.step()
                module.project()
                if regime is not None:
                    regime.after_step()
                total += loss.item()
            if regime is not None:
                regime.on_epoch_end(epoch - 1, {"total": total})
                if regime.should_stop(epoch - 1):
                    break
        if regime is not None:
            regime.on_train_end()
        self._save()

    @staticmethod
    def _regularization(module, regime):
        term = module.regularization_loss()
        if regime is not None:
            out = regime.regularization(term)
            if out is not NotImplemented:
                term = out
        return term

    def _save(self):
        th.save(self.module.state_dict(), self.model_filepath)

    # -- validation -----------------------------------------------------------------------
    def validation_triples(self):
        """The triples of the validation ontology that lie over the training graph, ``(V, 3)``
        in the ids of the graph (forward relations)."""
        if self.dataset.validation is None:
            raise ValueError("The dataset has no validation ontology.")
        nodes, relations = self.graph_node_to_id, self.graph_relation_to_id
        edges = [e for e in self.projector.project(self.dataset.validation)
                 if e.src in nodes and e.dst in nodes and e.rel in relations]
        if not edges:
            raise ValueError("The validation ontology gives no edge over the training graph.")
        return th.tensor([[nodes[e.src], relations[e.rel], nodes[e.dst]] for e in edges])

    # -- changing the ontology ------------------------------------------------------------
    def add_axioms(self, *axioms):
        """Add axioms to the ontology and rebuild the graph and the geometry. The rows of the
        tables of the nodes and the relations that were in the graph before keep their
        values, new ones are initialised as at the start."""
        old = self.module
        old_nodes = dict(self.triples_factory.entity_to_id)
        old_relations = dict(self.triples_factory.relation_to_id)
        self.dataset.add_axioms(*axioms)
        self._load_edges()
        self._triples_factory = None
        self._axiom_index_maps = None
        self._module = None
        new = self.module
        nodes, relations = self.triples_factory.entity_to_id, \
            self.triples_factory.relation_to_id
        for old_part, new_part, before, after in (
                (old.class_model, new.class_model, old_nodes, nodes),
                (old.relation_model, new.relation_model, old_relations, relations)):
            if old_part is None or new_part is None:
                continue
            _carry(old_part, new_part, before, after, self.create_inverse_triples
                   and old_part is old.relation_model)
        return self

    def from_pretrained(self, model):
        """Load the parameters saved by :meth:`train` (the file at ``model``)."""
        if not isinstance(model, str):
            raise TypeError("Parameter model must be a string pointing to the model file.")
        if not os.path.exists(model):
            raise FileNotFoundError("Pretrained model path does not exist")
        self._is_pretrained = True
        self.module.load_state_dict(th.load(model, weights_only=False))


def _batches(triples, batch_size, drop_last, sample):
    from mowl.representation.library.other_graph import _TripleBatches
    return _TripleBatches(triples, batch_size, drop_last, sample)


def _carry(old, new, before, after, doubled):
    """Copy the rows of the tables of the part ``old`` into ``new`` for the keys they share
    (``before``, ``after``: key -> row; ``doubled``: the rows of a relation are ``2 i`` and
    ``2 i + 1``)."""
    new_params = dict(new.named_parameters())
    for name, parameter in old.named_parameters():
        target = new_params.get(name)
        if target is None or parameter.shape[1:] != target.shape[1:]:
            continue
        if not parameter.requires_grad:
            continue
        with th.no_grad():
            for key, i in before.items():
                j = after.get(key)
                if j is None:
                    continue
                if doubled:
                    for d in (0, 1):
                        if 2 * i + d < len(parameter) and 2 * j + d < len(target):
                            target[2 * j + d] = parameter[2 * i + d]
                elif i < len(parameter) and j < len(target):
                    target[j] = parameter[i]


class GraphKGEModel(TripleGraphModel):
    """A knowledge graph embedding of a projection, built from the parts of a registered
    method (:data:`mowl.representation.METHODS`).

    ``GraphKGEModel(dataset, method="boxe")`` is BoxE and ``method="expressive"`` ExpressivE;
    ``method="graph_kge"`` (the default: TransE) and ``method="gda_projection"`` take any
    relation model, and every part of a method can be swapped by naming it:

    :param dataset: mOWL dataset.
    :param method: Key of the method (see above).
    :param embed_dim: Entity dimension.
    :param projector: The projection: a projector of :mod:`mowl.projection`, its class, or a
        name of :data:`mowl.representation.library.graph_projection.PROJECTIONS`. Default:
        that of the method (``"owl2vec_star"``; ``"gda"`` for ``gda_projection``).
    :param learning_rate: Learning rate of the optimiser.
    :param batch_size: Training batch size.
    :param num_negs_per_pos: Corrupted triples per positive triple (default: 1).
    :param seed: Random seed of the initial tables.
    :param optimizer: A class of :mod:`torch.optim` (Adam).
    :param classes, relations: Class model and relation model, by name or function, to swap
        the ones of the method.
    :param loss: A :class:`~mowl.representation.library.other_graph.TripleLoss` or its name.
    :param sampler: The negative sampler (default: uniform corruption of heads and tails).
    :param regularizers: Penalties (:class:`mowl.training.Regularizer`) of the geometry.
    :param reinitialise: Draw the parameters again at the start of every call of
        :meth:`train` (the behaviour of PyKEEN's training loop; default: continue from the
        current parameters).
    :param method_options: Options of the method's builder (``margin``, ``p``, ``tanh_map``,
        ``min_denom``, ...).
    """

    #: The key of the method; subclasses name theirs.
    METHOD = "graph_kge"

    def __init__(self, dataset, method=None, embed_dim=50, projector=None,
                 learning_rate=1e-3, batch_size=1024, num_negs_per_pos=None, seed=0,
                 model_filepath=None, device="cpu", optimizer=None, classes=None,
                 relations=None, loss=None, sampler=None, regularizers=(),
                 reinitialise=False, **method_options):
        super().__init__(dataset, model_filepath=model_filepath, device=device)
        self.reinitialise = bool(reinitialise)
        from mowl.representation import methods as M
        method = method or self.METHOD
        if method not in M.METHODS:
            raise ValueError(f"Unknown method {method!r}; available: {sorted(M.METHODS)}.")
        self.method = method
        self.embed_dim = embed_dim
        self.seed = seed
        self.classes, self.relations = classes, relations
        self.loss, self.sampler = loss, sampler
        self.regularizers = tuple(regularizers)
        self.method_options = method_options
        self.optimizer = optimizer or th.optim.Adam
        self.lr = learning_rate
        self.batch_size = batch_size
        self._num_negs_per_pos = num_negs_per_pos
        default = "gda" if method == "gda_projection" else "owl2vec_star"
        self.set_projector(make_projection(projector, default))
        self._make_module_now()

    def _make_module_now(self):
        self._module = None
        self._axiom_index_maps = None
        self._module = self._make_module()

    def _graph_arguments(self):
        """What the builder of the method needs of the graph."""
        factory = self.triples_factory
        subclass = factory.relation_to_id.get(self.SUBCLASS_RELATION, -1)
        if self.create_inverse_triples and subclass >= 0:
            subclass *= 2
        return dict(nb_classes=factory.num_entities, nb_relations=factory.num_relations,
                    subclass_id=subclass)

    def _make_module(self):
        import random
        from mowl.representation import methods as M
        build = M.METHODS[self.method]
        arguments = self._graph_arguments()
        options = dict(self.method_options)
        sampler = self.sampler
        if sampler is None:
            sampler = UniformCorruption(self._num_negs_per_pos)
        if self.classes is not None:
            options["classes"] = self.classes
        if self.relations is not None:
            options["relations"] = self.relations
        if self.loss is not None:
            options["loss"] = self.loss
        if self.regularizers:
            options["regularizers"] = self.regularizers
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            th.manual_seed(self.seed)
        module = build(arguments.pop("nb_classes"), arguments.pop("nb_relations"), 0,
                       sampler=sampler, **self._embedding_options(), **arguments, **options)
        return module.to(self.device)

    def _embedding_options(self):
        return dict(embed_dim=self.embed_dim)

    def _entity_table(self):
        return self.module.class_model.centres

    def _relation_table(self):
        return self.module.relation_model.vectors

    @property
    def kge_method(self):
        """The model as a knowledge graph embedding (``score_hrt`` of triples of ids, the
        parameters, the tables of entities and relations), read from the geometry."""
        return _Scorer(self.module)

    def set_kge_method(self, *args, **kwargs):
        raise TypeError(f"{type(self).__name__} is built from the parts of the method "
                        f"{self.method!r}; swap a part with the arguments of the constructor "
                        f"(classes, relations, loss, sampler, regularizers).")


class HitsEarlyStopping(TrainingRegime):
    """Early stopping on the Hits@k of the validation triples: the evaluation regime of the
    RESHUFFLE experiments (every ``frequency`` epochs, stop when ``patience`` evaluations in a
    row did not improve the best Hits@k by ``relative_delta``, then restore the best
    parameters). Hits@k is the share of validation triples, over both sides, whose true
    entity ranks among the first ``k`` of all entities (filtered: the other known triples of
    the graph are not counted; ties share their ranks).

    The validation triples are those of the validation ontology of the dataset projected like
    the training ontology (:meth:`TripleGraphModel.validation_triples`).
    """

    key = ""
    #: Attribute of the regime read by a model's loop: no hooks other than these are changed.

    def __init__(self, frequency=10, patience=10, relative_delta=0.01, k=10, batch_size=64):
        super().__init__()
        self.frequency, self.patience = int(frequency), int(patience)
        self.relative_delta, self.k, self.batch_size = float(relative_delta), int(k), batch_size
        self.history = []
        self._best, self._best_state, self._bad, self._val = None, None, 0, None

    def attach(self, model):
        super().attach(model)
        self.history, self._best, self._best_state, self._bad = [], None, None, 0
        self._val = None

    def _ranks(self, module, queries, relations, candidates_are, truth, known, n):
        """Rank of ``truth`` among all entities as the ``candidates_are`` ("tail" or
        "head") of the queries."""
        ranks = []
        everyone = th.arange(n)
        for start in range(0, len(queries), self.batch_size):
            q = queries[start:start + self.batch_size]
            r = relations[start:start + self.batch_size]
            t = truth[start:start + self.batch_size]
            b = len(q)
            repeated = q.repeat_interleave(n)
            rels = r.repeat_interleave(n)
            cands = everyone.repeat(b)
            cost = module.triple_cost(repeated, rels, cands).reshape(b, n)
            true_cost = cost[th.arange(b), t].clone()
            for row in range(b):
                for other in known.get((int(q[row]), int(r[row])), ()):
                    if other != int(t[row]):
                        cost[row, other] = float("inf")
            optimistic = (cost < true_cost[:, None]).sum(dim=1) + 1
            pessimistic = (cost <= true_cost[:, None]).sum(dim=1)
            ranks.append((optimistic + pessimistic) / 2.0)
        return th.cat(ranks)

    @th.no_grad()
    def hits_at_k(self):
        """Hits@k of the validation triples, filtered, both sides, all entities."""
        model = self.model
        module, factory = model.module, model.triples_factory
        n = factory.num_entities
        if self._val is None:
            self._val = model.validation_triples()
        val, inverse = self._val, model.create_inverse_triples
        forward = 2 * val[:, 1] if inverse else val[:, 1]
        backward = 2 * val[:, 1] + 1 if inverse else val[:, 1]
        tails, heads = {}, {}
        for h, r, t in th.cat([factory.mapped_triples, val]).tolist():
            tails.setdefault((h, forward_id(r, inverse)), set()).add(t)
            if inverse:
                heads.setdefault((t, 2 * r + 1), set()).add(h)
            else:
                heads.setdefault((t, r), set()).add(h)
        module.eval()
        with module.encoding():
            tail_ranks = self._ranks(module, val[:, 0], forward, "tail", val[:, 2], tails, n)
            if inverse:
                head_ranks = self._ranks(module, val[:, 2], backward, "tail", val[:, 0],
                                         heads, n)
            else:
                head_ranks = self._head_ranks(module, val, n, heads)
        ranks = th.cat([tail_ranks, head_ranks])
        return float((ranks <= self.k).float().mean())

    def _head_ranks(self, module, val, n, known):
        ranks = []
        everyone = th.arange(n)
        for start in range(0, len(val), self.batch_size):
            v = val[start:start + self.batch_size]
            b = len(v)
            cost = module.triple_cost(everyone.repeat(b), v[:, 1].repeat_interleave(n),
                                      v[:, 2].repeat_interleave(n)).reshape(b, n)
            true_cost = cost[th.arange(b), v[:, 0]].clone()
            for row in range(b):
                for other in known.get((int(v[row, 2]), int(v[row, 1])), ()):
                    if other != int(v[row, 0]):
                        cost[row, other] = float("inf")
            optimistic = (cost < true_cost[:, None]).sum(dim=1) + 1
            pessimistic = (cost <= true_cost[:, None]).sum(dim=1)
            ranks.append((optimistic + pessimistic) / 2.0)
        return th.cat(ranks)

    def should_stop(self, epoch):
        if (epoch + 1) % self.frequency:
            return False
        hits = self.hits_at_k()
        self.history.append(hits)
        if self._best is None or hits > self._best * (1.0 + self.relative_delta):
            self._best, self._bad = hits, 0
            self._best_state = copy.deepcopy(self.model.module.state_dict())
        else:
            self._bad += 1
        return self._bad >= self.patience

    def on_train_end(self):
        if self._best_state is not None:
            self.model.module.load_state_dict(self._best_state)


def forward_id(relation, inverse):
    """The model id of the forward relation ``relation`` of the graph."""
    return 2 * relation if inverse else relation
