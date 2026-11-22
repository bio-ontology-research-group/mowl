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

import logging
import os

import torch as th

import mowl.error.messages as msg
from mowl.base_models import KGEModel
from mowl.projection import Edge
from mowl.representation.library.graph_projection import projection as make_projection
from mowl.representation.library.other_graph import UniformCorruption
from mowl.evaluation.triple_axioms import AxiomTriples
from mowl.training.triples import (HitsEarlyStopping, carry_rows,  # noqa: F401
                                   forward_id, positive_triples, train_on_triples)
from mowl.utils.random import seed_generators

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
        triples when :attr:`create_inverse_triples`: ``(h, 2r, t)`` and ``(t, 2r + 1, h)``;
        :func:`mowl.training.triples.positive_triples`)."""
        return positive_triples(self.triples_factory.mapped_triples,
                                self.create_inverse_triples)

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
    def _axioms(self):
        """The evaluation part that reads axioms as triples
        (:class:`~mowl.evaluation.triple_axioms.AxiomTriples`)."""
        axioms = self._axiom_index_maps
        if axioms is None:
            factory = self.triples_factory
            dataset = self.dataset
            axioms = self._axiom_index_maps = AxiomTriples(
                self.class_index_dict, self.individual_index_dict,
                self.object_property_index_dict,
                (len(dataset.classes), len(dataset.individuals),
                 len(dataset.object_properties)),
                factory.entity_to_id, factory.relation_to_id, self.SUBCLASS_RELATION,
                self.TYPE_RELATION, self.create_inverse_triples)
        return axioms

    def axiom_costs(self, gci_name, rows):
        """Cost of every candidate axiom in ``rows`` (indices of the dataset's classes,
        object properties and individuals), lower is more plausible: minus the score the
        model gives to the edge the axiom is projected to (see
        :class:`~mowl.evaluation.triple_axioms.AxiomTriples`: ``gci0`` is the edge
        ``subclassof``, ``gci2`` the edge ``C -r-> D``, a class assertion the ``type`` edge,
        a role assertion its edge).

        An axiom whose role is not an edge label of the graph (and a class assertion when
        the projection has no type edges) gets an infinite cost."""
        if gci_name not in self.scored_forms:
            from mowl.evaluation.model_scoring import UnsupportedAxiomForm
            raise UnsupportedAxiomForm(
                f"{type(self).__name__} has no score for {gci_name!r} axioms.")
        return self._axioms().costs(self.module, gci_name, rows, self.device)

    # -- training -------------------------------------------------------------------------
    def _check_training_setup(self):
        if self.optimizer is None:
            raise AttributeError(msg.PYKEEN_OPTIMIZER_NOT_SET)
        if self.lr is None:
            raise AttributeError(msg.PYKEEN_LR_NOT_SET)
        if self.batch_size is None:
            raise AttributeError(msg.PYKEEN_BATCH_SIZE_NOT_SET)

    def train(self, epochs=0):
        """Train on the triples of the graph (:func:`mowl.training.triples.train_on_triples`):
        shuffled batches, the corrupted triples of the sampler of the geometry (or the
        regime's ``negatives``), the objective of the geometry (or the regime's ``loss``),
        its regularisation, and its constraints after every step.

        :param epochs: Number of epochs to train the model.
        :type epochs: int
        """
        self._check_training_setup()
        factory = self.triples_factory
        train_on_triples(self.module, self.training_triples(), factory.num_entities,
                         factory.num_relations, self.optimizer, self.lr, self.batch_size,
                         epochs, self.training_regime, self.device, self.drop_last,
                         self.reinitialise)
        self._save()

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
            doubled = self.create_inverse_triples and old_part is old.relation_model
            carry_rows(old_part, new_part, before, after, doubled)
        return self

    def from_pretrained(self, model):
        """Load the parameters saved by :meth:`train` (the file at ``model``)."""
        if not isinstance(model, str):
            raise TypeError("Parameter model must be a string pointing to the model file.")
        if not os.path.exists(model):
            raise FileNotFoundError("Pretrained model path does not exist")
        self._is_pretrained = True
        self.module.load_state_dict(th.load(model, weights_only=False))


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
            seed_generators(self.seed)
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
