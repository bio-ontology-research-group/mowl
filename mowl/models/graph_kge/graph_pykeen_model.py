import copy
import logging
import os

import torch as th
from deprecated.sphinx import versionadded
from pykeen.models import ERModel
from pykeen.nn.init import PretrainedInitializer

import mowl.error.messages as msg
from mowl.models.graph_kge.triple_model import EvaluationModel, TripleGraphModel  # noqa: F401
from mowl.projection import Edge
from mowl.training.triples import extend_embeddings
from mowl.base_models import KGEModel  # noqa: F401  (importable from here in 2.1.0)

logger = logging.getLogger(__name__)


@versionadded(version="0.2.0")
class GraphPlusPyKEENModel(TripleGraphModel):
    """
    A model that puts a projection of the ontology and any PyKEEN knowledge graph embedding
    together (:class:`pykeen.models.ERModel`): the projector gives the triples, the PyKEEN
    model gives the entity and relation representations, the interaction function and its
    loss, and the model is trained on the triples as every model of
    :class:`~mowl.models.graph_kge.triple_model.TripleGraphModel` is. :attr:`module` is the
    geometry that reads the PyKEEN model
    (:class:`~mowl.representation.library.other_graph.PyKEENGeometry`):
    its representations are the parts, its interaction the score, its loss the objective;
    the sampler is :class:`~mowl.representation.library.other_graph.UniformCorruption` with
    ``num_negs_per_pos`` corrupted triples. Use ``projector=mowl.projection.GDAProjector()``
    for the GDA projection.

    Training (:meth:`train`) is by default the run of the released class (mOWL 2.1.0): that
    of PyKEEN's sLCWA training loop, which, at every call, probes the memory and draws every
    parameter again before the first epoch, then trains on PyKEEN's shuffled batches with the
    negatives of its basic negative sampler (regime
    :class:`~mowl.training.triples.SLCWABatches` in mOWL's shared loop). Two attributes change
    that:

    * :attr:`reinitialise` (default ``True``): ``False`` continues from the current
      parameters, so a second :meth:`train` goes on where the first stopped and pretrained
      or loaded parameters are kept;
    * :attr:`batching` (default ``"pykeen"``): ``"mowl"`` uses mOWL's triple batches and its
      sampler (:class:`~mowl.training.triples.TripleBatches`,
      :class:`~mowl.representation.library.other_graph.UniformCorruption`), which a
      :attr:`training_regime` needs to replace the negatives, the loss or the
      regularisation.
    """

    #: Draw every parameter again at the start of each :meth:`train`, as PyKEEN's loop and
    #: the released class do (``False``: continue from the current parameters).
    reinitialise = True
    #: ``"pykeen"``: the batches, negatives and loss of PyKEEN's sLCWA loop (the released
    #: run); ``"mowl"``: mOWL's triple batches and uniform corruption.
    batching = "pykeen"
    #: The hooks a regime may override with ``batching="pykeen"`` (PyKEEN's loop makes the
    #: negatives, the loss and the regularisation itself).
    _DATA_HOOKS = frozenset(("negatives", "filter_negatives", "loss", "regularization"))

    def _ready(self):
        return self._kge_method is not None

    def _make_module(self):
        if self._kge_method is None:
            raise AttributeError(msg.PYKEEN_MODEL_NOT_SET)
        from mowl.representation.library.other_graph import PyKEENGeometry
        return PyKEENGeometry(self)

    def _entity_table(self):
        return lambda idxs: self._kge_method.entity_representations[0](indices=idxs)

    def _relation_table(self):
        return lambda idxs: self._kge_method.relation_representations[0](indices=idxs)

    def set_kge_method(self, kge_method, *args, **kwargs):
        """
        Set the KGE method of the model.

        :param kge_method: The KGE method.
        :type kge_method: :class:`pykeen.models.ERModel`
        """
        try:
            self._kge_method_uninitialized = kge_method
            initialized_kge_method = kge_method(*args, triples_factory=self.triples_factory,
                                                **kwargs)
        except TypeError:
            raise TypeError(f"Parameter 'kge_method' must be a pykeen.models.ERModel object. "
                            f"Got {type(kge_method)} instead.")

        if not isinstance(initialized_kge_method, ERModel):
            raise TypeError(f"Parameter 'kge_method' must be a pykeen.models.ERModel object. "
                            f"Got {type(kge_method)} instead.")

        self._kge_method = initialized_kge_method.to(self.device)
        self._kge_method_args = args
        self._kge_method_kwargs = kwargs
        self._module = None
        self._axiom_index_maps = None

    def train(self, epochs=0):
        """
        Train the model on the triples of the projection (see the class description for
        :attr:`reinitialise` and :attr:`batching`; the defaults give the run of the released
        class).

        :param epochs: Number of epochs to train the model.
        :type epochs: int
        """
        if self._kge_method is None:
            raise AttributeError(msg.PYKEEN_MODEL_NOT_SET)
        self._check_training_setup()
        if self.batching not in ("pykeen", "mowl"):
            raise ValueError("batching must be 'pykeen' or 'mowl'.")
        self._kge_method.train()
        if self.batching == "mowl":
            return super().train(epochs=epochs)
        from mowl.training.triples import SLCWABatches, train_on_triples
        regime = self.training_regime
        changed = (regime.overridden_hooks() & self._DATA_HOOKS) if regime is not None \
            else frozenset()
        if changed:
            raise TypeError(
                f"{type(regime).__name__} changes {sorted(changed)}, which PyKEEN's sLCWA "
                f"loop (batching='pykeen') makes itself; set model.batching = 'mowl' to "
                f"train with mOWL's batches and sampler, which consult these hooks.")
        factory = self.triples_factory
        sampler_options = None if self.num_negs_per_pos is None else \
            {"num_negs_per_pos": int(self.num_negs_per_pos)}
        batches = SLCWABatches(self._kge_method, factory, self.batch_size,
                               negative_sampler_kwargs=sampler_options,
                               drop_last=True if self.drop_last else None,
                               reinitialise=self.reinitialise)
        train_on_triples(self.module, self.training_triples(), factory.num_entities,
                         factory.num_relations, self.optimizer, self.lr, self.batch_size,
                         epochs, regime, self.device, batches=batches)
        self._save()

    def _save(self):
        th.save(self._kge_method, self.model_filepath)

    def add_axioms(self, *axioms):
        prev_class_embeds = copy.deepcopy(self.class_embeddings)
        prev_object_property_embeds = copy.deepcopy(self.object_property_embeddings)
        prev_individual_embeds = copy.deepcopy(self.individual_embeddings)
        print(f"Number of classes before adding axioms: {len(prev_class_embeds)}")
        print(f"Number of object properties before adding axioms: "
              f"{len(prev_object_property_embeds)}")
        print(f"Number of individuals before adding axioms: {len(prev_individual_embeds)}")

        self.dataset.add_axioms(*axioms)
        self._load_edges()
        self._triples_factory = Edge.as_pykeen(
            self.edges, entity_to_id=self.graph_node_to_id,
            relation_to_id=self.graph_relation_to_id,
            create_inverse_triples=self.create_inverse_triples)

        class_size = self._kge_method.entity_representations[0](indices=None).shape[1]
        op_size = self._kge_method.relation_representations[0](indices=None).shape[1]
        pretrained_cls_embeddings = extend_embeddings(
            [prev_class_embeds, prev_individual_embeds], self.graph_node_to_id, class_size)
        pretrained_op_embeddings = extend_embeddings(
            prev_object_property_embeds, self.graph_relation_to_id, op_size)

        new_kge_method = self._kge_method_uninitialized(
            triples_factory=self.triples_factory,
            entity_initializer=PretrainedInitializer(tensor=pretrained_cls_embeddings),
            relation_initializer=PretrainedInitializer(tensor=pretrained_op_embeddings),
            *self._kge_method_args, **self._kge_method_kwargs)
        self._kge_method = new_kge_method
        self._module = None
        self._axiom_index_maps = None

    def from_pretrained(self, model):
        if not isinstance(model, str):
            raise TypeError("Parameter model must be a string pointing to the PyKEEN model "
                            "file.")

        if not os.path.exists(model):
            raise FileNotFoundError("Pretrained model path does not exist")

        self._is_pretrained = True
        self._kge_method = th.load(model, weights_only=False)
        self._module = None
        self._axiom_index_maps = None
