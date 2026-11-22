import copy
import logging
import os

import numpy as np
import torch as th
from deprecated.sphinx import versionadded
from pykeen.models import ERModel
from pykeen.nn.init import PretrainedInitializer

import mowl.error.messages as msg
from mowl.models.graph_kge.triple_model import EvaluationModel, TripleGraphModel  # noqa: F401
from mowl.projection import Edge

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
    """

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
        Train the model on the triples of the projection.

        :param epochs: Number of epochs to train the model.
        :type epochs: int
        """
        if self._kge_method is None:
            raise AttributeError(msg.PYKEEN_MODEL_NOT_SET)
        self._check_training_setup()
        self._kge_method.train()
        return super().train(epochs=epochs)

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

        new_class_embeds = []
        for new_node, new_id in self.graph_node_to_id.items():
            if new_node in prev_class_embeds:
                new_class_embeds.append(prev_class_embeds[new_node])
            elif new_node in prev_individual_embeds:
                new_class_embeds.append(prev_individual_embeds[new_node])
            else:
                class_size = self._kge_method.entity_representations[0](indices=None).shape[1]
                new_class_embeds.append(np.random.normal(size=class_size))

        new_class_embeds = np.asarray(new_class_embeds)

        new_object_property_embeds = []
        for new_relation, new_id in self.graph_relation_to_id.items():
            if new_relation in prev_object_property_embeds:
                new_object_property_embeds.append(prev_object_property_embeds[new_relation])
            else:
                op_size = self._kge_method.relation_representations[0](indices=None).shape[1]
                new_object_property_embeds.append(np.random.normal(size=op_size))

        new_object_property_embeds = np.asarray(new_object_property_embeds)

        pretrained_cls_embeddings = th.tensor(new_class_embeds)
        pretrained_op_embeddings = th.tensor(new_object_property_embeds)

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
