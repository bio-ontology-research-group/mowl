from pykeen.models import ERModel

from mowl.base_models import KGEModel
from mowl.projection import Edge
import torch as th
import copy
import numpy as np
from pykeen.nn.init import PretrainedInitializer
import os
import mowl.error.messages as msg
from pykeen.training import SLCWATrainingLoop
from deprecated.sphinx import versionadded
import logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
logger.addHandler(handler)
logger.setLevel(logging.INFO)


@versionadded(version="0.2.0")
class GraphPlusPyKEENModel(KGEModel):
    """
    This is a wrapper class of :class:`pykeen.models.ERModel` that allows to use the PyKEEN models in the mOWL framework.
    """
    
    def __init__(self, *args, device="cpu", **kwargs):
        super().__init__(*args, **kwargs)

        self._triples_factory = None
        self._kge_method = None
        self.optimizer = None
        self.lr = None
        self.batch_size = None
        self.epochs = None
        self.device = device
                        

    @property
    def triples_factory(self):
        """
        The triples factory of the model.

        :rtype: :class:`pykeen.triples.TriplesFactory`
        """
        
        if self._triples_factory is not None:
            return self._triples_factory

        self._triples_factory = Edge.as_pykeen(self.edges, entity_to_id = self.graph_node_to_id, relation_to_id = self.graph_relation_to_id, create_inverse_triples=False)
        self._graph_node_to_id = self._triples_factory.entity_to_id
        self._graph_relation_to_id = self._triples_factory.relation_to_id
        return self._triples_factory


    @property
    def class_embeddings(self):
        if self._kge_method is None:
            raise AttributeError(msg.MODEL_NOT_TRAINED_OR_LOADED)
        
        classes = self.dataset.classes.as_str
        if len(classes) == 0:
            return dict()

        
        cls_graph_id = {cls: self.triples_factory.entity_to_id[cls] for cls in classes if cls in self.triples_factory.entity_to_id}

        def get_embedding(idxs):
            return self._kge_method.entity_representations[0](indices=idxs).cpu().detach().numpy()

        idxs = th.tensor(list(cls_graph_id.values()))
        cls_embeddings = dict(zip(cls_graph_id.keys(), get_embedding(idxs)))
        return cls_embeddings

    @property
    def object_property_embeddings(self):
        if self._kge_method is None:
            raise AttributeError(msg.MODEL_NOT_TRAINED_OR_LOADED)

        object_properties = self.graph_relation_to_id.keys()
        if len(object_properties) == 0:
            return dict()

        op_graph_id = {op: self.triples_factory.relation_to_id[op] for op in object_properties if op in self.triples_factory.relation_to_id}

        def get_embedding(idxs):
            return self._kge_method.relation_representations[0](indices=idxs).cpu().detach().numpy()

        idxs = th.tensor(list(op_graph_id.values()))
        op_embeddings = dict(zip(op_graph_id.keys(), get_embedding(idxs)))
        return op_embeddings
    
    @property
    def individual_embeddings(self):
        if self._kge_method is None:
            raise AttributeError(msg.MODEL_NOT_TRAINED_OR_LOADED)
        
        individuals = self.dataset.individuals.as_str
        if len(individuals) == 0:
            return dict()

        ind_graph_id = {ind: self.triples_factory.entity_to_id[ind] for ind in individuals if ind in self.triples_factory.entity_to_id}

        def get_embedding(idxs):
            return self._kge_method.entity_representations[0](indices=idxs).cpu().detach().numpy()

        idxs = th.tensor(list(ind_graph_id.values()))
        ind_embeddings = dict(zip(ind_graph_id.keys(), get_embedding(idxs)))
        return ind_embeddings

    @property
    def evaluation_model(self):
        if self._evaluation_model is None:
            self._evaluation_model = EvaluationModel(self._kge_method, self.device)

        return self._evaluation_model

    
    def set_kge_method(self, kge_method, *args, **kwargs):
        """
        Set the KGE method of the model.

        :param kge_method: The KGE method.
        :type kge_method: :class:`pykeen.models.ERModel`
        """
        try:
            self._kge_method_uninitialized = kge_method
            initialized_kge_method = kge_method(*args, triples_factory=self.triples_factory, **kwargs)
        except TypeError:
            raise TypeError(f"Parameter 'kge_method' must be a pykeen.models.ERModel object. Got {type(kge_method)} instead.")
                    
        if not isinstance(initialized_kge_method, ERModel):
            raise TypeError(f"Parameter 'kge_method' must be a pykeen.models.ERModel object. Got {type(kge_method)} instead.")

        
        self._kge_method = initialized_kge_method.to(self.device)
        self._kge_method_args = args
        self._kge_method_kwargs = kwargs



    def train(self, epochs=0):
        """
        Triggers the PyKEEN training process.

        :param epochs: Number of epochs to train the model. If None, the value of the epochs parameter passed to the constructor will be used.
        :type epochs: int
        """

        if self._kge_method is None:
            raise AttributeError(msg.PYKEEN_MODEL_NOT_SET)
        if self.optimizer is None:
            raise AttributeError(msg.PYKEEN_OPTIMIZER_NOT_SET)
        if self.lr is None:
            raise AttributeError(msg.PYKEEN_LR_NOT_SET)
        if self.batch_size is None:
            raise AttributeError(msg.PYKEEN_BATCH_SIZE_NOT_SET)

        self._kge_method.train()
        optimizer = self.optimizer(params=self._kge_method.get_grad_params(), lr=self.lr)

        training_loop = SLCWATrainingLoop(model=self._kge_method, triples_factory=self.triples_factory,
                                          optimizer=optimizer)

        _ = training_loop.train(triples_factory=self.triples_factory, num_epochs=epochs,
                                batch_size=self.batch_size)

        th.save(self._kge_method, self.model_filepath)

        
    def add_axioms(self, *axioms):
        prev_class_embeds = copy.deepcopy(self.class_embeddings)
        prev_object_property_embeds = copy.deepcopy(self.object_property_embeddings)
        prev_individual_embeds = copy.deepcopy(self.individual_embeddings)
        prev_relation_to_id = self.triples_factory.relation_to_id
        print(f"Number of classes before adding axioms: {len(prev_class_embeds)}")
        print(f"Number of object properties before adding axioms: {len(prev_object_property_embeds)}")
        print(f"Number of individuals before adding axioms: {len(prev_individual_embeds)}")
        
        self.dataset.add_axioms(*axioms)
        self._load_edges()
        self._triples_factory = Edge.as_pykeen(self.edges, entity_to_id = self.graph_node_to_id,
                                               relation_to_id = self.graph_relation_to_id, create_inverse_triples=False)


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
        
        new_kge_method = self._kge_method_uninitialized(triples_factory=self.triples_factory,
                                                        entity_initializer=PretrainedInitializer(tensor=pretrained_cls_embeddings),
                                                        relation_initializer=PretrainedInitializer(tensor=pretrained_op_embeddings),
                                                        *self._kge_method_args, **self._kge_method_kwargs)
        self._kge_method = new_kge_method



    def from_pretrained(self, model):
        #self._model_filepath = model

        if not isinstance(model, str):
            raise TypeError("Parameter model must be a string pointing to the PyKEEN model file.")

        if not os.path.exists(model):
            raise FileNotFoundError("Pretrained model path does not exist")
        
        self._is_pretrained = True
        if not isinstance(model, str):
            raise TypeError

        self._kge_method = th.load(model, weights_only=False)
        #self._kge_method = kge_method
    



class EvaluationModel(th.nn.Module):
    def __init__(self, kge_model, device):
        logger.warning("A custom EvaluationModel should be created depending on the task. This is a generic one.")
        super().__init__()

        self.kge_model = kge_model
        self.device = device
        
    def forward(self, data, *args, **kwargs):
        logits = self.kge_model.score_hrt(data)
        return - logits
    
