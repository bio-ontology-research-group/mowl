from pykeen.models import ERModel

from mowl.base_models import KGEModel
from mowl.projection import Edge
import torch as th
import copy
import numpy as np
from pykeen.nn.init import PretrainedInitializer

class GraphPlusPyKEENModel(KGEModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._triples_factory = None
        self._kge_method = None
                        

    @property
    def triples_factory(self):
        if self._triples_factory is not None:
            return self._triples_factory

        self._triples_factory = Edge.as_pykeen(self.edges, entity_to_id = self.graph_node_to_id, relation_to_id = self.graph_relation_to_id, create_inverse_triples=False)
        self._graph_node_to_id = self._triples_factory.entity_to_id
        self._graph_relation_to_id = self._triples_factory.relation_to_id
        return self._triples_factory


    @property
    def class_embeddings(self):
        classes = self.dataset.classes.as_str
        if len(classes) == 0:
            return dict()

        
        cls_graph_id = {cls: self.triples_factory.entity_to_id[cls] for cls in classes if cls in self.triples_factory.entity_to_id}

        def get_embedding(idxs):
            return self.kge_method.entity_representations[0](indices=idxs).cpu().detach().numpy()

        idxs = th.tensor(list(cls_graph_id.values()))
        cls_embeddings = dict(zip(cls_graph_id.keys(), get_embedding(idxs)))
        return cls_embeddings

    @property
    def object_property_embeddings(self):
        object_properties = self.graph_relation_to_id.keys()
        if len(object_properties) == 0:
            return dict()

        op_graph_id = {op: self.triples_factory.relation_to_id[op] for op in object_properties if op in self.triples_factory.relation_to_id}

        def get_embedding(idxs):
            return self.kge_method.relation_representations[0](indices=idxs).cpu().detach().numpy()

        idxs = th.tensor(list(op_graph_id.values()))
        op_embeddings = dict(zip(op_graph_id.keys(), get_embedding(idxs)))
        return op_embeddings
    
    @property
    def individual_embeddings(self):
        individuals = self.dataset.individuals.as_str
        if len(individuals) == 0:
            return dict()

        ind_graph_id = {ind: self.triples_factory.entity_to_id[ind] for ind in individuals if ind in self.triples_factory.entity_to_id}

        def get_embedding(idxs):
            return self.kge_method.entity_representations[0](indices=idxs).cpu().detach().numpy()

        idxs = th.tensor(list(ind_graph_id.values()))
        ind_embeddings = dict(zip(ind_graph_id.keys(), get_embedding(idxs)))
        return ind_embeddings
                     
    def set_kge_method(self, kge_method, *args, **kwargs):
        try:
            self._kge_method_uninitialized = kge_method
            initialized_kge_method = kge_method(triples_factory=self.triples_factory, *args, **kwargs)
            
        except TypeError:
            raise TypeError(f"Parameter 'kge_method' must be a pykeen.models.ERModel object. Got {type(kge_method)} instead.")

            
        if not isinstance(initialized_kge_method, ERModel):
            raise TypeError(f"Parameter 'kge_method' must be a pykeen.models.ERModel object. Got {type(kge_method)} instead.")

        
        self._kge_method = initialized_kge_method
        self._kge_method_args = args
        self._kge_method_kwargs = kwargs
                
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
                class_size = self.kge_method.entity_representations[0](indices=None).shape[1]
                new_class_embeds.append(np.random.normal(size=class_size))
                
        new_class_embeds = np.asarray(new_class_embeds)

        new_object_property_embeds = []
        for new_relation, new_id in self.graph_relation_to_id.items():
            if new_relation in prev_object_property_embeds:
                new_object_property_embeds.append(prev_object_property_embeds[new_relation])
            else:
                op_size = self.kge_method.relation_representations[0](indices=None).shape[1]
                new_object_property_embeds.append(np.random.normal(size=op_size))

        new_object_property_embeds = np.asarray(new_object_property_embeds)

        pretrained_cls_embeddings = th.tensor(new_class_embeds)
        pretrained_op_embeddings = th.tensor(new_object_property_embeds)
        
        new_kge_method = self._kge_method_uninitialized(triples_factory=self.triples_factory,
                                                        entity_initializer=PretrainedInitializer(tensor=pretrained_cls_embeddings),
                                                        relation_initializer=PretrainedInitializer(tensor=pretrained_op_embeddings),
                                                        *self._kge_method_args, **self._kge_method_kwargs)
        self._kge_method = new_kge_method
