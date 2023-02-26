from mowl.base_models.model import Model
from mowl.projection.base import ProjectionModel
from mowl.projection import Edge
from mowl.walking import WalkingModel
import pykeen
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import mowl.error.messages as msg

class GraphModel(Model):

    def __init__(self, *args, **kwargs):
        super(GraphModel, self).__init__(*args, **kwargs)

        self._edges = None
        self._graph_node_to_id = None
        self._graph_relation_to_id = None
        self._projector = None


    def _load_edges(self):
        if self.projector is None:
            raise ValueError(msg.GRAPH_MODEL_PROJECTOR_NOT_SET)

        self._edges = self.projector.project(self.dataset.ontology)
        nodes, relations = Edge.get_entities_and_relations(self._edges)
        nodes = list(set(nodes))
        relations = list(set(relations))
        nodes.sort()
        relations.sort()
        self._graph_node_to_id = {node: i for i, node in enumerate(nodes)}
        self._graph_relation_to_id = {relation: i for i, relation in enumerate(relations)}
        
    @property
    def edges(self):
        if self._edges is not None:
            return self._edges

        self._load_edges()
        return self._edges

    @property
    def graph_node_to_id(self):
        if self._graph_node_to_id is not None:
            return self._graph_node_to_id

        self._load_edges()
        return self._graph_node_to_id

    @property
    def graph_relation_to_id(self):
        if self._graph_relation_to_id is not None:
            return self._graph_relation_to_id

        self._load_edges()
        return self._graph_relation_to_id


    @property
    def projector(self):
        return self._projector
        
    def set_projector(self, projector):

        if not isinstance(projector, ProjectionModel):
            raise TypeError("Parameter 'projector' must be a mowl.projection.Projector object")
        
        self._projector = projector

    

class RandomWalkModel(GraphModel):

    def __init__(self, *args, **kwargs):
        super(RandomWalkModel, self).__init__(*args, **kwargs)

        self._walker = None

    @property
    def walker(self):
        return self._walker
        
    def set_walker(self, walker):
        if not isinstance(walker, WalkingModel):
            raise TypeError("Parameter 'walker' must be a mowl.walking.WalkingModel object")
        self._walker = walker

    def train(self):
        raise NotImplementedError


class RandomWalkPlusW2VModel(RandomWalkModel):

    def __init__(self, *args, **kwargs):
        super(RandomWalkPlusW2VModel, self).__init__(*args, **kwargs)

        self.w2v_model = None

    @property
    def class_embeddings(self):
        if self.w2v_model is None:
            raise AttributeError(msg.W2V_MODEL_NOT_SET)
        if len(self.w2v_model.wv) == 0:
            raise AttributeError(msg.RANDOM_WALK_MODEL_EMBEDDINGS_NOT_FOUND)
        
        cls_embeds = {}
        for cls in self.dataset.classes.as_str:
            if cls in self.w2v_model.wv:
                cls_embeds[cls] = self.w2v_model.wv[cls]
        return cls_embeds

    @property
    def object_property_embeddings(self):
        if self.w2v_model is None:
            raise AttributeError(msg.W2V_MODEL_NOT_SET)
        if len(self.w2v_model.wv) == 0:
            raise AttributeError(msg.RANDOM_WALK_MODEL_EMBEDDINGS_NOT_FOUND)

        obj_prop_embeds = {}
        for obj_prop in self.dataset.object_properties.as_str:
            if obj_prop in self.w2v_model.wv:
                obj_prop_embeds[obj_prop] = self.w2v_model.wv[obj_prop]
        return obj_prop_embeds

    @property
    def individual_embeddings(self):
        if self.w2v_model is None:
            raise AttributeError(msg.W2V_MODEL_NOT_SET)
        if len(self.w2v_model.wv) == 0:
            raise AttributeError(msg.RANDOM_WALK_MODEL_EMBEDDINGS_NOT_FOUND)
        
        
        ind_embeds = {}
        for ind in self.dataset.individuals.as_str:
            if ind in self.w2v_model.wv:
                obj_prop_embeds[ind] = self.w2v_model.wv[ind]
        return ind_embeds

    
    def set_w2v_model(self, *args, **kwargs):
        self.w2v_model = Word2Vec(*args, **kwargs)

    def train(self):
        if self.projector is None:
            raise AttributeError(msg.GRAPH_MODEL_PROJECTOR_NOT_SET)
        if self.walker is None:
            raise AttributeError(msg.RANDOM_WALK_MODEL_WALKER_NOT_SET)
        if self.w2v_model is None:
            raise AttributeError(msg.W2V_MODEL_NOT_SET)
        
        edges = self.projector.project(self.dataset.ontology)
        self.walker.walk(edges)
        sentences = LineSentence(self.walker.outfile)
        self.w2v_model.build_vocab(sentences)
        self.w2v_model.train(sentences, total_examples=self.w2v_model.corpus_count, epochs=self.w2v_model.epochs)
        

        

class KGEModel(GraphModel):

    def __init__(self, *args, **kwargs):
        super(KGEModel, self).__init__(*args, **kwargs)

        self._triples_factory = None
                        
    @property
    def triples_factory(self):
        if self._triples_factory is not None:
            return self._triples_factory

        self._triples_factory = Edge.as_pykeen(self.edges, entity_to_id = self.graph_node_to_id, relation_to_id = self.graph_relation_to_id)

        return self._triples_factory
        
    def set_kge_method(self, kge_method):
        if not isinstance(kge_method, pykeen.models.ERModel):
            raise TypeError("Parameter 'kge_method' must be a pykeen.models.ERModel object")
        self.kge_method = kge_method
    
    
