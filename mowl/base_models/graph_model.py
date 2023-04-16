from mowl.base_models.model import Model
from mowl.projection.base import ProjectionModel
from mowl.projection import Edge
from mowl.walking import WalkingModel
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



class KGEModel(GraphModel):

    def __init__(self, *args, **kwargs):
        super(KGEModel, self).__init__(*args, **kwargs)

        self._kge_method = None

    @property
    def kge_method(self):
        if self._kge_method is None:
            raise AttributeError(msg.KGE_METHOD_NOT_SET)
        return self._kge_method
        
    def set_kge_method(self, kge_method):
        raise NotImplementedError
                            
    
