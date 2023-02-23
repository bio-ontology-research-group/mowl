from mowl.base_models.model import Model
from mowl.projection.base import ProjectionModel
from mowl.projection import Edge
from mowl.walking import WalkingModel
import pykeen

class GraphModel(Model):

    def __init__(self, *args, **kwargs):
        super(GraphModel, self).__init__(*args, **kwargs)

        self._edges = None
        self._graph_node_to_id = None
        self._graph_relation_to_id = None
        self.projector = None


    def _load_edges(self):
        if self.projector is None:
            raise ValueError("No projector found. Please set a projector first using the 'set_projector' method")

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

        
    def set_projector(self, projector):

        if not isinstance(projector, ProjectionModel):
            raise TypeError("Parameter 'projector' must be a mowl.projection.Projector object")
        
        self.projector = projector

    

class RandomWalkModel(GraphModel):

    def __init__(self, *args, **kwargs):
        super(RandomWalkModel, self).__init__(*args, **kwargs)

    def set_walker(self, walker):
        if not isinstance(walker, WalkingModel):
            raise TypeError("Parameter 'walker' must be a mowl.walking.WalkingModel object")
        self.walker = walker

    

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
    
    
