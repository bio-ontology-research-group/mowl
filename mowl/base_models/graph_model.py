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
        """
        Returns the edges of the graph as a list of :class:`mowl.projection.edge.Edge` objects.

        :rtype: list of :class:`Edge <mowl.projection.edge.Edge>` objects
        """
        if self._edges is not None:
            return self._edges

        self._load_edges()
        return self._edges

    @property
    def graph_node_to_id(self):
        """
        Returns a dictionary that maps graph nodes to ids.

        :rtype: dict
        """
        if self._graph_node_to_id is not None:
            return self._graph_node_to_id

        self._load_edges()
        return self._graph_node_to_id

    @property
    def graph_relation_to_id(self):
        """
        Returns a dictionary that maps graph relations (edge labels) to ids.

        :rtype: dict
        """
        
        if self._graph_relation_to_id is not None:
            return self._graph_relation_to_id

        self._load_edges()
        return self._graph_relation_to_id


    @property
    def projector(self):
        """
        Returns the projector of the graph model.

        :rtype: :class:`ProjectionModel <mowl.projection.base.ProjectionModel>` object
        """
        return self._projector
        
    def set_projector(self, projector):
        """
        Sets the projector of the graph model.

        :param projector: the projector to be set
        :type projector: :class:`Projection <mowl.projection.base.ProjectionModel>` object
        """
        
        if not isinstance(projector, ProjectionModel):
            raise TypeError("Parameter 'projector' must be a mowl.projection.Projector object")
        
        self._projector = projector

    

class RandomWalkModel(GraphModel):

    def __init__(self, *args, **kwargs):
        super(RandomWalkModel, self).__init__(*args, **kwargs)

        self._walker = None

    @property
    def walker(self):
        """
        Returns the walker object of the random walk model.

        :rtype: :class:`WalkingModel <mowl.walking.base.WalkingModel>` object
        """
        return self._walker
        
    def set_walker(self, walker):
        """
        Sets the walker object of the random walk model.

        :param walker: the walker to be set
        :type walker: :class:`WalkingModel <mowl.walking.walking.WalkingModel>` object
        """
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
        """
        Returns the knowledge graph embedding method of the KGE model.

        :rtype: Defined in child classes
        """
        
        if self._kge_method is None:
            raise AttributeError(msg.KGE_METHOD_NOT_SET)
        return self._kge_method
        
    def set_kge_method(self, kge_method):
        """
        Sets the knowledge graph embedding method of the KGE model.

        :param kge_method: the kge method to be set
        :type kge_method: Defined in child classes
        """
        
        raise NotImplementedError
                            
    
