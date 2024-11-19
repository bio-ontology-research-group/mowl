from pykeen.triples import TriplesFactory
from deprecated.sphinx import versionadded, deprecated, versionchanged
import numpy as np
import torch as th


class Edge:
    """Class representing a graph edge.
    """

    def __init__(self, src, rel, dst, weight=1.):

        if not isinstance(src, str):
            raise TypeError("Parameter src must be a string")
        if not isinstance(rel, str):
            raise TypeError("Parameter rel must be a string")
        if not isinstance(dst, str):
            raise TypeError("Parameter dst must be a string")
        if not isinstance(weight, float):
            raise TypeError("Optional parameter weight must be a float")

        self._src = src
        self._rel = rel
        self._dst = "" if dst == "" else dst
        self._weight = weight

    @property
    def src(self):
        """
        Getter method for ``_src`` attribute

        :rtype: str
        """
        return self._src

    @property
    def rel(self):
        """
        Getter method for ``_rel`` attribute

        :rtype: str
        """
        return self._rel

    @property
    def dst(self):
        """
        Getter method for ``_dst`` attribute

        :rtype: str
        """
        return self._dst

    @property
    def weight(self):
        """
        Getter method for ``_weight`` attribute

        :rtype: str
        """
        return self._weight

    def astuple(self):
        return tuple(map(str, (self._src, self._rel, self._dst)))

    @staticmethod
    @deprecated(version="0.1.0", reason="Use get_entities_and_relations instead")
    def getEntitiesAndRelations(edges):
        return Edge.get_entities_and_relations(edges)

    @staticmethod
    @versionchanged(version="1.0.2", reason="Method return type changed to tuple of lists")
    def get_entities_and_relations(edges):
        '''
        :param edges: list of edges
        :type edges: :class:`Edge`

        :returns: Returns a 2-tuple containing the list of entities (heads and tails) and the \
            list of relations
        :rtype: (list of str, list of str)
        '''

        entities = set()
        relations = set()

        for edge in edges:
            entities |= {edge.src, edge.dst}
            relations |= {edge.rel}

        entities = sorted(list(entities))
        relations = sorted(list(relations))
            
        return (entities, relations)

    @staticmethod
    def zip(edges):
        return tuple(zip(*[x.astuple() for x in edges]))

    @staticmethod
    @versionadded(version="0.1.0", reason="This method is available to transform graph edges \
        obtained from ontologies into PyKEEN triples.")
    def as_pykeen(edges, create_inverse_triples=True, entity_to_id=None, relation_to_id=None):
        """
        This method transform a set of edges into an object of the type
        :class:`pykeen.triples.triples_factory.TriplesFactory`. This method is intended to be
        used for PyKEEN methods.

        :param edges: List of edges.
        :type edges: list of :class:`Edge`
        :param create_inverse_triple: Whether to create inverse triples. Defaults to ``True``
        :type create_inverse_triple: bool, optional
        :rtype: :class:`pykeen.triples.triples_factory.TriplesFactory`
        """
        if entity_to_id is None or relation_to_id is None:
            classes, relations = Edge.getEntitiesAndRelations(edges)

        if entity_to_id is None:
            entity_to_id = {v: k for k, v in enumerate(classes)}
        if relation_to_id is None:
            relation_to_id = {v: k for k, v in enumerate(relations)}

        def map_edge(edge):
            return [entity_to_id[edge.src], relation_to_id[edge.rel], entity_to_id[edge.dst]]

        triples = [map_edge(edge) for edge in edges]
        triples = np.array(triples, dtype=int)
        tensor_triples = th.tensor(triples)

        triples_factory = TriplesFactory(tensor_triples, entity_to_id=entity_to_id,
                                         relation_to_id=relation_to_id,
                                         create_inverse_triples=create_inverse_triples)
        return triples_factory
