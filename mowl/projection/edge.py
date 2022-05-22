
class Edge:
    def __init__(self, src, rel, dst, weight = 1):
        self._src = src
        self._rel = rel
        self._dst = "" if dst == "" else dst
        self._weight = weight

    def src(self):
        """Getter method for _src attribute
        
        :rtype: str
        """
        return self._src

    def rel(self):
        """Getter method for _rel attribute
        
        :rtype: str
        """
        return self._rel

    def dst(self):
        """Getter method for _dst attribute
        
        :rtype: str
        """
        return self._dst

    def weight(self):
        """Getter method for _weight attribute
        
        :rtype: str
        """
        return self._weight

    def astuple(self):
        return tuple(map(str, (self.src(), self.rel(), self.dst())))


    
    @staticmethod
    def getEntitiesAndRelations(edges):
        '''
        :param edges: list of edges 
        :type edges: :class:`Edge`

        :returns: Returns a 2-tuple containing the list of entities (heads and tails) and the list of relations
        :rtype: (Set of str, Set of str)
        '''


        entities = set()
        relations = set()

        for edge in edges:
            entities |= {edge.src(), edge.dst()}
            relations |= {edge.rel()}

        return (entities, relations)

    @staticmethod
    def zip(edges):
        return tuple(zip(*[x.astuple() for x in edges]))
