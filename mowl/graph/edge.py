class Edge:
    def __init__(self, src, rel, dst, weight = 1):
        self._src = self.prettyFormat(src)
        self._rel = rel
        self._dst = self.prettyFormat(dst)
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



    def prettyFormat(self, text):
        """If text is of the form <http://purl.obolibrary.org/obo/GO_0071554> this function returns GO:0071554
        
        :param text: Text to be formatted
        :type text: str
        
        :rtype: str
        """
        if text[0] == "<" and text[-1] == ">":
            text = text[1:-1]
            text = text.split("/")[-1]
            text = text.replace("_", ":")
        elif text.startswith("http"):
            text = text.split("/")[-1]
            text = text.replace("_", ":")
        elif text.startswith("GO:"):
            pass
        else:
            pass
#            raise Exception("prettyFormat: unrecognized text format: %s", text)

        return text
