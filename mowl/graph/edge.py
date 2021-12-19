class Edge:
    def __init__(self, src, rel, dst):
        self.src_ = self.prettyFormat(src)
        self.rel_ = rel
        self.dst_ = self.prettyFormat(dst)


    def src(self):
        return self.src_

    def rel(self):
        return self.rel_

    def dst(self):
        return self.dst_

    def prettyFormat(self, string):
        '''
        If string is of the form <http://purl.obolibrary.org/obo/GO_0071554> this function returns GO:0071554
        '''
        if string[0] == "<" and string[-1] == ">":
            string = string[1:-1]
            string = string.split("/")[-1]
            string = string.replace("_", ":")
        elif string.startswith("http"):
            string = string.split("/")[-1]
            string = string.replace("_", ":")
        elif string.startswith("GO:"):
            pass
        else:
            pass
#            raise Exception("prettyFormat: unrecognized string format: %s", string)

        return string


    @staticmethod
    def getEntitiesAndRelations(edges):
        '''
        :param edges: list of edges 
        :type edges: :class:`Edge`

        :returns: Returns a 2-tuple containing the list of entities (heads and tails) and the list of relations
        :rtype: ([str], [str])
        '''


        entities = set()
        relations = set()

        for edge in edges:
            entities |= {edge.src(), edge.dst()}
            relations |= {edge.rel()}

        return (entities, relations)
