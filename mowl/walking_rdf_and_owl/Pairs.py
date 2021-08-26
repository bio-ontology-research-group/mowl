class Pair():
    
    def __init__(self, node1, node2, score = 0.0):
        self.node1 = node1
        self.node2 = node2
        self.score = float(score)
    def __repr__(self):
        return '\t'.join((self.node1, self.node2, str(self.score)))

    def __eq__(self, other):
        on_node1 = self.node1 == other.node1
        on_node2 = self.node2 == other.node2

        return on_node1 and on_node2

    def __key(self):
        return (self.node1, self.node2)

    def __hash__(self):
        return hash(self.__key())
