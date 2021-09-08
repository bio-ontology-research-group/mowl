from mowl.model import Model

class GraphGenModel(Model):
    def __init__(self, dataset):
        super().__init__(dataset)

    def parseOWL(self):
        raise NotImplementedError()


    
