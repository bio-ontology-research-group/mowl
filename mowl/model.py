from org.mowl.



class Model(object):

    def __init__(self, dataset):
        self.dataset = dataset

    def train(self):
        raise NotImplementedError()
    
    def evaluate(self):
        raise NotImplementedError()




class GraphGenModel(Model):
    def __init__(self, dataset):
        super().__init__(dataset)

    def parseOWL(self):
        raise NotImplementedError()


    
def gen_factory(method_name, params):
    methods = [
        "taxonomy",
        "taxonomy_rels"
    ]
    if method_name == "taxonomy":
        return TaxonomyParser()
    elif method_name == "taxonomy_rels":
        return TaxonomyParser()
    else:
        raise Exception(f"Graph generation method unrecognized. Recognized methods are: {methods}")

