


class GraphGenModel(Model):
    def __init__(self, dataset):
        super().__init__(dataset)

    def parseOWL(self):
        raise NotImplementedError()


    
def gen_factory(method_name, dataset):
    methods = [
        "taxonomy",
        "taxonomy_rels"
    ]
    if method_name == "taxonomy":
        return TaxonomyParser(dataset, subclass=True, relations=False)
    elif method_name == "taxonomy_rels":
        return TaxonomyParser(subclass = True, relations=True)
    else:
        raise Exception(f"Graph generation method unrecognized. Recognized methods are: {methods}")

