from mowl.graph.taxonomy.model import TaxonomyParser

def gen_factory(method_name, dataset):
    methods = [
        "taxonomy",
        "taxonomy_rels"
    ]
    if method_name == "taxonomy":
        return TaxonomyParser(dataset, subclass=True, relations=False)
    elif method_name == "taxonomy_rels":
        return TaxonomyParser(dataset, subclass = True, relations=True)
    else:
        raise Exception(f"Graph generation method unrecognized. Recognized methods are: {methods}")
