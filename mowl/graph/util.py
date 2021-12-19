from org.mowl.Parsers import TaxonomyParser
from org.mowl.Parsers import TaxonomyWithRelsParser
from org.mowl.Parsers import DL2VecParser

from mowl.graph.owl2vec_star.model import OWL2VecParser

def gen_factory(method_name, dataset):
    methods = [
        "taxonomy",
        "taxonomy_rels",
        "dl2vec",
        "owl2vec_star"
        "categorical"
    ]
    
    if method_name == "taxonomy":
        return TaxonomyParser(dataset, True)
    elif method_name == "taxonomy_rels":
        return TaxonomyParser(dataset, True)
    elif method_name == "dl2vec":
        return DL2VecParser(dataset, True)
    elif method_name == "owl2vec_star":
        return OWL2VecParser(dataset)
    else:
        raise Exception(f"Graph generation method unrecognized. Recognized methods are: {methods}")
