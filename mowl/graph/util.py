from mowl.graph.taxonomy.model import TaxonomyParser
from mowl.graph.dl2vec.model import DL2VecParser
from mowl.graph.owl2vec_star.model import OWL2VecParser
from mowl.graph.catont.model import CatOnt

def parser_factory(method_name, dataset, bidirectional_taxonomy=False): #TODO include parameters for OWL2Vec*
    methods = [
        "taxonomy",
        "taxonomy_rels",
        "dl2vec",
        "owl2vec_star"
    ]
    
    if method_name == "taxonomy":
        return TaxonomyParser(dataset, bidirectional_taxonomy=bidirectional_taxonomy)
    elif method_name == "taxonomy_rels":
        return TaxonomyParser(dataset, bidirectional_taxonomy=bidirectional_taxonomy)
    elif method_name == "dl2vec":
        return DL2VecParser(dataset, bidirectional_taxonomy= bidirectional_taxonomy)
    elif method_name == "owl2vec_star":
        return OWL2VecParser(dataset, bidirectional_taxonomy=bidirectional_taxonomy)
    else:
        raise Exception(f"Graph generation method unrecognized. Recognized methods are: {methods}")
