from mowl.graph.taxonomy.model import TaxonomyParser
from mowl.graph.taxonomyRels.model import TaxonomyWithRelsParser
from mowl.graph.dl2vec.model import DL2VecParser

from mowl.graph.owl2vec_star.model import OWL2VecStarParser



PARSING_METHODS = ["taxonomy", "taxonomy_rels", "dl2vec", "owl2vec_star"]

def parser_factory(method_name, dataset, bidirectional_taxonomy, include_literals = False, only_taxonomy = False): 
    
    if method_name == "taxonomy":
        return TaxonomyParser(dataset, bidirectional_taxonomy=bidirectional_taxonomy)
    elif method_name == "taxonomy_rels":
        return TaxonomyWithRelsParser(dataset, bidirectional_taxonomy=bidirectional_taxonomy)
    elif method_name == "dl2vec":
        return DL2VecParser(dataset, bidirectional_taxonomy= bidirectional_taxonomy)
    elif method_name == "owl2vec_star":
        return OWL2VecStarParser(dataset, bidirectional_taxonomy=bidirectional_taxonomy, include_literals = include_literals, only_taxonomy = only_taxonomy)
    else:
        raise Exception(f"Graph generation method unrecognized. Recognized methods are: {PARSING_METHODS}")
