from mowl.projection.taxonomy.model import TaxonomyProjector
from mowl.projection.taxonomy_rels.model import TaxonomyWithRelationsProjector
from mowl.projection.dl2vec.model import DL2VecProjector

from mowl.projection.owl2vec_star.model import OWL2VecStarProjector
from mowl.projection.gda.model import GDAProjector


PARSING_METHODS = ["taxonomy", "taxonomy_rels", "dl2vec", "owl2vecstar", "gda"]


def projector_factory(method_name, taxonomy=False, bidirectional_taxonomy=False,
                      include_literals=False, only_taxonomy=False, relations=None):

    if method_name == "taxonomy":
        return TaxonomyProjector(bidirectional_taxonomy=bidirectional_taxonomy)
    elif method_name == "taxonomy_rels":
        return TaxonomyWithRelationsProjector(taxonomy=taxonomy,
                                              bidirectional_taxonomy=bidirectional_taxonomy,
                                              relations=relations)
    elif method_name == "dl2vec":
        return DL2VecProjector(bidirectional_taxonomy=bidirectional_taxonomy)
    elif method_name == "owl2vecstar":
        return OWL2VecStarProjector(bidirectional_taxonomy=bidirectional_taxonomy,
                                    include_literals=include_literals, only_taxonomy=only_taxonomy)
    elif method_name == "gda":
        return GDAProjector(bidirectional_taxonomy=bidirectional_taxonomy,
                            include_literals=include_literals, only_taxonomy=only_taxonomy)
    else:
        raise Exception(f"Graph generation method {method_name} unrecognized. Recognized methods \
are: {PARSING_METHODS}")
