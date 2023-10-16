from org.mowl.Projectors import TaxonomyWithRelsProjector as Projector
from org.semanticweb.owlapi.model import OWLOntology
from mowl.projection.edge import Edge

from mowl.projection.base import ProjectionModel

from java.util import ArrayList


class TaxonomyWithRelationsProjector(ProjectionModel):

    r'''
    Projection of axioms :math:`A \sqsubseteq B` and :math:`A \sqsubseteq \exists R.B`.

    * :math:`A \sqsubseteq B` will generate the triple
      :math:`\langle A, subClassOf, B \rangle`
    * :math:`A \sqsubseteq \exists R. B` will generate the triple
      :math:`\left\langle A, R, B \right\rangle`


    :param taxonomy: If ``True`` taxonomy axioms will be included.
    :type taxonomy: 
    :param bidirectional_taxonomy: If true then per each SubClass edge one SuperClass edge will \
        be generated.
    '''

    def __init__(self, taxonomy=False, bidirectional_taxonomy: bool = False, relations=None):
        super().__init__()

        if not isinstance(taxonomy, bool):
            raise TypeError('Optional parameter taxonomy must be of type boolean')
        if not isinstance(bidirectional_taxonomy, bool):
            raise TypeError('Optional parameter bidirectional_taxonomy must be of type boolean')
        if relations is not None and not isinstance(relations, list):
            raise TypeError('Optional parameter relations must be of type list or None')

        if not taxonomy and bidirectional_taxonomy:
            raise ValueError("Parameter taxonomy=False incompatible with parameter \
bidirectional_taxonomy=True")
        if not taxonomy and (relations is None or relations == []):
            raise ValueError("Bad configuration of parameters. Either taxonomy should be True or \
relations a non-empty list")

        relations = [] if relations is None else relations
        relationsJ = ArrayList()
        for r in relations:
            relationsJ.add(r)

        self.projector = Projector(taxonomy, bidirectional_taxonomy, relationsJ)

    def project(self, ontology):
        r"""Generates the projection of the ontology.

        :param ontology: The ontology to be processed.
        :type ontology: :class:`org.semanticweb.owlapi.model.OWLOntology`
        """
        
        if not isinstance(ontology, OWLOntology):
            raise TypeError('Parameter ontology must be of type \
org.semanticweb.owlapi.model.OWLOntology')
        edges = self.projector.project(ontology)
        edges = [Edge(str(e.src()), str(e.rel()), str(e.dst())) for e in edges]
        return edges
