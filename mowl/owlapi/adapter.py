"""This module implements shortcut methods to access some OWLAPI objects."""
from org.semanticweb.owlapi.apibinding import OWLManager
from org.semanticweb.owlapi.model import IRI

import mowl.error as err

class OWLAPIAdapter():
    """Adapter class adapting OWLAPI. Here you can find shortcuts to:
    * OWLManager
    * OWLDataFactory
    * IRI
    """
    def __init__(self):

        self._owl_manager = None
        self._data_factory = None

    @property
    def owl_manager(self):
        """Creates a OWLManager from OWLAPI
        :rtype: org.semanticweb.owlapi.apibinding.OWLManager
        """
        if self._owl_manager is None:
            self._owl_manager = OWLManager()

        return self._owl_manager

    @property
    def data_factory(self):
        """Creates an OWLDataFactory from OWLAPI. If OWLManager does not exist, it is created as well.
        :rtype: org.semanticweb.owlapi.model.OWLDataFactory
        """

        if self._data_factory is None:
            self._data_factory = self.owl_manager.getOWLDataFactory()
        return self._data_factory


    def create_class(self, iri):
        """Creates and OWL class given a valid IRI string"""

        if not isinstance(iri, str):
            raise TypeError(f"IRI must be a string to use this method. {err.OWLAPI_DIRECT}")
        return self.data_factory.getOWLClass(IRI.create(iri))
