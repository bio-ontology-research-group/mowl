"""Make the OWLAPI 5 ontology methods callable from Python as they were with OWLAPI 4.

OWLAPI 5 declares methods such as ``getAxioms``, ``getClassesInSignature`` or
``getSignature`` as default methods of several interfaces (``HasAxioms``,
``OWLAxiomCollection``, ``OWLAxiomIndex``, ...). JPype resolves a method name that a Java
class does not declare itself to the first interface that has it, and then sees only the
overloads of that one interface: ``ontology.getAxioms()`` fails with "No matching overloads"
although Java accepts it. The customizers below define these names on the implementation
classes and try the interfaces in turn, so every overload is reachable under its Java name.

The module only registers the customizers. JPype applies them when it first loads the
classes, so the module is imported before the JVM starts and needs no JVM itself.
"""
from jpype import JClass, JImplementationFor

_MODEL = "org.semanticweb.owlapi.model."

#: Method name -> the interfaces that declare overloads of it, most common first.
_ONTOLOGY_METHODS = {
    "getAxioms": ("HasAxioms", "HasAxiomsByType", "OWLAxiomCollection", "OWLAxiomIndex",
                  "OWLAxiomCollectionBooleanArgs"),
    "getLogicalAxioms": ("HasLogicalAxioms", "OWLAxiomCollection",
                         "OWLAxiomCollectionBooleanArgs"),
    "getAxiomsIgnoreAnnotations": ("OWLAxiomCollection", "OWLAxiomCollectionBooleanArgs"),
    "getReferencingAxioms": ("OWLAxiomCollection", "OWLAxiomCollectionBooleanArgs"),
    "getSignature": ("HasSignature", "OWLOntology"),
    "getClassesInSignature": ("HasClassesInSignature", "OWLSignature",
                              "OWLSignatureBooleanArgs"),
    "getObjectPropertiesInSignature": ("HasObjectPropertiesInSignature", "OWLSignature",
                                       "OWLSignatureBooleanArgs"),
    "getDataPropertiesInSignature": ("HasDataPropertiesInSignature", "OWLSignature",
                                     "OWLSignatureBooleanArgs"),
    "getIndividualsInSignature": ("HasIndividualsInSignature", "OWLSignature",
                                  "OWLSignatureBooleanArgs"),
    "getDatatypesInSignature": ("HasDatatypesInSignature", "OWLSignature",
                                "OWLSignatureBooleanArgs"),
    "getAnnotationPropertiesInSignature": ("HasAnnotationPropertiesInSignature",
                                           "OWLSignature", "OWLSignatureBooleanArgs"),
    "getEntitiesInSignature": ("HasGetEntitiesInSignature", "OWLSignature",
                               "OWLSignatureBooleanArgs"),
    "getReferencedAnonymousIndividuals": ("OWLSignature", "OWLSignatureBooleanArgs"),
    "applyChangesAndGetDetails": ("HasApplyChanges", "OWLOntology"),
    "remove": ("HasDirectRemoveAxiom", "HasDirectRemoveAxioms"),
}

_MANAGER_METHODS = {
    "getOntologies": ("HasGetOntologies", "OWLOntologyManager"),
}


def _dispatch(name, interfaces):
    """A method that calls ``name`` through the first interface with a matching overload."""

    def method(self, *args):
        error = None
        for interface in interfaces:
            try:
                return getattr(JClass(_MODEL + interface), name)(self, *args)
            except TypeError as exc:
                if "overload" not in str(exc):
                    raise
                error = exc
        raise error

    method.__name__ = name
    method.__doc__ = f"``{name}`` of OWLAPI, with the overloads of {', '.join(interfaces)}."
    return method


@JImplementationFor("uk.ac.manchester.cs.owl.owlapi.OWLImmutableOntologyImpl")
class _OntologyMethods:
    pass


@JImplementationFor("uk.ac.manchester.cs.owl.owlapi.OWLOntologyManagerImpl")
class _ManagerMethods:
    pass


for _name, _interfaces in _ONTOLOGY_METHODS.items():
    setattr(_OntologyMethods, _name, _dispatch(_name, _interfaces))
for _name, _interfaces in _MANAGER_METHODS.items():
    setattr(_ManagerMethods, _name, _dispatch(_name, _interfaces))
