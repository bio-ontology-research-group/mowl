"""Reasoner backends, chosen by expressivity, and subsumption-based entailment checks.

mOWL packages two reasoners: ELK (polynomial, for the EL family) and HermiT (complete for
OWL 2 DL, slower). A backend records which expressivity a reasoner is complete for
(:class:`mowl.logic.Expressivity`), so that the cheapest complete reasoner can be chosen
for an ontology, and a reasoner that is not complete for it is used only with a warning.
Further reasoners (any ``OWLReasonerFactory``, or a Python callable) are added with
:func:`register_reasoner`.

:class:`SubsumptionOracle` answers the standard inference tasks through one operation,
subsumption between class expressions, which it reduces to queries about named classes by
adding definitions ``X ≡ C`` (the reduction every reasoner supports).
"""

import itertools
import warnings
from dataclasses import dataclass

from mowl.logic import Construct, Expressivity, LogicalStatus, ontology_expressivity


@dataclass(frozen=True)
class ReasonerBackend:
    """A reasoner mOWL can use.

    :param name: Identifier, e.g. ``"elk"``.
    :param expressivity: The :class:`~mowl.logic.Expressivity` the reasoner is sound and
        complete for.
    :param factory: ``factory(ontology) -> OWLReasoner``.
    :param cost: Relative cost; among complete backends the cheapest is chosen.
    :param summary: One sentence.
    :param instance_queries: Whether the reasoner answers instance queries
        (``getInstances``, ``getTypes``) through the OWL API. If not,
        :class:`SubsumptionOracle` checks instances as the subsumption ``{a} ⊑ C``.
    """

    name: str
    expressivity: Expressivity
    factory: object
    cost: int = 1
    summary: str = ""
    available: object = None
    instance_queries: bool = True

    def is_available(self):
        """Whether the reasoner can be created in this installation (optional reasoners
        need their JAR on the class path)."""
        return True if self.available is None else bool(self.available())

    def create(self, ontology):
        """Create an ``org.semanticweb.owlapi.reasoner.OWLReasoner`` for ``ontology``."""
        return self.factory(ontology)


def _elk(ontology):
    import java.lang
    from mowl import ELK_LOG_PROPERTY
    # ELK logs through SLF4J. init_jvm sets its level; this covers a JVM started
    # elsewhere, as long as no ELK class has created its logger yet.
    if java.lang.System.getProperty(ELK_LOG_PROPERTY) is None:
        java.lang.System.setProperty(ELK_LOG_PROPERTY, "error")
    from org.semanticweb.elk.owlapi import ElkReasonerFactory
    return ElkReasonerFactory().createReasoner(ontology)


def _hermit(ontology):
    import jpype
    factory = jpype.JClass("org.semanticweb.HermiT.Reasoner$ReasonerFactory")()
    return factory.createReasoner(ontology)


def _structural(ontology):
    from org.semanticweb.owlapi.reasoner.structural import StructuralReasonerFactory
    return StructuralReasonerFactory().createReasoner(ontology)


_BACKENDS = {}


def register_reasoner(name, factory, expressivity, cost=5, summary="", replace=False,
                      available=None, instance_queries=True):
    """Make a reasoner available under ``name``.

    :param factory: ``factory(ontology) -> OWLReasoner``. To register an OWL API
        ``OWLReasonerFactory`` ``f``, pass ``f.createReasoner``.
    :param expressivity: An :class:`~mowl.logic.Expressivity` or a name for
        :meth:`~mowl.logic.Expressivity.parse`: what the reasoner is complete for.
    :param cost: Relative cost used by :func:`select_reasoner` (ELK is 1, HermiT 10).
    :param available: ``available() -> bool``; backends for which it returns ``False`` are
        listed but never chosen automatically.
    """
    if name in _BACKENDS and not replace:
        raise ValueError(f"A reasoner named {name!r} is already registered.")
    if isinstance(expressivity, str):
        expressivity = Expressivity.parse(expressivity)
    _BACKENDS[name] = ReasonerBackend(name, expressivity, factory, cost, summary,
                                      available, instance_queries)
    return _BACKENDS[name]


register_reasoner(
    "structural", _structural, Expressivity.parse("taxonomy"), cost=0,
    summary="OWL API structural reasoner: told subsumptions only, complete for nothing "
            "beyond a taxonomy.")
register_reasoner(
    "elk", _elk,
    Expressivity.parse("ELH(∘)⊥ with ABox") | {Construct.DOMAIN_RANGE}, cost=1,
    summary="ELK 0.4.3: consequence-based, polynomial, for EL with role hierarchies and "
            "chains, ⊥ and assertions; no nominals, inverses or datatypes.")
KM_FACTORY = "org.bioontology.kobayashimarust.protege.KMReasonerFactory"


def _km_owlapi_ok():
    """KM's OWL API binding is built against OWL API 4.5.29 and uses a constructor of the
    functional syntax renderer that older 4.5 releases do not have."""
    import jpype
    try:
        renderer = jpype.JClass(
            "org.semanticweb.owlapi.functional.renderer.FunctionalSyntaxObjectRenderer")
        return any(len(c.getParameterTypes()) == 3
                   and "OWLDocumentFormat" in str(c.getParameterTypes()[1].getName())
                   for c in renderer.class_.getConstructors())
    except Exception:
        return False


def _km_available():
    import jpype
    try:
        jpype.JClass(KM_FACTORY)
    except Exception:
        return False
    return _km_owlapi_ok()


def _km(ontology):
    import jpype
    try:
        factory = jpype.JClass(KM_FACTORY)()
    except Exception as error:
        raise ImportError(
            "The Kobayashi-MaRust (KM) reasoner is not on the class path. Download "
            "kobayashi-marust-protege5-<version>.jar from "
            "https://github.com/bio-ontology-research-group/kobayashi-marust/releases and "
            "start the JVM with mowl.init_jvm(memory, extra_classpath='<path to the jar>') "
            "or set MOWL_EXTRA_CLASSPATH.") from error
    if not _km_owlapi_ok():
        raise RuntimeError(
            "The Kobayashi-MaRust (KM) OWL API binding needs OWL API 4.5.29 or later; the "
            "OWL API on the class path is older.")
    return factory.createReasoner(ontology)


register_reasoner(
    "km", _km, Expressivity.parse("SROIQ(D)").with_abox(equality=True), cost=5,
    available=_km_available, instance_queries=False,
    summary="Kobayashi-MaRust 1.6.5: classifier for OWL 2 DL with verified certificates. "
            "Its OWL API binding classifies; instances are checked as {a} ⊑ C.")
register_reasoner(
    "hermit", _hermit, Expressivity.parse("SROIQ(D)").with_abox(equality=True), cost=10,
    summary="HermiT 1.3.8: hypertableau, sound and complete for OWL 2 DL.")


def list_reasoners():
    """The registered :class:`ReasonerBackend` objects, cheapest first."""
    return sorted(_BACKENDS.values(), key=lambda b: (b.cost, b.name))


def get_backend(name):
    if name not in _BACKENDS:
        raise KeyError(f"Unknown reasoner {name!r}. Registered: {sorted(_BACKENDS)}")
    return _BACKENDS[name]


def select_reasoner(expressivity):
    """The cheapest registered backend that is complete for ``expressivity``.

    :param expressivity: An :class:`~mowl.logic.Expressivity` or its name.
    :raises LookupError: If no registered reasoner covers it.
    """
    if isinstance(expressivity, str):
        expressivity = Expressivity.parse(expressivity)
    for backend in list_reasoners():
        if expressivity <= backend.expressivity and backend.is_available():
            return backend
    raise LookupError(f"No registered reasoner is complete for {expressivity}.")


def create_reasoner(ontology, name="auto", warn=True):
    """Create a reasoner for an OWL API ontology.

    :param name: A registered backend, or ``"auto"`` for the cheapest backend that is
        complete for the expressivity the ontology uses.
    :param warn: Warn if the chosen backend is not complete for the ontology.
    :returns: ``(reasoner, backend)``.
    """
    used = ontology_expressivity(ontology)
    backend = select_reasoner(used) if name == "auto" else get_backend(name)
    if warn and not used <= backend.expressivity:
        missing = ", ".join(c.value for c in backend.expressivity.missing(used))
        warnings.warn(f"The reasoner {backend.name!r} is complete for "
                      f"{backend.expressivity} but the ontology uses {used}; inferences "
                      f"that depend on {missing} are missed.", UserWarning, stacklevel=2)
    return backend.create(ontology), backend


class SubsumptionOracle:
    """Entailment checks for class expressions, by reduction to named classes.

    A reasoner classifies *named* classes. To test ``C ⊑ D`` for class expressions, the
    oracle adds the definitions ``X ≡ C`` and ``Y ≡ D`` for fresh names, asks whether
    ``X ⊑ Y`` is entailed, and removes the definitions again. Instance checking is the same
    test with an individual. This is the explicit ("beforehand") way of turning an inference
    task into a subsumption test; it works with every reasoner and every expressivity the
    reasoner supports.

    :param ontology: An ``org.semanticweb.owlapi.model.OWLOntology``. It is modified only
        temporarily during a query.
    :param reasoner: A backend name, or ``"auto"``.
    """

    _fresh = itertools.count()

    def __init__(self, ontology, reasoner="auto"):
        self.ontology = ontology
        self.manager = ontology.getOWLOntologyManager()
        self.factory = self.manager.getOWLDataFactory()
        self.reasoner, self.backend = create_reasoner(ontology, reasoner)

    def _name(self, expression, added):
        """A named class equivalent to ``expression`` (a new definition if needed)."""
        from org.semanticweb.owlapi.model import IRI
        if not expression.isAnonymous():
            return expression
        fresh = self.factory.getOWLClass(
            IRI.create(f"http://mowl.borg/oracle#X{next(self._fresh)}"))
        from java.util import HashSet
        pair = HashSet()
        pair.add(fresh)
        pair.add(expression)
        axiom = self.factory.getOWLEquivalentClassesAxiom(pair)
        self.manager.addAxiom(self.ontology, axiom)
        added.append(axiom)
        return fresh

    def _remove(self, added):
        for axiom in added:
            self.manager.removeAxiom(self.ontology, axiom)
        if added:
            self.reasoner.flush()

    def is_consistent(self):
        """Whether the ontology has a model."""
        self.reasoner.flush()
        return bool(self.reasoner.isConsistent())

    def entails_subsumption(self, sub, sup):
        """Whether ``sub ⊑ sup`` is entailed, for OWL API class expressions."""
        added = []
        try:
            x, y = self._name(sub, added), self._name(sup, added)
            self.reasoner.flush()
            if not self.reasoner.isSatisfiable(x) or y.isOWLThing():
                return True
            supers = self.reasoner.getSuperClasses(x, False)
            return bool(supers.containsEntity(y)) or bool(
                self.reasoner.getEquivalentClasses(x).contains(y))
        finally:
            self._remove(added)

    def is_satisfiable(self, expression):
        """Whether the class expression can have an instance."""
        added = []
        try:
            x = self._name(expression, added)
            self.reasoner.flush()
            return bool(self.reasoner.isSatisfiable(x))
        finally:
            self._remove(added)

    def entails_instance(self, individual, expression):
        """Whether ``expression(individual)`` is entailed (instance checking). With a
        backend that has no instance queries this is the subsumption ``{a} ⊑ C``."""
        if not self.backend.instance_queries:
            from java.util import HashSet
            members = HashSet()
            members.add(individual)
            return self.entails_subsumption(self.factory.getOWLObjectOneOf(members),
                                            expression)
        added = []
        try:
            x = self._name(expression, added)
            self.reasoner.flush()
            return bool(self.reasoner.getInstances(x, False).containsEntity(individual))
        finally:
            self._remove(added)

    def status(self, sub, sup):
        """The :class:`~mowl.logic.LogicalStatus` of the candidate axiom ``sub ⊑ sup``:
        entailed, contradictory (adding it makes the ontology inconsistent) or
        undetermined."""
        if self.entails_subsumption(sub, sup):
            return LogicalStatus.ENTAILED
        axiom = self.factory.getOWLSubClassOfAxiom(sub, sup)
        self.manager.addAxiom(self.ontology, axiom)
        try:
            self.reasoner.flush()
            consistent = bool(self.reasoner.isConsistent())
        finally:
            self.manager.removeAxiom(self.ontology, axiom)
            self.reasoner.flush()
        return LogicalStatus.UNDETERMINED if consistent else LogicalStatus.CONTRADICTORY

    def dispose(self):
        self.reasoner.dispose()
