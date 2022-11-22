from de.tudresden.inf.lat.jcel.ontology.normalization import OntologyNormalizer
from de.tudresden.inf.lat.jcel.ontology.axiom.extension import IntegerOntologyObjectFactoryImpl
from de.tudresden.inf.lat.jcel.owlapi.translator import ReverseAxiomTranslator
from de.tudresden.inf.lat.jcel.owlapi.translator import Translator
from org.semanticweb.owlapi.model.parameters import Imports
from uk.ac.manchester.cs.owl.owlapi import OWLClassImpl, OWLObjectSomeValuesFromImpl, \
    OWLObjectIntersectionOfImpl
from org.semanticweb.owlapi.model import OWLAxiom, OWLOntology

from java.util import HashSet

import logging
logging.basicConfig(level=logging.INFO)
from mowl.owlapi import OWLAPIAdapter


class ELNormalizer():

    """This class wraps the normalization functionality found in the Java library :class:`Jcel`. \
The normalization process transforms an ontology into 7 normal forms in the description \
logic EL language.
    """

    def __init__(self):
        return

    def normalize(self, ontology):
        """Performs the normalization.
        :param ontology: Input ontology
        :type ontology: :class:`org.semanticweb.owlapi.model.OWLOntology`

        :rtype: Dictionary where the keys are labels for each normal form and the values are a \
            list of axioms of each normal form.
        """

        # Type check
        if not isinstance(ontology, OWLOntology):
            raise TypeError(f"Parameter 'ontology' must be of \
type org.semanticweb.owlapi.model.OWLOntology. Found: {type(ontology)}")

        # jreasoner = JcelReasoner(ontology, False)
        # root_ont = jreasoner.getRootOntology()
        ontology = self.preprocess_ontology(ontology)
        root_ont = ontology
        translator = Translator(ontology.getOWLOntologyManager().getOWLDataFactory(),
                                IntegerOntologyObjectFactoryImpl())
        # translator = jreasoner.getTranslator()
        axioms = HashSet()
        axioms.addAll(root_ont.getAxioms())
        translator.getTranslationRepository().addAxiomEntities(root_ont)

        for ont in root_ont.getImportsClosure():
            axioms.addAll(ont.getAxioms())
            translator.getTranslationRepository().addAxiomEntities(ont)

        intAxioms = translator.translateSA(axioms)

        normalizer = OntologyNormalizer()

        factory = IntegerOntologyObjectFactoryImpl()
        normalized_ontology = normalizer.normalize(intAxioms, factory)
        self.rTranslator = ReverseAxiomTranslator(translator, ontology)

        axioms_dict = self.__revert_translation(normalized_ontology)

        return axioms_dict

    def __revert_translation(self, normalized_ontology):
        axioms_dict = {
            "gci0": [], "gci1": [], "gci2": [], "gci3": [], "gci0_bot": [], "gci1_bot": [],
            "gci3_bot": []}

        for ax in normalized_ontology:
            try:
                axiom = self.rTranslator.visit(ax)
                key, value = process_axiom(axiom)
                axioms_dict[key].append(value)
            except Exception as e:
                logging.info("Reverse translation. Ignoring axiom: %s", ax)
                logging.info(e)

        return axioms_dict

    # TODO: This method is missing unit tests
    def preprocess_ontology(self, ontology):
        """Preprocesses the ontology to remove axioms that are not supported by the normalization \
            process.

        :param ontology: Input ontology
        :type ontology: :class:`org.semanticweb.owlapi.model.OWLOntology`

        :rtype: :class:`org.semanticweb.owlapi.model.OWLOntology`
        """

        # Type check
        if not isinstance(ontology, OWLOntology):
            raise TypeError("Parameter 'ontology' must be of \
type org.semanticweb.owlapi.model.OWLOntology")

        tbox_axioms = ontology.getTBoxAxioms(Imports.fromBoolean(True))
        new_tbox_axioms = HashSet()
        for axiom in tbox_axioms:
            axiom_as_str = axiom.toString()

            if "UnionOf" in axiom_as_str:
                continue
            elif "MinCardinality" in axiom_as_str:
                continue
            elif "ComplementOf" in axiom_as_str:
                continue
            elif "AllValuesFrom" in axiom_as_str:
                continue
            elif "MaxCardinality" in axiom_as_str:
                continue
            elif "ExactCardinality" in axiom_as_str:
                continue
            elif "Annotation" in axiom_as_str:
                continue
            elif "ObjectHasSelf" in axiom_as_str:
                continue
            elif "urn:swrl" in axiom_as_str:
                continue
            elif "EquivalentObjectProperties" in axiom_as_str:
                continue
            elif "SymmetricObjectProperty" in axiom_as_str:
                continue
            elif "AsymmetricObjectProperty" in axiom_as_str:
                continue
            elif "ObjectOneOf" in axiom_as_str:
                continue
            else:
                new_tbox_axioms.add(axiom)

        owl_manager = OWLAPIAdapter().owl_manager
        new_ontology = owl_manager.createOntology(new_tbox_axioms)
        return new_ontology


def process_axiom(axiom: OWLAxiom):

    # Type check
    if not isinstance(axiom, OWLAxiom):
        raise TypeError("Parameter 'axiom' must be of type org.semanticweb.owlapi.model.OWLAxiom")

    subclass = axiom.getSubClass()
    superclass = axiom.getSuperClass()

    if type(subclass) == OWLObjectIntersectionOfImpl:
        superclass = superclass.toStringID()
        if superclass.contains("owl#Nothing"):
            return "gci1_bot", GCI1_BOT(axiom)
        return "gci1", GCI1(axiom)

    elif type(subclass) == OWLObjectSomeValuesFromImpl:
        superclass = superclass.toStringID()
        if superclass.contains("owl#Nothing"):
            return "gci3_bot", GCI3_BOT(axiom)

        return "gci3", GCI3(axiom)

    elif type(subclass) == OWLClassImpl:

        if type(superclass) == OWLClassImpl:
            superclass = superclass.toStringID()
            if superclass.contains("owl#Nothing"):
                return "gci0_bot", GCI0_BOT(axiom)

            return "gci0", GCI0(axiom)

        elif type(superclass) == OWLObjectSomeValuesFromImpl:
            return "gci2", GCI2(axiom)

        else:
            logging.info("Superclass type not recognized. Ignoring axiom: %s", axiom)

    else:
        logging.info("Subclass type not recognized. Ignoring axiom: %s", axiom)


class GCI():
    """Base class for all GCI types in the :math:`\\mathcal{EL}` language"""

    def __init__(self, axiom):
        self._axiom = axiom
        return

    def __eq__(self, other):
        return self._axiom.equals(other._axiom)

    @property
    def owl_subclass(self):
        """Returns the subclass of the GCI

        :rtype: :class:`org.semanticweb.owlapi.model.OWLClassExpression`
        """
        return self._axiom.getSubClass()

    @property
    def owl_superclass(self):
        """Returns the superclass of the GCI

        :rtype: :class:`org.semanticweb.owlapi.model.OWLClassExpression`
        """
        return self._axiom.getSuperClass()

    @property
    def owl_axiom(self):
        """Returns the axiom of the GCI

        :rtype: :class:`org.semanticweb.owlapi.model.OWLAxiom`
        """
        return self._axiom

    @staticmethod
    def get_entities(gcis):
        """Returns all the classes and object properties that appear in the GCIs

        :param gcis: List of GCIs
        :type gcis: list

        :rtype: tuple(set, set)
        """

        classes = set()
        object_properties = set()

        for gci in gcis:
            new_classes, new_obj_props = gci.get_entities()
            classes |= new_classes
            object_properties |= new_obj_props

        return classes, object_properties


class GCI0(GCI):
    """
    GCI of the form :math:`C \\sqsubseteq D`

    :param axiom: Axiom of the form :math:`C \\sqsubseteq D`
    :type axiom: :class:`org.semanticweb.owlapi.model.OWLAxiom`
    """

    def __init__(self, axiom):
        super().__init__(axiom)
        self._subclass = None
        self._superclass = None

    @property
    def subclass(self):
        """Returns the subclass of the GCI: :math:`C`

        :rtype: :class:`org.semanticweb.owlapi.model.OWLClass`
        """
        if not self._subclass:
            self._subclass = str(self.owl_subclass.toStringID())
        return self._subclass

    @property
    def superclass(self):
        """"
        Returns the superclass of the GCI: :math:`D` or :math:`\\bot`

        :rtype: :class:`org.semanticweb.owlapi.model.OWLClass`
        """

        if not self._superclass:
            self._superclass = str(self.owl_superclass.toStringID())
        return self._superclass

    def get_entities(self):
        return set([self.subclass, self.superclass]), set()


class GCI0_BOT(GCI0):
    """
    GCI of the form :math:`C \\sqsubseteq \\bot`

    :param axiom: Axiom of the form :math:`C \\sqsubseteq \\bot`
    :type axiom: :class:`org.semanticweb.owlapi.model.OWLAxiom`
    """

    def __init__(self, axiom):
        super().__init__(axiom)
        if "owl#Nothing" not in self.superclass:
            raise ValueError("Superclass in GCI0_BOT must be the bottom concept.")


class GCI1(GCI):
    """
    GCI of the form :math:`C \\sqcap D \\sqsubseteq E`

    :param axiom: Axiom of the form :math:`C \\sqcap D \\sqsubseteq E`
    :type axiom: :class:`org.semanticweb.owlapi.model.OWLAxiom`
    """

    def __init__(self, axiom):
        super().__init__(axiom)

        self._left_subclass = None
        self._right_subclass = None
        self._superclass = None

    def _process_left_side(self):
        operands = self.owl_subclass.getOperandsAsList()
        left_subclass = operands[0].toStringID()
        right_subclass = operands[1].toStringID()

        self._left_subclass = str(left_subclass)
        self._right_subclass = str(right_subclass)

    @property
    def left_subclass(self):
        """"
        Returns the left operand of the subclass of the GCI: :math:`C`

        :rtype: :class:`org.semanticweb.owlapi.model.OWLClass`
        """
        if not self._left_subclass:
            self._process_left_side()
        return self._left_subclass

    @property
    def right_subclass(self):
        """
        Returns the right operand of the subclass of the GCI: :math:`D`

        :rtype: :class:`org.semanticweb.owlapi.model.OWLClass`
        """
        if not self._right_subclass:
            self._process_left_side()
        return self._right_subclass

    @property
    def superclass(self):
        """
        Returns the superclass of the GCI: :math:`E` or :math:`\\bot`

        :rtype: :class:`org.semanticweb.owlapi.model.OWLClass`
        """

        if not self._superclass:
            self._superclass = str(self.owl_superclass.toStringID())
        return self._superclass

    def get_entities(self):
        return set([self.left_subclass, self.right_subclass, self.superclass]), set()


class GCI1_BOT(GCI1):
    """
    GCI of the form :math:`C \\sqcap D \\sqsubseteq \\bot`

    :param axiom: Axiom of the form :math:`C \\sqcap D \\sqsubseteq \\bot`
    :type axiom: :class:`org.semanticweb.owlapi.model.OWLAxiom`
    """

    def __init__(self, axiom):
        super().__init__(axiom)
        if "owl#Nothing" not in self.superclass:
            raise ValueError("Superclass in GCI1_BOT must be the bottom concept.")


class GCI2(GCI):
    """
    GCI of the form :math:`C \\sqsubseteq \\exists R.D`

    :param axiom: Axiom of the form :math:`C \\sqsubseteq \\exists R.D`
    :type axiom: :class:`org.semanticweb.owlapi.model.OWLAxiom`
    """

    def __init__(self, axiom):
        super().__init__(axiom)

        self._subclass = None
        self._object_property = None
        self._filler = None

    def _process_right_side(self):
        object_property = str(self.owl_superclass.getProperty().toString())
        filler = str(self.owl_superclass.getFiller().toStringID())

        self._object_property = object_property[1:-1] if object_property.startswith("<") \
            else object_property
        self._filler = filler

    @property
    def subclass(self):
        """
        Returns the subclass of the GCI: :math:`C`

        :rtype: :class:`org.semanticweb.owlapi.model.OWLClass`
        """

        if not self._subclass:
            self._subclass = str(self.owl_subclass.toStringID())
        return self._subclass

    @property
    def object_property(self):
        """
        Returns the object property of the GCI: :math:`R`

        :rtype: :class:`org.semanticweb.owlapi.model.OWLObjectProperty`
        """

        if not self._object_property:
            self._process_right_side()
        return self._object_property

    @property
    def filler(self):
        """
        Returns the filler of the GCI: :math:`D`

        :rtype: :class:`org.semanticweb.owlapi.model.OWLClass`
        """

        if not self._filler:
            self._process_right_side()
        return self._filler

    def get_entities(self):
        return set([self.subclass, self.filler]), set([self.object_property])


class GCI3(GCI):
    """
    GCI of the form :math:`\\exists R.C \\sqsubseteq D`

    :param axiom: Axiom of the form :math:`\\exists R.C \\sqsubseteq D`
    :type axiom: :class:`org.semanticweb.owlapi.model.OWLAxiom`
    """

    def __init__(self, axiom):
        super().__init__(axiom)

        self._object_property = None
        self._filler = None
        self._superclass = None

    def _process_left_side(self):
        object_property = str(self.owl_subclass.getProperty().toString())
        filler = str(self.owl_subclass.getFiller().toStringID())

        self._object_property = object_property[1:-1] if object_property.startswith("<") \
            else object_property
        self._filler = filler

    @property
    def object_property(self):
        """
        Returns the object property of the GCI: :math:`R`

        :rtype: :class:`org.semanticweb.owlapi.model.OWLObjectProperty`
        """

        if not self._object_property:
            self._process_left_side()
        return self._object_property

    @property
    def filler(self):
        """
        Returns the filler of the GCI: :math:`C`

        :rtype: :class:`org.semanticweb.owlapi.model.OWLClass`
        """
        if not self._filler:
            self._process_left_side()
        return self._filler

    @property
    def superclass(self):
        """
        Returns the superclass of the GCI: :math:`D` or :math:`\\bot`

        :rtype: :class:`org.semanticweb.owlapi.model.OWLClass`
        """

        if not self._superclass:
            self._superclass = str(self.owl_superclass.toStringID())
        return self._superclass

    def get_entities(self):
        return set([self.filler, self.superclass]), set([self.object_property])


class GCI3_BOT(GCI3):
    """
    GCI of the form :math:`\\exists R.C \\sqsubseteq \\bot`

    :param axiom: Axiom of the form :math:`\\exists R.C \\sqsubseteq \\bot`
    :type axiom: :class:`org.semanticweb.owlapi.model.OWLAxiom`
    """

    def __init__(self, axiom):
        super().__init__(axiom)
        if "owl#Nothing" not in self.superclass:
            raise ValueError("Superclass in GCI3_BOT must be the bottom concept.")
