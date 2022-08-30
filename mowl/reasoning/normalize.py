from de.tudresden.inf.lat.jcel.owlapi.main import JcelReasoner
from de.tudresden.inf.lat.jcel.ontology.normalization import OntologyNormalizer
from de.tudresden.inf.lat.jcel.ontology.axiom.extension import IntegerOntologyObjectFactoryImpl
from de.tudresden.inf.lat.jcel.owlapi.translator import ReverseAxiomTranslator
from de.tudresden.inf.lat.jcel.owlapi.translator import Translator

from uk.ac.manchester.cs.owl.owlapi import OWLClassImpl, OWLObjectSomeValuesFromImpl, OWLObjectIntersectionOfImpl
from org.semanticweb.owlapi.model import OWLAxiom

from java.util import HashSet

import logging
logging.basicConfig(level = logging.INFO)

class ELNormalizer():

    """This class wraps the normalization functionality found in the Java library :class:`Jcel`. The normalization process transforms an ontology into 7 normal forms in the description logic EL language.
    """
    
    def __init__(self):
        return

    def normalize(self, ontology):

        """Performs the normalization.
        :param ontology: Input ontology
        :type ontology: :class:`org.semanticweb.owlapi.model.OWLOntology`

        :rtype: Dictionary where the keys are labels for each normal form and the values are a list of axioms of each normal form.
        """

        #jreasoner = JcelReasoner(ontology, False)
        #root_ont = jreasoner.getRootOntology()
        root_ont = ontology
        translator = Translator(ontology.getOWLOntologyManager().getOWLDataFactory(), IntegerOntologyObjectFactoryImpl())
        #translator = jreasoner.getTranslator()
        axioms = HashSet()
        axioms.addAll(root_ont.getAxioms())
        translator.getTranslationRepository().addAxiomEntities(root_ont)

        for ont in root_ont.getImportsClosure():
            axioms.addAll(ont.getAxioms())
            translator.getTranslationRepository().addAxiomEntities(ont)

        intAxioms = translator.translateSA(axioms)

        normalizer = OntologyNormalizer()
            
        factory = IntegerOntologyObjectFactoryImpl()
        normalizedOntology = normalizer.normalize(intAxioms, factory)
        rTranslator = ReverseAxiomTranslator(translator, ontology)

        axioms_dict = {"gci0": [], "gci1": [], "gci2": [], "gci3": [], "gci0_bot": [], "gci1_bot": [], "gci3_bot": []}
        
        for ax in normalizedOntology:
            try:
                axiom = rTranslator.visit(ax)
                key, value = process_axiom(axiom)
                axioms_dict[key].append(value)
            except Exception as e:
                logging.info("Reverse translation. Ignoring axiom: %s", ax)
                logging.info(e)
        return axioms_dict
    
def process_axiom(axiom: OWLAxiom):

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
            logging.info("Processing axiom. Ignoring axiom %s", axiom)

    else:
        logging.info("Processing axiom. Ignoring axiom %s", axiom)


class GCI():
    def __init__(self, axiom):
        self._axiom = axiom
        return

    @property
    def owl_subclass(self):
        return self._axiom.getSubClass()

    @property
    def owl_superclass(self):
        return self._axiom.getSuperClass()

    @property
    def owl_axiom(self):
        return self._axiom

    
    @staticmethod
    def get_entities(gcis):
        classes = set()
        object_properties = set()

        for gci in gcis:
            new_classes, new_obj_props = gci.get_entities()
            classes |= new_classes
            object_properties |= new_obj_props
            
        return classes, object_properties
        
        
class GCI0(GCI):

    def __init__(self, axiom):
        super().__init__(axiom)
        self._subclass = None
        self._superclass = None

    @property
    def subclass(self):
        if not self._subclass:
            self._subclass = str(self.owl_subclass.toStringID()) 
        return self._subclass

    @property
    def superclass(self):
        if not self._superclass:
            self._superclass = str(self.owl_superclass.toStringID())
        return self._superclass
 
    def get_entities(self):
        return set([self.subclass, self.superclass]), set()

class GCI0_BOT(GCI0):

    def __init__(self, axiom):
        super().__init__(axiom)
        if not "owl#Nothing" in self.superclass:
            raise ValueError("Superclass in GCI0_BOT must be the bottom concept.")
        
        
class GCI1(GCI):
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
        if not self._left_subclass:
            self._process_left_side()
        return self._left_subclass

    @property
    def right_subclass(self):
        if not self._right_subclass:
            self._process_left_side()
        return self._right_subclass

    @property
    def superclass(self):
        if not self._superclass:
            self._superclass = str(self.owl_superclass.toStringID())
        return self._superclass
        
    def get_entities(self):
        return set([self.left_subclass, self.right_subclass, self.superclass]), set()
            
class GCI1_BOT(GCI1):
    def __init__(self, axiom):
        super().__init__(axiom)
        if not "owl#Nothing" in self.superclass:
            raise ValueError("Superclass in GCI1_BOT must be the bottom concept.")
        
        
class GCI2(GCI):
    def __init__(self, axiom):
        super().__init__(axiom)

        self._subclass = None
        self._object_property = None
        self._filler = None

    def _process_right_side(self):
         object_property = str(self.owl_superclass.getProperty().toString())
         filler = str(self.owl_superclass.getFiller().toStringID())

         self._object_property = object_property[1:-1] if object_property.startswith("<") else object_property
         self._filler = filler

    @property
    def subclass(self):
        if not self._subclass:
            self._subclass = str(self.owl_subclass.toStringID()) 
        return self._subclass

    @property
    def object_property(self):
        if not self._object_property:
            self._process_right_side()
        return self._object_property

    @property
    def filler(self):
        if not self._filler:
            self._process_right_side()
        return self._filler
         
    
    def get_entities(self):
        return set([self.subclass, self.filler]), set(self.object_property)

class GCI3(GCI):
    def __init__(self, axiom):
        super().__init__(axiom)

        self._object_property = None
        self._filler = None
        self._superclass = None

    def _process_left_side(self):
         object_property = str(self.owl_subclass.getProperty().toString())
         filler = str(self.owl_subclass.getFiller().toStringID())

         self._object_property = object_property[1:-1] if object_property.startswith("<") else object_property
         self._filler = filler

    @property
    def object_property(self):
        if not self._object_property:
            self._process_left_side()
        return self._object_property

    @property
    def filler(self):
        if not self._filler:
            self._process_left_side()
        return self._filler
         
    @property
    def superclass(self):
        if not self._superclass:
            self._superclass = str(self.owl_superclass.toStringID()) 
        return self._superclass

    def get_entities(self):
        return set([self.filler, self.superclass]), set(self.object_property)
                
class GCI3_BOT(GCI3):
    def __init__(self, axiom):
        super().__init__(axiom)
        if not "owl#Nothing" in superclass:
            raise ValueError("Superclass in GCI3_BOT must be the bottom concept.")
        

        
