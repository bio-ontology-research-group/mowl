from de.tudresden.inf.lat.jcel.owlapi.main import JcelReasoner
from de.tudresden.inf.lat.jcel.ontology.normalization import OntologyNormalizer
from de.tudresden.inf.lat.jcel.ontology.axiom.extension import IntegerOntologyObjectFactoryImpl
from de.tudresden.inf.lat.jcel.owlapi.translator import ReverseAxiomTranslator

from uk.ac.manchester.cs.owl.owlapi import OWLClassImpl, OWLObjectSomeValuesFromImpl, OWLObjectIntersectionOfImpl
from org.semanticweb.owlapi.model import OWLAxiom

from java.util import HashSet

import logging
logging.basicConfig(level = logging.INFO)

class ELNormalizer():
    
    
    def __init__(self):
        return

    def normalize(self, ontology):

        jreasoner = JcelReasoner(ontology, False)
        root_ont = jreasoner.getRootOntology()
        translator = jreasoner.getTranslator()
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
        operands = subclass.getOperandsAsList()
        left_subclass = operands[0].toStringID()
        right_subclass = operands[1].toStringID()
        superclass = superclass.toStringID()
        if superclass.contains("owl#Nothing"):
            return "gci1_bot", GCI1_BOT(left_subclass, right_subclass, superclass)
        
        return "gci1", GCI1(left_subclass, right_subclass, superclass)

    elif type(subclass) == OWLObjectSomeValuesFromImpl:
        obj_property = subclass.getProperty().toString()
        filler = subclass.getFiller().toStringID()
        superclass = superclass.toStringID()

        if superclass.contains("owl#Nothing"):
            return "gci3_bot", GCI3_BOT(left_subclass, right_subclass, superclass)
        
        return "gci3", GCI3(obj_property, filler, superclass)

    elif type(subclass) == OWLClassImpl:

        if type(superclass) == OWLClassImpl:
            superclass = superclass.toStringID()
            if superclass.contains("owl#Nothing"):
                return "gci0_bot", GCI0_BOT(subclass.toStringID(), superclass)
        
            return "gci0", GCI0(subclass.toStringID(), superclass)

        elif type(superclass) == OWLObjectSomeValuesFromImpl:
            obj_property = superclass.getProperty().toString()
            filler = superclass.getFiller().toStringID()
            return "gci2", GCI2(subclass.toStringID(), obj_property, filler)

        else:
            logging.info("Processing axiom. Ignoring axiom %s", axiom)

    else:
        logging.info("Processing axiom. Ignoring axiom %s", axiom)

        
class GCI0():

    def __init__(self, subclass, superclass):

        self.subclass = str(subclass)
        self.superclass = str(superclass)

class GCI0_BOT(GCI0):

    def __init__(self, subclass, superclass):

        if not superclass.contains("owl#Nothing"):
            raise ValueError("Superclass in GCI0_BOT must be the bottom concept.")

        super().__init__(subclass, superclass)


        
class GCI1():

    def __init__(self, left_subclass, right_subclass, superclass):

        self.left_subclass = str(left_subclass)
        self.right_subclass = str(right_subclass)
        self.superclass = str(superclass)

class GCI1_BOT(GCI1):

    def __init__(self, l, r, superclass):

        if not superclass.contains("owl#Nothing"):
            raise ValueError("Superclass in GCI1_BOT must be the bottom concept.")

        super().__init__(l, r, superclass)
        
class GCI2():

    def __init__(self, subclass, obj_property, filler):

        self.subclass = str(subclass)
        obj_property = str(obj_property)
        self.obj_property = obj_property[1:-1] if obj_property.startswith("<") else obj_property
        self.filler = filler
        
class GCI3():

    def __init__(self, obj_property, filler, superclass):
        obj_property = str(obj_property)
        self.obj_property = obj_property[1:-1] if obj_property.startswith("<") else obj_property
        self.filler = filler
        self.superclass = superclass
                    
class GCI3_BOT(GCI3):

    def __init__(self, o, f, superclass):

        if not superclass.contains("owl#Nothing"):
            raise ValueError("Superclass in GCI3_BOT must be the bottom concept.")

        super().__init__(o, f, superclass)

        
