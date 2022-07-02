from org.mowl.Normalizers import OntologyNormalizer1 as ON1
from java.lang import String

class OntologyNormalizer1():
    def __init__(self):
        self.normalizer = ON1()

    def normalize(self, ontology):
        norms = self.normalizer.normalize(ontology)

        new_norms = dict()
        new_norms["gci_type_1"] = [GCIType1(x) for x in norms["gci_type_1"]] if "gci_type_1" in norms else []
        new_norms["gci_type_2"] = [GCIType2(x) for x in norms["gci_type_2"]] if "gci_type_2" in norms else []
        new_norms["gci_type_3"] = [GCIType3(x) for x in norms["gci_type_3"]] if "gci_type_3" in norms else []
        new_norms["gci_type_4"] = [GCIType4(x) for x in norms["gci_type_4"]] if "gci_type_4" in norms else []
        new_norms["gci_type_5"] = [GCIType5(x) for x in norms["gci_type_5"]] if "gci_type_5" in norms else []
        new_norms["gci_type_6"] = [GCIType6(x) for x in norms["gci_type_6"]] if "gci_type_6" in norms else []


        return new_norms

class GCIType1():
    def __init__(self, gci):
        self.subclass = str(gci.subclass())
        self.superclass = str(gci.superclass())

class GCIType2():
    def __init__(self, gci):
        self.subclass = str(gci.subclass())
        self.obj_property = str(gci.obj_property())
        self.filler = str(gci.filler())

class GCIType3():
    def __init__(self, gci):
        self.obj_property = str(gci.obj_property())
        self.filler = str(gci.filler())
        self.superclass = str(gci.superclass())

class GCIType4():
    def __init__(self, gci):
        self.left_subclass = str(gci.left_subclass())
        self.right_subclass = str(gci.right_subclass())
        self.superclass = str(gci.superclass())

class GCIType5():
    def __init__(self, gci):
        self.left_superclass = str(gci.left_superclass())
        self.right_superclass = str(gci.right_superclass())
        self.subclass = str(gci.subclass())

class GCIType6():
    def __init__(self, gci):
        self.left_subclass = str(gci.left_subclass())
        self.obj_property = str(gci.obj_property())
        self.filler = str(gci.filler())
        self.superclass = str(gci.superclass())
                
