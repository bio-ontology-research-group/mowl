import mowl
from mowl.datasets import PathDataset
from mowl.datasets.builtin import PPIYeastSlimDataset
from mowl.datasets.base import OWLClasses
import os

class SubsumptionDataset(PathDataset):
    def __init__(self, root_dir):
        super().__init__(root_dir + "train.owl", root_dir + "valid.owl", root_dir + "test.owl")

        self.root_dir = root_dir
        self._deductive_closure_ontology = None
        
    @property
    def deductive_closure_ontology(self):
        if self._deductive_closure_ontology is None:
            self._deductive_closure_ontology = PathDataset(self.root_dir + "train_deductive_closure.owl").ontology

        return self._deductive_closure_ontology

    @property
    def evaluation_classes(self):
        if self._evaluation_classes is None:

            train_classes = self.ontology.getClassesInSignature()
            valid_classes = self.validation.getClassesInSignature()
            test_classes = self.testing.getClassesInSignature()

            assert set(valid_classes) - set(train_classes) == set(), f"Valid classes not in train: {set(valid_classes) - set(train_classes)}"
            assert set(test_classes) - set(train_classes) == set(), f"Test classes not in train: {set(test_classes) - set(train_classes)}"
            
            classes = self.ontology.getClassesInSignature()

            bot_in_classes = False
            top_in_classes = False

            for cls in classes:
                if cls.isOWLNothing():
                    bot_in_classes = True
                    continue
                if cls.isOWLThing():
                    top_in_classes = True
                    continue

            if not bot_in_classes:
                print("Did not find owl:Nothing in ontology classes. Adding it")
                classes.add(self.ontology.getOWLOntologyManager().getOWLDataFactory().getOWLNothing())
            if not top_in_classes:
                print("Did not find owl:Thing in classes. Adding it")
                classes.add(self.ontology.getOWLOntologyManager().getOWLDataFactory().getOWLThing())

            
                

            
            classes = OWLClasses(classes)
            self._evaluation_classes = classes, classes

        return self._evaluation_classes


class PPIDataset(PathDataset):
    def __init__(self, root_dir):
        super().__init__(root_dir + "ontology.owl", root_dir + "valid.owl", root_dir + "test.owl")

        self.root_dir = root_dir
        self._deductive_closure_ontology = None
        
    @property
    def evaluation_classes(self):
        if self._evaluation_classes is None:
            proteins = set()
            for owl_name, owl_cls in self.classes.as_dict.items():
                if "http://4932" in owl_name:
                    proteins.add(owl_cls)
            self._evaluation_classes = OWLClasses(proteins), OWLClasses(proteins)

        return self._evaluation_classes

class PPIDatasetV2(PPIDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._deductive_closure_ontology = None
        
    @property
    def deductive_closure_ontology(self):
        if self._deductive_closure_ontology is None:
            root_dir = os.path.dirname(os.path.abspath(self.ontology_path))
            ontology_path = os.path.join(root_dir, "ontology_deductive_closure.owl")
            self._deductive_closure_ontology = PathDataset(ontology_path).ontology

        return self._deductive_closure_ontology




