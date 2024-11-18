from ..base import RemoteDataset, OWLClasses
from deprecated.sphinx import versionadded
import os

import java
from org.semanticweb.owlapi.apibinding import OWLManager



GO_DATA_URL = 'https://bio2vec.net/data/mowl/go_subsumption.tar.gz'
FOODON_DATA_URL = 'https://bio2vec.net/data/mowl/foodon_subsumption.tar.gz'

@versionadded(version='1.0.0')
class SubsumptionDataset(RemoteDataset):
    """
    Dataset for subsumption reasoning tasts of axioms :math:`A \sqsubseteq B` where :math:`A` and :math:`B` are concept names.
    """
    
    def __init__(self, url):
        super().__init__(url)

        self.dataset_dir = os.path.basename(self.ontology_path)
        self.deductive_closure_ontology_path = os.path.join(self.dataset_dir,
                                                            "ontology_deductive_closure.owl")
        self._deductive_closure_ontology = None

    @property
    def deductive_closure_ontology(self):
        if self._deductive_closure_ontology is None:
            ont_manager = OWLManager.createOWLOntologyManager()
            ontology = ont_manager.loadOntologyFromOntologyDocument(
                java.io.File(self.deductive_closure_ontology_path))
            self._deductive_closure_ontology = ontology

        return self._deductive_closure_ontology

        
    # @property
    # def deductive_closure_ontology(self):
        # if self._deductive_closure_ontology is None:
            # self._deductive_closure_ontology = PathDataset(self.root_dir + "train_deductive_closure.owl").ontology

        # return self._deductive_closure_ontology

    @property
    def evaluation_classes(self):
        if self._evaluation_classes is None:

            train_classes = self.ontology.getClassesInSignature()
            valid_classes = self.validation.getClassesInSignature()
            test_classes = self.testing.getClassesInSignature()
            deductive_closure_classes = self.deductive_closure_ontology.getClassesInSignature()
            assert set(valid_classes) - set(train_classes) == set(), f"Valid classes not in train: {set(valid_classes) - set(train_classes)}"
            assert set(test_classes) - set(train_classes) == set(), f"Test classes not in train: {set(test_classes) - set(train_classes)}"
            assert set(deductive_closure_classes) - set(train_classes) == set(), f"Deductive closure classes not in train: {set(deductive_closure_classes) - set(train_classes)}"
            
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


class GOSubsumptionDataset(SubsumptionDataset):
    """
    Dataset for subsumption prediction in the Gene Ontology. Axioms to be predicted are of the form :math:`A \sqsubseteq B` where :math:`A` and :math:`B` are GO terms. This dataset is based on [chen2020b]_.
    """
    def __init__(self):
        super().__init__(GO_DATA_URL)


class FoodOnSubsumptionDataset(SubsumptionDataset):
    """
    Dataset for subsumption prediction in the Gene Ontology. Axioms to be predicted are of the form :math:`A \sqsubseteq B` where :math:`A` and :math:`B` are GO terms. This dataset is based on [chen2020b]_.
    """
    def __init__(self):
        super().__init__(FOODON_DATA_URL)
