from mowl.models import ELEmbeddings
from mowl.evaluation import PPIEvaluator
import torch as th
import numpy as np


class ELEmPPI(ELEmbeddings):
    """
    Example of ELEmbeddings for protein-protein interaction prediction.

    This model customizes negative sampling to use only protein IDs
    from the evaluation classes instead of all ontology classes.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_evaluator(PPIEvaluator)
        self._protein_ids = None

    @property
    def protein_ids(self):
        """Get protein IDs from evaluation classes (cached)."""
        if self._protein_ids is None:
            self._protein_ids = [
                self.class_index_dict[p]
                for p in self.dataset.evaluation_classes[0].as_str
            ]
        return self._protein_ids

    @property
    def evaluation_model(self):
        if self._evaluation_model is None:
            self._evaluation_model = self.module
        return self._evaluation_model

    def get_negative_sampling_config(self):
        """Only do negative sampling for gci2."""
        return {
            "gci2": {"corrupt_column": 2}
        }

    def generate_negatives(self, gci_name, gci_dataset):
        """Generate negatives using only protein IDs for gci2."""
        if gci_name != "gci2":
            return None

        data = gci_dataset[:]
        idxs_for_negs = np.random.choice(
            self.protein_ids, size=len(gci_dataset), replace=True
        )
        rand_index = th.tensor(idxs_for_negs, dtype=th.long, device=self.device)
        neg_data = th.cat([data[:, :2], rand_index.unsqueeze(1)], dim=1)
        return neg_data

    def evaluate_ppi(self):
        """Convenience method to evaluate PPI prediction."""
        self.init_module()
        print("Load the best model", self.model_filepath)
        self.load_best_model()
        with th.no_grad():
            self.evaluate(
                self.dataset.testing, filter_ontologies=[self.dataset.ontology]
            )
            print(self.metrics)
