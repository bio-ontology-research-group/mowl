from mowl.models import BoxSquaredEL
from mowl.evaluation import PPIEvaluator
import torch as th
import numpy as np


class BoxSquaredELPPI(BoxSquaredEL):
    """
    Example of BoxSquaredEL for protein-protein interaction prediction.

    This model customizes negative sampling to use only protein IDs
    from the evaluation classes and generates multiple negatives per positive
    using the num_negs parameter.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_evaluator(PPIEvaluator)
        self.eval_gci_name = "gci2"
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

    def get_negative_sampling_config(self):
        """Only do negative sampling for gci2."""
        return {
            "gci2": {"corrupt_column": 2}
        }

    def generate_negatives(self, gci_name, gci_dataset):
        """Generate multiple negatives per positive using protein IDs."""
        if gci_name != "gci2":
            return None

        data = gci_dataset[:]
        # Generate num_negs negatives per positive sample
        idxs_for_negs = np.random.choice(
            self.protein_ids, size=(self.num_negs, len(data)), replace=True
        )
        rand_index = th.tensor(idxs_for_negs, dtype=th.long, device=self.device)
        rand_index = rand_index.reshape(-1)

        # Repeat positive data to match negatives
        data_repeated = data.repeat(self.num_negs, 1)
        neg_data = th.cat([data_repeated[:, :2], rand_index.unsqueeze(1)], dim=1)
        return neg_data

    def evaluate_ppi(self):
        """Convenience method to evaluate PPI prediction."""
        self.init_module()
        print("Load the best model", self.model_filepath)
        self.load_best_model()
        with th.no_grad():
            metrics = self.evaluate()
            print(metrics)
