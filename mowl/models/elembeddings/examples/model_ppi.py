from mowl.base_models.elmodel import EmbeddingELModel

from mowl.models.elembeddings.evaluate import ELEmbeddingsPPIEvaluator
from mowl.projection.factory import projector_factory
from tqdm import trange, tqdm
import torch as th

import numpy as np
from mowl.models import ELEmbeddings

class ELEmPPI(ELEmbeddings):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def train(self):

        optimizer = th.optim.Adam(self.module.parameters(), lr=self.learning_rate)
        best_loss = float('inf')

        for epoch in trange(self.epochs):
            self.module.train()

            train_loss = 0
            loss = 0

            for gci_name, gci_dataset in self.training_datasets.items():
                if len(gci_dataset) == 0:
                    continue

                loss += th.mean(self.module(gci_dataset[:], gci_name))
                if gci_name == "gci2":
                    prots = [self.class_index_dict[p] for p
                             in self.dataset.evaluation_classes.as_str]
                    idxs_for_negs = np.random.choice(prots, size=len(gci_dataset), replace=True)
                    rand_index = th.tensor(idxs_for_negs).to(self.device)
                    data = gci_dataset[:]
                    neg_data = th.cat([data[:, :2], rand_index.unsqueeze(1)], dim=1)
                    loss += th.mean(self.module(neg_data, gci_name, neg=True))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.detach().item()

            loss = 0
            with th.no_grad():
                self.module.eval()
                valid_loss = 0
                gci2_data = self.validation_datasets["gci2"][:]
                loss = th.mean(self.module(gci2_data, "gci2"))
                valid_loss += loss.detach().item()

            checkpoint = 100
            if best_loss > valid_loss:
                best_loss = valid_loss
                th.save(self.module.state_dict(), self.model_filepath)
            if (epoch + 1) % checkpoint == 0:
                print(f'Epoch {epoch}: Train loss: {train_loss} Valid loss: {valid_loss}')

        return 1

    def eval_method(self, data):
        return self.module.gci2_loss(data)

    def evaluate_ppi(self):
        self.init_module()
        print('Load the best model', self.model_filepath)
        self.load_best_model()
        with th.no_grad():
            eval_method = self.module.gci2_loss

            evaluator = ELEmbeddingsPPIEvaluator(
                self.dataset.testing, eval_method, self.dataset.ontology, self.class_index_dict,
                self.object_property_index_dict, device=self.device)
            evaluator()
            evaluator.print_metrics()

