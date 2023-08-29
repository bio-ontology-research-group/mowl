from mowl.models import ELBoxEmbeddings
from mowl.projection.factory import projector_factory
from mowl.projection.edge import Edge
import math
import logging
import numpy as np

from mowl.models.elboxembeddings.evaluate import ELBoxEmbeddingsPPIEvaluator

from tqdm import trange, tqdm

import torch as th
from torch import nn


class ELBoxPPI(ELBoxEmbeddings):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
                                                                                            
    def train(self):
        criterion = nn.MSELoss()
        optimizer = th.optim.Adam(self.module.parameters(), lr=self.learning_rate)
        best_loss = float('inf')

        training_datasets = {
            k: v.data for k, v in self.training_datasets.items()}
        validation_dataset = self.validation_datasets["gci2"][:]

        for epoch in trange(self.epochs):
            self.module.train()

            train_loss = 0
            loss = 0
            for gci_name, gci_dataset in training_datasets.items():
                if len(gci_dataset) == 0:
                    continue
                rand_index = np.random.choice(len(gci_dataset), size=512)
                dst = self.module(gci_dataset[rand_index], gci_name)
                mse_loss = criterion(dst, th.zeros(dst.shape, requires_grad=False).to(self.device))
                loss += mse_loss

                if gci_name == "gci2":
                    rand_index = np.random.choice(len(gci_dataset), size=512)
                    gci_batch = gci_dataset[rand_index]
                    prots = [self.class_index_dict[p] for p
                             in self.dataset.evaluation_classes.as_str]
                    idxs_for_negs = np.random.choice(prots, size=len(gci_batch), replace=True)
                    rand_prot_ids = th.tensor(idxs_for_negs).to(self.device)
                    neg_data = th.cat([gci_batch[:, :2], rand_prot_ids.unsqueeze(1)], dim=1)

                    dst = self.module(neg_data, gci_name, neg=True)
                    mse_loss = criterion(dst,
                                         th.ones(dst.shape, requires_grad=False).to(self.device))
                    loss += mse_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.detach().item()

            with th.no_grad():
                self.module.eval()
                valid_loss = 0
                gci2_data = validation_dataset

                dst = self.module(gci2_data, "gci2")
                loss = criterion(dst, th.zeros(dst.shape, requires_grad=False).to(self.device))
                valid_loss += loss.detach().item()

            if best_loss > valid_loss:
                best_loss = valid_loss
                print("Saving model..")
                th.save(self.module.state_dict(), self.model_filepath)
            print(f'Epoch {epoch}: Train loss: {train_loss} Valid loss: {valid_loss}')

        return 1

    def evaluate_ppi(self):
        self.init_module()
        print('Load the best model', self.model_filepath)
        self.load_best_model()
        with th.no_grad():
            eval_method = self.module.gci2_loss

            evaluator = ELBoxEmbeddingsPPIEvaluator(
                self.dataset.testing, eval_method, self.dataset.ontology, self.class_index_dict,
                self.object_property_index_dict, device=self.device)
            evaluator()
            evaluator.print_metrics()


            
