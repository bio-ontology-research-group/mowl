from mowl.models import ELBE

from mowl.projection.factory import projector_factory
from mowl.projection.edge import Edge
import math
import logging
import numpy as np

from tqdm import trange, tqdm

import torch as th
from torch import nn


class ELBEGDA(ELBE):
    """
    Example of ELBoxEmbeddings for gene-disease associations prediction.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def train(self):
        _, diseases = self.dataset.evaluation_classes

        criterion = nn.MSELoss()
        optimizer = th.optim.Adam(self.module.parameters(), lr=self.learning_rate)
        best_loss = float('inf')

        training_datasets = {k: v.data for k, v in
                             self.training_datasets.items()}
        validation_dataset = self.validation_datasets["gci2"][:]

        for epoch in trange(self.epochs):
            self.module.train()

            train_loss = 0
            loss = 0

            for gci_name, gci_dataset in training_datasets.items():
                if len(gci_dataset) == 0:
                    continue
                rand_index = np.random.choice(len(gci_dataset), size=self.batch_size)
                dst = self.module(gci_dataset[rand_index], gci_name)
                mse_loss = criterion(dst, th.zeros(dst.shape, requires_grad=False).to(self.device))
                loss += mse_loss

                if gci_name == "gci2":
                    rand_index = np.random.choice(len(gci_dataset), size=self.batch_size)
                    gci_batch = gci_dataset[rand_index]
                    idxs_for_negs = np.random.choice(len(self.class_index_dict),
                                                     size=len(gci_batch), replace=True)
                    rand_dis_ids = th.tensor(idxs_for_negs).to(self.device)
                    neg_data = th.cat([gci_batch[:, :2], rand_dis_ids.unsqueeze(1)], dim=1)

                    dst = self.module(neg_data, gci_name, neg=True)
                    mse_loss = criterion(dst, th.ones(dst.shape,
                                                      requires_grad=False).to(self.device))
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

            checkpoint = 100
            if best_loss > valid_loss and (epoch + 1) % checkpoint == 0:
                best_loss = valid_loss
                print("Saving model..")
                th.save(self.module.state_dict(), self.model_filepath)
            if (epoch + 1) % checkpoint == 0:
                print(f'Epoch {epoch}: Train loss: {train_loss} Valid loss: {valid_loss}')

        return 1

    
