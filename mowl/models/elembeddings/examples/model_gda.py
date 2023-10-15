from mowl.base_models.elmodel import EmbeddingELModel

from tqdm import trange, tqdm
import torch as th

import numpy as np
from mowl.models import ELEmbeddings

class ELEmGDA(ELEmbeddings):
    """
    Example of ELEmbeddings for gene-disease associations prediction.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
                                                                                        
    
    def train(self):
        _, diseases = self.dataset.evaluation_classes

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
                    idxs_for_negs = np.random.choice(len(self.class_index_dict),
                                                     size=len(gci_dataset), replace=True)
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

            checkpoint = 10
            if best_loss > valid_loss:
                best_loss = valid_loss
                th.save(self.module.state_dict(), self.model_filepath)
            if (epoch + 1) % (checkpoint * 10) == 0:
                print(f'Epoch {epoch}: Train loss: {train_loss} Valid loss: {valid_loss}')

        return 1
