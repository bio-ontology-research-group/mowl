from mowl.base_models.elmodel import EmbeddingELModel
import mowl.models.elboxembeddings.losses as L
from mowl.nn.elmodule import ELModule
from mowl.projection.factory import projector_factory
from mowl.projection.edge import Edge
import math
import logging
import numpy as np

from .evaluate import ELBoxEmbeddingsPPIEvaluator

from tqdm import trange, tqdm

import torch as th
from torch import nn

class ELBoxEmbeddings(EmbeddingELModel):

    def __init__(self,
                 dataset,
                 embed_dim=50,
                 margin=0,
                 reg_norm=1,
                 learning_rate=0.001,
                 epochs=1000,
                 batch_size = 4096*8,
                 model_filepath = None,
                 device = 'cpu'
                 ):
        super().__init__(dataset, batch_size, extended = True)


        self.embed_dim = embed_dim
        self.margin = margin
        self.reg_norm = reg_norm
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.device = device
        self.model_filepath = model_filepath
        self._loaded = False
        self._loaded_eval = False
        self.extended = False

                
    def init_model(self):
        self.model = ELBoxModule(
            len(self.class_index_dict),
            len(self.object_property_index_dict),
            embed_dim = self.embed_dim,
            margin = self.margin
        ).to(self.device)
    
        
    def train(self):
        self.init_model()
        criterion = nn.MSELoss()
        optimizer = th.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        best_loss = float('inf')

        training_datasets = {k: v.data for k,v in self.training_datasets.get_gci_datasets().items()}
        validation_dataset = self.validation_datasets.get_gci_datasets()["gci2"][:]

        for epoch in trange(self.epochs):
            self.model.train()

            train_loss = 0
            loss = 0

            
            for gci_name, gci_dataset in training_datasets.items():
                if len(gci_dataset) == 0:
                    continue
                rand_index = np.random.choice(len(gci_dataset), size = 512)
                dst = self.model(gci_dataset[rand_index], gci_name)
                mse_loss = criterion(dst, th.zeros(dst.shape, requires_grad = False).to(self.device))
                loss += mse_loss
                
                if gci_name == "gci2":
                    rand_index = np.random.choice(len(gci_dataset), size = 512)
                    dst = self.model(gci_dataset[rand_index], gci_name, neg = True)
                    mse_loss = criterion(dst, th.ones(dst.shape, requires_grad = False).to(self.device))
                    loss += mse_loss
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.detach().item()

            with th.no_grad():
                self.model.eval()
                valid_loss = 0
                gci2_data = validation_dataset

                dst = self.model(gci2_data, "gci2")
                loss = criterion(dst, th.zeros(dst.shape, requires_grad = False).to(self.device))
                valid_loss += loss.detach().item()
                
            checkpoint = 1000
            if best_loss > valid_loss and (epoch+1) % checkpoint == 0:
                best_loss = valid_loss
                print("Saving model..")
                th.save(self.model.state_dict(), self.model_filepath)
            if (epoch+1) % checkpoint == 0:
                print(f'Epoch {epoch}: Train loss: {train_loss} Valid loss: {valid_loss}')

    def evaluate_ppi(self):
        self.init_model()
        print('Load the best model', self.model_filepath)
        self.model.load_state_dict(th.load(self.model_filepath))
        with th.no_grad():
            self.model.eval()

            eval_method = self.model.gci2_loss

            evaluator = ELBoxEmbeddingsPPIEvaluator(self.dataset.testing, eval_method, self.dataset.ontology, self.class_index_dict, self.object_property_index_dict, device = self.device)
            evaluator()
            evaluator.print_metrics()

    
class ELBoxModule(ELModule):

    def __init__(self, nb_ont_classes, nb_rels, embed_dim=50, margin=0.1):
        super().__init__()
        self.nb_ont_classes = nb_ont_classes
        self.nb_rels = nb_rels

        self.embed_dim = embed_dim
        
        self.class_embed = nn.Embedding(self.nb_ont_classes, embed_dim)
        nn.init.uniform_(self.class_embed.weight, a=-1, b=1)
        self.class_embed.weight.data /= th.linalg.norm(self.class_embed.weight.data,axis=1).reshape(-1,1)

        self.class_offset = nn.Embedding(self.nb_ont_classes, embed_dim)
        nn.init.uniform_(self.class_offset.weight, a=-1, b=1)
        self.class_offset.weight.data /= th.linalg.norm(self.class_offset.weight.data,axis=1).reshape(-1,1)

        self.rel_embed = nn.Embedding(nb_rels, embed_dim)
        nn.init.uniform_(self.rel_embed.weight, a=-1, b=1)
        self.rel_embed.weight.data /= th.linalg.norm(self.rel_embed.weight.data,axis=1).reshape(-1,1)
        
        self.margin = margin

    def gci0_loss(self, data, neg = False):
        return L.gci0_loss(data, self.class_embed, self.class_offset, self.margin, neg = neg)
    def gci1_loss(self, data, neg = False):
        return L.gci1_loss(data, self.class_embed, self.class_offset, self.margin, neg = neg)
    def gci1_bot_loss(self, data, neg = False):
        return L.gci1_bot_loss(data, self.class_embed, self.class_offset, self.margin, neg = neg)
    def gci2_loss(self, data, neg = False):
        return L.gci2_loss(data, self.class_embed, self.class_offset, self.rel_embed, self.margin, neg = neg)
    def gci3_loss(self, data, neg = False):
        return L.gci3_loss(data, self.class_embed, self.class_offset, self.rel_embed, self.margin, neg = neg)
