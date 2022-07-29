from mowl.base_models.elmodel import EmbeddingELModel

from mowl.models.elboxembeddings.module import ELBoxModule

from mowl.projection.factory import projector_factory
from mowl.projection.edge import Edge
import math
import logging
import numpy as np

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
        self.batch_size = batch_size
        self._loaded = False
        self._loaded_eval = False
        self.extended = False
        self.init_model()
                
    def init_model(self):
        self.model = ELBoxModule(
            len(self.class_index_dict),
            len(self.object_property_index_dict),
            embed_dim = self.embed_dim,
            margin = self.margin
        ).to(self.device)
    
        
    def train(self):
        _, diseases = self.dataset.evaluation_classes
        
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
                rand_index = np.random.choice(len(gci_dataset), size = self.batch_size)
                dst = self.model(gci_dataset[rand_index], gci_name)
                mse_loss = criterion(dst, th.zeros(dst.shape, requires_grad = False).to(self.device))
                loss += mse_loss
                
                if gci_name == "gci2":
                    rand_index = np.random.choice(len(gci_dataset), size = self.batch_size)
                    gci_batch = gci_dataset[rand_index]
                    dis = [self.class_index_dict[p] for p in list(diseases)]
                    idxs_for_negs = np.random.choice(len(self.class_index_dict), size = len(gci_batch), replace = True)
                    rand_dis_ids = th.tensor(idxs_for_negs).to(self.device)
                    neg_data = th.cat([gci_batch[:, :2], rand_dis_ids.unsqueeze(1)], dim = 1)
                    
                    dst = self.model(neg_data, gci_name, neg = True)
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
                
            checkpoint = 100
            if best_loss > valid_loss and (epoch+1) % checkpoint == 0:
                best_loss = valid_loss
                print("Saving model..")
                th.save(self.model.state_dict(), self.model_filepath)
            if (epoch+1) % checkpoint == 0:
                print(f'Epoch {epoch}: Train loss: {train_loss} Valid loss: {valid_loss}')



    def load_eval_data(self):
        
        if self._loaded_eval:
            return

        eval_property = self.dataset.get_evaluation_property()
        eval_classes = self.dataset.get_evaluation_classes()
        
        self._head_entities = eval_classes[0]
        self._tail_entities = eval_classes[1]


        eval_projector = projector_factory('taxonomy_rels', taxonomy=False, relations=[eval_property])

        self._training_set = eval_projector.project(self.dataset.ontology)
        self._testing_set = eval_projector.project(self.dataset.testing)
        
        self._loaded_eval = True

    def get_embeddings(self):
        self.init_model()
        
        print('Load the best model', self.model_filepath)
        self.model.load_state_dict(th.load(self.model_filepath))
        self.model.eval()

        ent_embeds = {k:v for k,v in zip(self.class_index_dict.keys(), self.model.class_embed.weight.cpu().detach().numpy())}
        rel_embeds = {k:v for k,v in zip(self.object_property_index_dict.keys(), self.model.rel_embed.weight.cpu().detach().numpy())}
        return ent_embeds, rel_embeds

    def load_best_model(self):
        self.init_model()
        self.model.load_state_dict(th.load(self.model_filepath))
        self.model.eval()

    def eval_method(self, data):
        return self.model.gci2_loss(data)

    @property
    def training_set(self):
        self.load_eval_data()
        return self._training_set

#        self.load_eval_data()

    @property
    def testing_set(self):
        self.load_eval_data()
        return self._testing_set

    @property
    def head_entities(self):
        self.load_eval_data()
        return self._head_entities

    @property
    def tail_entities(self):
        self.load_eval_data()
        return self._tail_entities
