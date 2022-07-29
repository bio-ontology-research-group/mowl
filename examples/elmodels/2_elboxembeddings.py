"""
ELBoxEmbeddings 
===========================

This example is based on the paper `Description Logic EL++ Embeddings with Intersectional Closure <https://arxiv.org/abs/2202.14018v1>`_. This paper is based on the idea of :doc:`/examples/elmodels/1_elembeddings`, but in this work the main point is to solve the *intersectional closure* problem. 

In the case of :doc:`/examples/elmodels/1_elembeddings`, the geometric objects representing ontology classes are :math:`n`-dimensional balls. One of the normal forms in EL is:

.. math::
   C_1 \sqcap C_2 \sqsubseteq D

As we can see, there is an intersection operation :math:`C_1 \sqcap C_2`. Computing this intersection using balls is not a closed operations because the region contained in the intersection of two balls is not a ball. To solve that issue, this paper proposes the idea of changing the geometric objects to boxes, for which the intersection operation has the closure property.
"""

# %%
# This example is quite similar to the one found in :doc:`/examples/elmodels/1_elembeddings`. There might be slight changes in the training part but the most important changes are in the `Definition of loss functions`_ definition of the loss functions for each normal form.

from mowl.base_models.elmodel import EmbeddingELModel
import mowl.models.elboxembeddings.losses as L
from mowl.nn.elmodule import ELModule
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


# %%
#
# Definition of loss functions
# ------------------------------
            
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

    def gci0_loss(data, neg = False):
        c = self.class_embed(data[:, 0])
        d = self.class_embed(data[:, 1])

        off_c = th.abs(self.class_offset(data[:, 0]))
        off_d = th.abs(self.class_offset(data[:, 1]))

        euc = th.abs(c-d)
        dst = th.reshape(th.linalg.norm(th.relu(euc + off_c - off_d + self.margin), axis=1), [-1, 1])
            
        return dst

    def gci1_loss(data, neg = False):
        c = self.class_embed(data[:, 0])
        d = self.class_embed(data[:, 1])
        e = self.class_embed(data[:, 2])
        off_c = th.abs(self.class_offset(data[:, 0]))
        off_d = th.abs(self.class_offset(data[:, 1]))
        off_e = th.abs(self.class_offset(data[:, 2]))
    
        startAll = th.maximum(c - off_c, d - off_d)
        endAll   = th.minimum(c + off_c, d + off_d)
    
        new_offset = th.abs(startAll-endAll)/2
 
        cen1 = (startAll+endAll)/2
        euc = th.abs(cen1 - e)
    
        dst = th.reshape(th.linalg.norm(th.relu(euc + new_offset - off_e + self.margin), axis=1), [-1, 1]) +th.linalg.norm(th.relu(startAll-endAll), axis=1)
        return dst

    def gci1_bot_loss(data, neg = False):
        c = self.class_embed(data[:, 0])
        d = self.class_embed(data[:, 1])

        off_c = th.abs(self.class_offset(data[:, 0]))
        off_d = th.abs(self.class_offset(data[:, 1]))

    
        euc = th.abs(c - d)
        dst = th.reshape(th.linalg.norm(th.relu(-euc + off_c + off_d + self.margin), axis=1), [-1, 1])
        return dst
        
    def gci2_loss(data, neg = False):
        if neg:
            return gci2_loss_neg(data,   rel_embed, self.margin)
        else:
            c = self.class_embed(data[:, 0])
            r = self.rel_embed(data[:, 1])
            d = self.class_embed(data[:, 2])
            
            off_c = th.abs(self.class_offset(data[:, 0]))
            off_d = th.abs(self.class_offset(data[:, 2]))
            
            euc = th.abs(c + r - d)
            dst = th.reshape(th.linalg.norm(th.relu(euc + off_c - off_d + self.margin), axis=1),[-1,1])
            return  dst


    def gci2_loss_neg(data):
        c = self.class_embed(data[:, 0])
        r = self.rel_embed(data[:, 1])
            
        rand_index = np.random.choice(self.class_embed.weight.shape[0], size = len(data))
        rand_index = th.tensor(rand_index).to(self.class_embed.weight.device)
        d = self.class_embed(rand_index)
            
    
        off_c = th.abs(self.class_offset(data[:,0]))
        off_d = th.abs(self.class_offset(rand_index))

        euc = th.abs(c + r - d)
        dst = th.reshape(th.linalg.norm(th.relu(euc - off_c - off_d - self.margin), axis=1), [-1, 1])
        return dst

    def gci3_loss(data, neg = False):
        r = self.rel_embed(data[:, 0])
        c = self.class_embed(data[:, 1])
        d = self.class_embed(data[:, 2])
            
        off_c = th.abs(self.class_offset(data[:, 1]))
        off_d = th.abs(self.class_offset(data[:, 2]))

        euc = th.abs(c - r - d)
        dst = th.reshape(th.linalg.norm(th.relu(euc - off_c - off_d + self.margin), axis=1), [-1, 1])
        return dst
