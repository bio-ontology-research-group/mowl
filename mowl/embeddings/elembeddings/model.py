import os

from mowl.base_models.elmodel import EmbeddingELModel
import mowl.embeddings.elembeddings.losses as L
from mowl.projection.factory import projector_factory
from mowl.projection.edge import Edge
import math
import logging

from mowl.embeddings.elembeddings.evaluate import ELEmbeddingsPPIEvaluator
from tqdm import trange, tqdm

import torch as th
from torch import nn

class ELEmbeddings(EmbeddingELModel):

    def __init__(self, dataset, embed_dim=50, margin=0, reg_norm=1, learning_rate=0.001, epochs=1000, model_filepath = None, device = 'cpu'):
        super().__init__(dataset)


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

#        self.load_eval_data()

    def get_entities_index_dict(self):
        return self.classes_index_dict, self.relations_index_dict
                                                                     
    def init_model(self):
        self.load_data(extended = self.extended)
        self.model = ELModel(
            len(self.classes_index_dict),
            len(self.relations_index_dict),
            device = self.device).to(self.device)
    
        
    def train(self):
        self.load_data(extended=self.extended)
        self.train_nfs = self.train_nfs[:4]
        self.valid_nfs = self.valid_nfs[:4]
        self.test_nfs = self.test_nfs[:4]
      
        self.init_model()
        optimizer = th.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        best_loss = float('inf')

        for epoch in trange(self.epochs):
            self.model.train()
            loss = self.model(self.train_nfs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            self.model.eval()
            valid_loss = self.model(self.valid_nfs).cpu().detach().item()
            if best_loss > valid_loss:
                best_loss = valid_loss
#                print('Saving the model')
                th.save(self.model.state_dict(), self.model_filepath)
            if epoch % 100 == 0:
                print(f'Epoch {epoch}: Train loss: {loss.cpu().detach().item()} Valid loss: {valid_loss}')

                
    def eval_method(self, data):
        self.load_data(extended=self.extended)
        scores = self.model.gci2_loss(data)
        return scores
        
    def load_eval_data(self):
        
        if self._loaded_eval:
            return

        eval_property = self.dataset.get_evaluation_property()
        eval_classes = self.dataset.get_evaluation_classes()
        if isinstance(eval_classes, tuple) and len(eval_classes) == 2:
            self._head_entities = eval_classes[0]
            self._tail_entities = eval_classes[1]
        else:
            self._head_entities = set(list(eval_classes)[:])
            self._tail_entities = set(list(eval_classes)[:])

        eval_projector = projector_factory('taxonomy_rels', taxonomy=False, relations=[eval_property])

        self._training_set = eval_projector.project(self.dataset.ontology)
        self._testing_set = eval_projector.project(self.dataset.testing)
        
        self._loaded_eval = True


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
        
            
    def evaluate(self):
        self.load_data(extended=self.extended)
        self.model = ELModel(len(self.classes_index_dict), len(self.relations_index_dict), self.device).to(self.device)
        print('Load the best model', self.model_filepath)
        self.model.load_state_dict(th.load(self.model_filepath))
        self.model.eval()
        test_loss = self.model(self.test_nfs).cpu().detach().item()
        print('Test Loss:', test_loss)
        
    def get_embeddings(self):
        self.load_data(extended=self.extended)
        self.init_model()
        
        print('Load the best model', self.model_filepath)
        self.model.load_state_dict(th.load(self.model_filepath))
        self.model.eval()

        ent_embeds = {k:v for k,v in zip(self.classes_index_dict.keys(), self.model.class_embed.weight.cpu().detach().numpy())}
        rel_embeds = {k:v for k,v in zip(self.relations_index_dict.keys(), self.model.rel_embed.weight.cpu().detach().numpy())}
        return ent_embeds, rel_embeds
 

    def evaluate_ppi(self):
        self.load_data(extended=self.extended)

        self.init_model()
        print('Load the best model', self.model_filepath)
        self.model.load_state_dict(th.load(self.model_filepath))
        self.model.eval()

        eval_method = self.model.gci2_loss

        evaluator = ELEmbeddingsPPIEvaluator(self.dataset.testing, eval_method, self.dataset.ontology, self.classes_index_dict, self.relations_index_dict, device = self.device)

        evaluator()

        evaluator.print_metrics()

    #Methods used to be exported for inference
    def gci0_loss(self, data):
        x, y = data
        x, y = self.classes_index_dict[x], self.classes_index_dict[y]
        ###implement code that checks dimensionality
        point = self.point_to_tensor([x,y])
        return self.model.gci0_loss(point)

    def gci1_loss(self, data):
        x, y, z = data
        x = self.classes_index_dict[x]
        y = self.classes_index_dict[y]
        z = self.classes_index_dict[z]
        point = self.point_to_tensor([x,y,z])
        return self.model.gci1_loss(point)
    
    def gci2_loss(self, data):
        x, y, z = data
        x = self.classes_index_dict[x]
        y = self.relations_index_dict[y]
        z = self.classes_index_dict[z]
        point = self.point_to_tensor([x,y,z])
        return self.model.gci2_loss(point)

    def gci3_loss(self, data):
        x, y, z = data
        x = self.relations_index_dict[x]
        y = self.classes_index_dict[y]
        z = self.classes_index_dict[z]
        point = self.point_to_tensor([x,y,z])
        return self.model.gci3_loss(point)

    def point_to_tensor(self, point):
        point = [list(point)]
        point = th.tensor(point).to(self.device)
        return point

class ELModel(nn.Module):

    def __init__(self, nb_ont_classes, nb_rels, embed_dim=50, margin=0.1, device = "cpu"):
        super().__init__()
        self.nb_ont_classes = nb_ont_classes
        self.nb_rels = nb_rels

        # ELEmbeddings
        self.embed_dim = embed_dim
        self.class_embed = nn.Embedding(self.nb_ont_classes, embed_dim)
        self.class_norm = nn.BatchNorm1d(embed_dim)
        k = math.sqrt(1 / embed_dim)
        nn.init.uniform_(self.class_embed.weight, -0, 1)
        self.class_rad = nn.Embedding(self.nb_ont_classes, 1)
        nn.init.uniform_(self.class_rad.weight, -0, 1)
        
        self.rel_embed = nn.Embedding(nb_rels, embed_dim)
        nn.init.uniform_(self.rel_embed.weight, -k, k)
        self.all_gos = th.arange(self.nb_ont_classes).to(device)
        self.margin = margin

    def gci0_loss(self, data):
        return L.gci0_loss(data, self.class_norm, self.class_embed, self.class_rad, self.margin)
    def gci1_loss(self, data):
        return L.gci1_loss(data, self.class_norm, self.class_embed, self.class_rad, self.margin)
    def gci3_loss(self, data):
        return L.gci3_loss(data, self.class_norm, self.class_embed, self.class_rad, self.rel_embed, self.margin)
    def gci2_loss(self, data):
        return L.gci2_loss(data, self.class_norm, self.class_embed, self.class_rad, self.rel_embed, self.margin)
    def gci2_loss_neg(self, data):
        return L.gci2_loss_neg(data, self.class_norm, self.class_embed, self.class_rad, self.rel_embed, self.margin)

        
    def forward(self, go_normal_forms):
        gci0, gci1, gci2, gci3 = go_normal_forms
        
        loss = 0
        if len(gci0) > 1:
            loss += th.mean(self.gci0_loss(gci0))
        if len(gci1) > 1:
            loss += th.mean(self.gci1_loss(gci1))
        if len(gci2) > 1:
            loss += th.mean(self.gci2_loss(gci2))
            loss += th.mean(self.gci2_loss_neg(gci2))
        if len(gci3) > 1:
            loss += th.mean(self.gci3_loss(gci3))
        return loss



