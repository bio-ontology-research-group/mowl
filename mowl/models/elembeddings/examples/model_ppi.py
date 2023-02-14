"""
EL Embeddings
===============

This example corresponds to the paper `EL Embeddings: Geometric Construction of Models for the \
Description Logic EL++ <https://www.ijcai.org/proceedings/2019/845>`_.

The idea of this paper is to embed EL by modeling ontology classes as :math:`n`-dimensional \
balls (:math:`n`-balls) and ontology object properties as transformations of those \
:math:`n`-balls. For each of the normal forms, there is a distance function defined that will \
work as loss functions in the optimization framework.
"""

# %%
# Let's just define the imports that will be needed along the example:
from mowl.base_models.elmodel import EmbeddingELModel

from mowl.models.elembeddings.module import ELEmModule

from mowl.models.elembeddings.evaluate import ELEmbeddingsPPIEvaluator
from mowl.projection.factory import projector_factory
from tqdm import trange, tqdm
import torch as th

import numpy as np


class ELEmbeddings(EmbeddingELModel):

    def __init__(self,
                 dataset,
                 embed_dim=50,
                 margin=0,
                 reg_norm=1,
                 learning_rate=0.001,
                 epochs=1000,
                 batch_size=4096 * 8,
                 model_filepath=None,
                 device='cpu'
                 ):
        super().__init__(dataset, batch_size, extended=True, model_filepath=model_filepath)

        self.embed_dim = embed_dim
        self.margin = margin
        self.reg_norm = reg_norm
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.device = device
        self._loaded = False
        self._loaded_eval = False
        self.extended = False
        self.init_model()

    def init_model(self):
        self.module = ELEmModule(
            len(self.class_index_dict),  # number of ontology classes
            len(self.object_property_index_dict),  # number of ontology object properties
            embed_dim=self.embed_dim,
            margin=self.margin
        ).to(self.device)

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
                th.save(self.module.state_dict(), self.module_filepath)
            if (epoch + 1) % checkpoint == 0:
                print(f'Epoch {epoch}: Train loss: {train_loss} Valid loss: {valid_loss}')

        return 1

    def eval_method(self, data):
        return self.module.gci2_loss(data)

    def evaluate_ppi(self):
        self.init_model()
        print('Load the best model', self.module_filepath)
        self.module.load_state_dict(th.load(self.module_filepath))
        with th.no_grad():
            self.module.eval()

            eval_method = self.module.gci2_loss

            evaluator = ELEmbeddingsPPIEvaluator(
                self.dataset.testing, eval_method, self.dataset.ontology, self.class_index_dict,
                self.object_property_index_dict, device=self.device)
            evaluator()
            evaluator.print_metrics()

    def load_eval_data(self):

        if self._loaded_eval:
            return

        eval_property = self.dataset.get_evaluation_property()
        eval_classes = self.dataset.evaluation_classes

        self._head_entities = set(list(eval_classes)[:])
        self._tail_entities = set(list(eval_classes)[:])

        eval_projector = projector_factory('taxonomy_rels', taxonomy=False,
                                           relations=[eval_property])

        self._training_set = eval_projector.project(self.dataset.ontology)
        self._testing_set = eval_projector.project(self.dataset.testing)

        self._loaded_eval = True

    def get_embeddings(self):
        self.init_model()

        print('Load the best model', self.module_filepath)
        self.module.load_state_dict(th.load(self.module_filepath))
        self.module.eval()

        ent_embeds = {
            k: v for k, v in zip(self.class_index_dict.keys(),
                                 self.module.class_embed.weight.cpu().detach().numpy())}
        rel_embeds = {
            k: v for k, v in zip(self.object_property_index_dict.keys(),
                                 self.module.rel_embed.weight.cpu().detach().numpy())}
        return ent_embeds, rel_embeds

    def load_best_model(self):
        self.init_model()
        self.module.load_state_dict(th.load(self.module_filepath))
        self.module.eval()

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
