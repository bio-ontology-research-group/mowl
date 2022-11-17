from mowl.base_models.alcmodel import EmbeddingALCModel

from mowl.models.falcon.module import FALCONModule
from mowl.projection import TaxonomyWithRelationsProjector

from tqdm import trange, tqdm
import numpy as np
import torch


class FALCON(EmbeddingALCModel):

    def __init__(self,
                 dataset,
                 embed_dim=128,
                 anon_e=4,
                 batch_size=256,
                 epochs=128,
                 learning_rate=0.01,
                 model_filepath=None,
                 device='cpu'
                 ):
        super().__init__(dataset, batch_size, model_filepath=model_filepath)

        self.embed_dim = embed_dim
        self.anon_e = anon_e
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.device = device
        self._loaded = False
        self._loaded_eval = False
        self.init_model()

    def init_model(self):
        ind_triples = self.training_dataset.get_obj_prop_assertion_data()
        heads_dict = {}
        tails_dict = {}
        for item in ind_triples:
            h, r, t = item[0]  # ,item[0], item[1], item[2]
            h, r, t = h.item(), r.item(), t.item()
            if (t, r) not in heads_dict:
                heads_dict[(t, r)] = [h, ]
            else:
                heads_dict[(t, r)].append(h)
            if (h, r) not in tails_dict:
                tails_dict[(h, r)] = [t, ]
            else:
                tails_dict[(h, r)].append(t)

        self.model = FALCONModule(
            len(self.dataset.classes),
            len(self.dataset.individuals),
            len(self.dataset.object_properties),
            heads_dict,
            tails_dict,
            embed_dim=self.embed_dim,
            anon_e=self.anon_e,
        ).to(self.device)

    def train(self):

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        best_loss = float('inf')

        for epoch in trange(self.epochs):
            self.model.train()

            anon_e_emb_1 = self.model.e_embedding.weight.detach()[:self.anon_e // 2] \
                + torch.normal(0, 0.1, size=(self.anon_e // 2, self.embed_dim)).to(self.device)
            anon_e_emb_2 = torch.rand(self.anon_e // 2, self.embed_dim).to(self.device)
            torch.nn.init.xavier_uniform_(anon_e_emb_2)
            anon_e_emb = torch.cat([anon_e_emb_1, anon_e_emb_2], dim=0)

            train_loss = 0
            loss = 0

            for axiom, dataloader in self.training_dataloaders.items():
                for batch_data in dataloader:
                    loss += torch.mean(self.model(axiom, batch_data[0], anon_e_emb))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.detach().item()

            loss = 0
            with torch.no_grad():
                self.model.eval()
                valid_loss = 0
                for axiom, dataloader in self.validation_dataloaders.items():
                    for batch_data in dataloader:
                        loss = torch.mean(self.model(axiom, batch_data[0], anon_e_emb))
                        valid_loss += loss.detach().item()

            if best_loss > valid_loss:
                best_loss = valid_loss
                torch.save(self.model.state_dict(), self.model_filepath)
            print(f'Epoch {epoch}: Train loss: {train_loss} Valid loss: {valid_loss}')

        return 1

    def load_eval_data(self):

        if self._loaded_eval:
            return

        eval_property = self.dataset.get_evaluation_property()
        eval_classes = self.dataset.get_evaluation_classes()

        self._head_entities = eval_classes[0]
        self._tail_entities = eval_classes[1]

        eval_projector = TaxonomyWithRelationsProjector(taxonomy=False,
                                                        relations=[eval_property])

        self._training_set = eval_projector.project(self.dataset.ontology)
        self._testing_set = eval_projector.project(self.dataset.testing)

        self._loaded_eval = True

    def get_embeddings(self):
        self.init_model()

        print('Load the best model', self.model_filepath)
        self.model.load_state_dict(torch.load(self.model_filepath))
        self.model.eval()

        ent_embeds = {k: v for k, v in zip(self.class_index_dict.keys(),
                                           self.model.c_embedding.weight.cpu().detach().numpy())}
        rel_embeds = {k: v for k, v in zip(self.object_property_index_dict.keys(),
                                           self.model.r_embedding.weight.cpu().detach().numpy())}
        return ent_embeds, rel_embeds

    def load_best_model(self):
        self.init_model()
        self.model.load_state_dict(torch.load(self.model_filepath))
        self.model.eval()

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
