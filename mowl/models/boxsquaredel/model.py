from mowl.base_models.elmodel import EmbeddingELModel
from mowl.nn import BoxSquaredELModule
from tqdm import trange, tqdm
import torch as th
import numpy as np
import logging

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
logger.addHandler(handler)
logger.setLevel(logging.INFO)


class BoxSquaredEL(EmbeddingELModel):
    """
    Implementation based on [jackermeier2023]_.
    """

    def __init__(self,
                 dataset,
                 embed_dim=50,
                 margin=0.02,
                 reg_norm=1,
                 learning_rate=0.001,
                 epochs=1000,
                 batch_size=4096 * 8,
                 delta=2.5,
                 reg_factor=0.2,
                 num_negs=4,
                 model_filepath=None,
                 device='cpu'
                 ):
        super().__init__(dataset, embed_dim, batch_size, extended=True, model_filepath=model_filepath)

        self.margin = margin
        self.reg_norm = reg_norm
        self.delta = delta
        self.reg_factor = reg_factor
        self.num_negs = num_negs
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.device = device
        self._loaded = False
        self.extended = False
        self.init_module()

    def init_module(self):
        self.module = BoxSquaredELModule(
            len(self.class_index_dict),
            len(self.object_property_index_dict),
            len(self.individual_index_dict),
            embed_dim=self.embed_dim,
            gamma=self.margin,
            delta=self.delta,
            reg_factor=self.reg_factor

        ).to(self.device)

    def train(self, epochs=None, validate_every=1):
        logger.warning('You are using the default training method. If you want to use a cutomized training method (e.g., different negative sampling, etc.), please reimplement the train method in a subclass.')

        points_per_dataset = {k: len(v) for k, v in self.training_datasets.items()}
        string = "Training datasets: \n"
        for k, v in points_per_dataset.items():
            string += f"\t{k}: {v}\n"

        logger.info(string)
            
        optimizer = th.optim.Adam(self.module.parameters(), lr=self.learning_rate)
        best_loss = float('inf')

        all_classes_ids = list(self.class_index_dict.values())
        all_inds_ids = list(self.individual_index_dict.values())
        
        if epochs is None:
            epochs = self.epochs
        
        for epoch in trange(epochs):
            self.module.train()

            train_loss = 0
            loss = 0

            for gci_name, gci_dataset in self.training_datasets.items():
                if len(gci_dataset) == 0:
                    continue

                loss += th.mean(self.module(gci_dataset[:], gci_name))
                if gci_name == "gci2":
                    idxs_for_negs = np.random.choice(all_classes_ids, size=len(gci_dataset), replace=True)
                    rand_index = th.tensor(idxs_for_negs).to(self.device)
                    data = gci_dataset[:]
                    neg_data = th.cat([data[:, :2], rand_index.unsqueeze(1)], dim=1)
                    loss += th.mean(self.module(neg_data, gci_name, neg=True))

                if gci_name == "object_property_assertion":
                    idxs_for_negs = np.random.choice(all_inds_ids, size=len(gci_dataset), replace=True)
                    rand_index = th.tensor(idxs_for_negs).to(self.device)
                    data = gci_dataset[:]
                    neg_data = th.cat([data[:, :2], rand_index.unsqueeze(1)], dim=1)
                    loss += th.mean(self.module(neg_data, gci_name, neg=True))
                    
            loss += self.module.regularization_loss()
                    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.detach().item()

            loss = 0

            if (epoch + 1) % validate_every == 0:
                if self.dataset.validation is not None:
                    with th.no_grad():
                        self.module.eval()
                        valid_loss = 0
                        gci2_data = self.validation_datasets["gci2"][:]
                        loss = th.mean(self.module(gci2_data, "gci2"))
                        valid_loss += loss.detach().item()


                        if valid_loss < best_loss:
                            best_loss = valid_loss
                            th.save(self.module.state_dict(), self.model_filepath)
                    print(f'Epoch {epoch+1}: Train loss: {train_loss} Valid loss: {valid_loss}')
                else:
                    print(f'Epoch {epoch+1}: Train loss: {train_loss}')
 
    def eval_method(self, data):
        return self.module.gci2_loss(data)

    def get_embeddings(self):
        self.init_module()

        print('Load the best model', self.model_filepath)
        self.load_best_model()
                
        ent_embeds = {
            k: v for k, v in zip(self.class_index_dict.keys(),
                                 self.module.class_embed.weight.cpu().detach().numpy())}
        rel_embeds = {
            k: v for k, v in zip(self.object_property_index_dict.keys(),
                                 self.module.rel_embed.weight.cpu().detach().numpy())}
        if self.module.ind_embed is not None:
            ind_embeds = {
                k: v for k, v in zip(self.individual_index_dict.keys(),
                                     self.module.ind_embed.weight.cpu().detach().numpy())}
        else:
            ind_embeds = None
        return ent_embeds, rel_embeds, ind_embeds

    def load_best_model(self):
        self.init_module()
        self.module.load_state_dict(th.load(self.model_filepath))
        self.module.eval()

