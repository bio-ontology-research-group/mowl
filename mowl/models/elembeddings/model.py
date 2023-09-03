"""
This example corresponds to the paper `EL Embeddings: Geometric Construction of Models for the \
Description Logic EL++ <https://www.ijcai.org/proceedings/2019/845>`_.

The idea of this paper is to embed EL by modeling ontology classes as :math:`n`-dimensional \
balls (:math:`n`-balls) and ontology object properties as transformations of those \
:math:`n`-balls. For each of the normal forms, there is a distance function defined that will \
work as loss functions in the optimization framework.
"""


from mowl.base_models.elmodel import EmbeddingELModel
from mowl.nn import ELEmModule
from tqdm import trange, tqdm
import torch as th
import numpy as np
from mowl.projection import projector_factory

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
        super().__init__(dataset, embed_dim, batch_size, extended=True, model_filepath=model_filepath)

        self.margin = margin
        self.reg_norm = reg_norm
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.device = device
        self._loaded = False
        self._loaded_eval = False
        self.extended = False
        self.init_module()

    def init_module(self):
        self.module = ELEmModule(
            len(self.class_index_dict),  # number of ontology classes
            len(self.object_property_index_dict),  # number of ontology object properties
            embed_dim=self.embed_dim,
            margin=self.margin
        ).to(self.device)

    def train(self):
        raise NotImplementedError
 
    def eval_method(self, data):
        return self.module.gci2_loss(data)

    def load_eval_data(self):

        if self._loaded_eval:
            return

        eval_property = self.dataset.get_evaluation_property()
        eval_classes = self.dataset.evaluation_classes.as_str
        print(eval_classes)

        self._head_entities = set(list(eval_classes)[:])
        self._tail_entities = set(list(eval_classes)[:])

        eval_projector = projector_factory('taxonomy_rels', taxonomy=False,
                                           relations=[eval_property])

        self._training_set = eval_projector.project(self.dataset.ontology)
        self._testing_set = eval_projector.project(self.dataset.testing)

        self._loaded_eval = True

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
        return ent_embeds, rel_embeds

    def load_best_model(self):
        self.init_module()
        self.module.load_state_dict(th.load(self.model_filepath))
        self.module.eval()

    @property
    def training_set(self):
        self.load_eval_data()
        return self._training_set

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
