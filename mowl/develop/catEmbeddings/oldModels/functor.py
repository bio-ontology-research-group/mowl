import torch as th
import torch.nn as nn


class Functor(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()

        self.map_object = nn.Sequential(
            nn.Linear(embedding_size, embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, embedding_size),
            nn.Sigmoid()
        )

        m_embedding_size = embedding_size*embedding_size

        self.map_morphism = nn.Sequential(
            nn.Linear(m_embedding_size, m_embedding_size),
            nn.ReLU(),
            nn.Linear(m_embedding_size, m_embedding_size),
            nn.Tanh()
        )


    def forward(self, obj = None, morphism = None):
        
        if not obj is None:
            obj = self.map_object(obj)

        if not morphism is None:
            bs, w, h = morphism.shape
            morphism = morphism.reshape(bs, -1)
            morphism = self.map_morphism(morphism)
            morphism = morphism.reshape(bs, w, h)
            
        return obj, morphism


