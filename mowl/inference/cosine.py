from mowl.inference.axiom_scoring import AxiomScoring
import torch as th
import torch.nn as nn
from gensim.models.keyedvectors import KeyedVectors


class CosineSimilarityInfer(AxiomScoring):

    def __init__(self, embeddings, relation):
        embeddings = self.embeddings_to_dict(embeddings)
        method = CosineSimilarity(embeddings)
        class_list = list(embeddings.keys())
        patterns = [f"c?? SubClassOf {relation} some c??"]
        super().__init__(patterns, method, class_list)

    def embeddings_to_dict(self, embeddings):
        embeddings_dict = dict()
        if isinstance(embeddings, KeyedVectors):
            for idx, word in enumerate(embeddings.index_to_key):
                embeddings_dict[word] = embeddings[word]
        elif isinstance(embeddings, dict):
            embeddings_dict = embeddings
        else:
            raise TypeError("Embeddings type {type(embeddings)} not recognized. Expected types \
                are dict or gensim.models.keyedvectors.KeyedVectors")

        return embeddings_dict


class CosineSimilarity(nn.Module):

    def __init__(self, class_embeddings, device="cpu"):
        super().__init__()

        self.class_index_dict = {v: k for k, v in enumerate(class_embeddings.keys())}
        self.class_vectors = list(class_embeddings.values())
        self.device = device
        num_classes = len(self.class_vectors)
        embedding_size = len(self.class_vectors[0])

        if isinstance(self.class_vectors, list):
            self.class_vectors = th.tensor(self.class_vectors).to(device)

        self.class_embedding_layer = nn.Embedding(num_classes, embedding_size)
        self.class_embedding_layer.weight = nn.parameter.Parameter(self.class_vectors)

    def forward(self, data):
        x, y = data
        x, y = self.class_index_dict[x], self.class_index_dict[y]
        x = th.tensor([x]).to(self.device)
        y = th.tensor([y]).to(self.device)
        # implement code that checks dimensionality

        srcs = self.class_embedding_layer(x)
        dsts = self.class_embedding_layer(y)

        x = th.sum(srcs * dsts, dim=1)
        return 1 - th.sigmoid(x)
