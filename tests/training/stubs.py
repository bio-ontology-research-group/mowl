"""A tiny stand-in for an EL model, enough to call training-regime hooks directly."""

import torch as th

from mowl.training.positives import _TensorDataset

BOT = "http://www.w3.org/2002/07/owl#Nothing"


class StubModule(th.nn.Module):
    """Score of a row = sum of its entries times a learnable scale (deterministic)."""

    def __init__(self, neg_capable=("gci0", "gci2", "class_assertion")):
        super().__init__()
        self.neg_capable_gcis = frozenset(neg_capable)
        self.scale = th.nn.Parameter(th.tensor(0.1))

    def forward(self, data, gci_name, neg=False):
        s = data.float().sum(dim=1, keepdim=True) * self.scale
        return -s if neg else s


class StubModel:
    def __init__(self, datasets, num_classes=20, num_individuals=5, num_roles=3,
                 neg_capable=("gci0", "gci2", "class_assertion"), batch_size=4):
        self.class_index_dict = {f"C{i}": i for i in range(num_classes)}
        self.class_index_dict[BOT] = num_classes
        self.named_class_index_dict = {f"C{i}": i for i in range(num_classes)}
        self.individual_index_dict = {f"a{i}": i for i in range(num_individuals)}
        self.object_property_index_dict = {f"r{i}": i for i in range(num_roles)}
        self.training_datasets = {k: _TensorDataset(th.as_tensor(v, dtype=th.long))
                                  for k, v in datasets.items()}
        self.module = StubModule(neg_capable)
        self.batch_size = batch_size
        self.learning_rate = 0.1

    def get_optimizer(self):
        return th.optim.SGD(self.module.parameters(), lr=self.learning_rate)

    def compute_loss(self, pos_scores, neg_scores=None):
        loss = th.mean(pos_scores)
        if neg_scores is not None:
            loss = loss + th.mean(neg_scores)
        return loss
