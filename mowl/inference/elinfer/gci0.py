import numpy as np
from torch.utils.data import IterableDataset, DataLoader
from tqdm import tqdm
from scipy.stats import rankdata
import torch.nn as nn


class GCI0Inference():
    def __init__(self, method, device):
        self.method = method
        self.device = device

    def infer_subclass(self, class_index_dict):
        self.index_class_dict = {v: k for k, v in class_index_dict.items()}

        dataset = InferGCI0Dataset(class_index_dict, mode="infer_subclass")
        dataloader = DataLoader(dataset, batch_size=32)
        nb_subclasses = 100  # len(class_index_dict)
        nb_superclasses = len(class_index_dict)
        self.preds_subclass = np.zeros((nb_subclasses, nb_superclasses), dtype=np.float32)

        infer_module = InferGCI0Module(self.method)
        for subclass_idxs, batch in tqdm(dataloader):
            res = infer_module(batch.to(self.device))
            res = res.cpu().detach().numpy()
            self.preds_subclass[subclass_idxs, :] = res

        self.ranks = rankdata(self.preds_subclass, method='ordinal')
        self.ranks = self.ranks.reshape(self.preds_subclass.shape)
        np.fill_diagonal(self.preds_subclass, 100000)

    def infer_superclass(self, class_index_dict):
        self.index_class_dict = {v: k for k, v in class_index_dict.items()}

        dataset = InferGCI0Dataset(class_index_dict, mode="infer_superclass")
        dataloader = DataLoader(dataset, batch_size=4)
        nb_subclasses = len(class_index_dict)
        nb_superclasses = len(class_index_dict)
        self.preds_superclass = np.zeros((nb_subclasses, nb_superclasses), dtype=np.float32)

        for superclass_idxs, batch in tqdm(dataloader):
            res = self.method(batch.to(self.device))
            res = res.cpu().detach().numpy()
            self.preds_superclass[:, superclass_idxs] = res

        self.ranks = rankdata(self.preds_superclass).reshape(self.preds_superclass.shape)

    def get_inferences(self, top_k=3, mode="subclass"):
        if mode == "subclass":
            preds = self.preds_subclass
        elif mode == "superclass":
            preds = self.preds_superclass
        else:
            raise ValueError()

        subs, supers = np.where(self.ranks <= top_k)

        for sub, sup in zip(subs, supers):
            score = preds[sub, sup]
            sub, sup = self.index_class_dict[sub], self.index_class_dict[sup]
            print(sub, sup, score)


class InferGCI0Module(nn.Module):
    def __init__(self, method):
        super().__init__()

        self.method = method

    def forward(self, x):
        bs, num_classes, ents = x.shape
        assert 2 == ents
        x = x.reshape(-1, ents)

        x = self.method(x)

        x = x.reshape(bs, num_classes)

        return x


class InferGCI0Dataset(IterableDataset):
    def __init__(self, class_name_indexemb, mode="infer_subclass"):
        super().__init__()
        self.data = list(class_name_indexemb.keys())[:100]
        self.class_name_indexemb = class_name_indexemb
        self.len_data = len(self.data)
        self.mode = mode  # this could be "infer_subclass" or "infer_superclass"

        if mode == "infer_subclass":
            self.predata = np.array([[0, x] for x in class_name_indexemb.values()])
        elif mode == "infer_superclass":
            self.predata = np.array([[x, 0] for x in class_name_indexemb.values()])

    def get_data(self):
        for c in self.data:

            c = self.class_name_indexemb[c]

            new_array = np.array(self.predata, copy=True)

            if self.mode == "infer_subclass":
                new_array[:, 0] = c
            else:
                new_array[:, 1] = c
            tensor = new_array
            yield c, tensor

    def __iter__(self):
        return self.get_data()

    def __len__(self):
        return self.len_data
