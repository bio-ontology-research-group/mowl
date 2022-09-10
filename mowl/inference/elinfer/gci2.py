import numpy as np
from torch.utils.data import IterableDataset, DataLoader
from tqdm import tqdm
from scipy.stats import rankdata
import torch.nn as nn
import itertools
from mowl.projection.factory import projector_factory


class GCI2Inference():
    def __init__(self, method, class_index_dict, property_index_dict, device):
        self.method = method
        self.class_index_dict = class_index_dict
        self.property_index_dict = property_index_dict
        self.index_class_dict = {v: k for k, v in self.class_index_dict.items()}
        self.index_property_dict = {v: k for k, v in self.property_index_dict.items()}

        self.device = device

    def infer_subclass(
            self, subclass_condition=None, property_condition=None, filler_condition=None,
            axioms_to_filter=None):
        if subclass_condition is None:
            def subclass_condition(x):
                return True

        if property_condition is None:
            def property_condition(x):
                return True

        if filler_condition is None:
            def filler_condition(x):
                return True

        subclasses = {x for x in self.class_index_dict if subclass_condition(x)}
        subclasses = sorted(subclasses)
        properties = {x for x in self.property_index_dict if property_condition(x)}
        properties = sorted(properties)
        fillers = {x for x in self.class_index_dict if filler_condition(x)}
        fillers = sorted(fillers)

        subclass_name_indexemb = {s: self.class_index_dict[s] for s in subclasses}
        subclass_indexemb_indexsc = {v: k for k, v in enumerate(subclass_name_indexemb.values())}
        self.subclass_name_indexemb_inverse = invert_dict(subclass_name_indexemb)
        self.subclass_indexemb_indexsc_inverse = invert_dict(subclass_indexemb_indexsc)

        property_filler = {(self.property_index_dict[p], self.class_index_dict[f]) for p in
                           properties for f in fillers}  # (embed_index, embed_index)

        # (embed_idx, embed_idx) -> score_idx
        prop_filler_idx = {v: k for k, v in enumerate(property_filler)}

        # score_idx -> (embed_idx, embed_idx)
        self.prop_filler_idx_inverse = {v: k for k, v in prop_filler_idx.items()}

        dataset = InferGCI2Dataset(
            self.class_index_dict, self.property_index_dict, prop_filler_idx,
            subclass_indexemb_indexsc, mode="infer_subclass")
        dataloader = DataLoader(dataset, batch_size=4)

        nb_subclasses = len(subclasses)

        self.preds_subclass = np.zeros((len(property_filler), nb_subclasses), dtype=np.float32)

        self.filtered_scores = np.ones((len(property_filler), nb_subclasses), dtype=np.int32)

        if axioms_to_filter is not None:
            triples = process_axioms(
                axioms_to_filter, "taxonomy_rels", taxonomy=False, properties=properties)
            rows, cols = zip(
                *[((self.property_index_dict[p], self.class_index_dict[d]),
                   subclass_indexemb_indexsc[self.class_index_dict[c]]) for c, p, d in triples if
                  (self.property_index_dict[p], self.class_index_dict[d]) in prop_filler_idx and c
                    in subclasses])
            rows = [prop_filler_idx[r] for r in rows]
            self.filtered_scores[rows, cols] = 10000000

        diagonal = [(prop_filler_idx[(p, d)], subclass_indexemb_indexsc[c]) for p, d in
                    prop_filler_idx for c in subclass_indexemb_indexsc if c == d]

        rows, cols = zip(*diagonal)
        self.filtered_scores[rows, cols] = 1000000
        infer_model = InferGCI2Module(self.method)
        for property_idxs, filler_idxs, batch in tqdm(dataloader):
            res = infer_model(batch.to(self.device))
            res = res.cpu().detach().numpy()
            idxs = [prop_filler_idx[(p.detach().item(), d.detach().item())] for p, d in
                    zip(property_idxs, filler_idxs)]

            self.preds_subclass[idxs, :] = res

        self.preds_subclass *= self.filtered_scores

        self.ranks = rankdata(self.preds_subclass, method='ordinal')
        self.ranks = self.ranks.reshape(self.preds_subclass.shape)

    def infer_superclass_property(
            self, subclass_condition=None, property_condition=None, filler_condition=None,
            axioms_to_filter=None):
        if subclass_condition is None:
            def subclass_condition(x):
                return True

        if property_condition is None:
            def property_condition(x):
                return True

        if filler_condition is None:
            def filler_condition(x):
                return True

        subclasses = {x for x in self.class_index_dict if subclass_condition(x)}
        subclasses = sorted(subclasses)
        properties = {x for x in self.property_index_dict if property_condition(x)}
        properties = sorted(properties)
        fillers = {x for x in self.class_index_dict if filler_condition(x)}
        fillers = sorted(fillers)

        property_name_indexemb = {p: self.property_index_dict[p] for p in properties}
        property_indexemb_indexsc = {v: k for k, v in enumerate(property_name_indexemb.values())}
        self.property_name_indexemb_inverse = invert_dict(property_name_indexemb)
        self.property_indexemb_indexsc_inverse = invert_dict(property_indexemb_indexsc)

        subclass_filler = {(self.class_index_dict[s], self.class_index_dict[f]) for s in
                           subclasses for f in fillers}  # (embed_index, embed_index)

        # (embed_idx, embed_idx) -> score_idx
        sub_filler_idx = {v: k for k, v in enumerate(subclass_filler)}

        # score_idx -> (embed_idx, embed_idx)
        self.sub_filler_idx_inverse = {v: k for k, v in sub_filler_idx.items()}

        dataset = InferGCI2Dataset(
            self.class_index_dict, self.property_index_dict, sub_filler_idx,
            property_indexemb_indexsc, mode="infer_property")
        dataloader = DataLoader(dataset, batch_size=4)

        nb_properties = len(properties)

        self.preds_property = np.zeros((len(subclass_filler), nb_properties), dtype=np.float32)

        self.filtered_scores = np.ones((len(subclass_filler), nb_properties), dtype=np.int32)

        if axioms_to_filter is not None:
            triples = process_axioms(
                axioms_to_filter, "taxonomy_rels", taxonomy=False, properties=properties)
            rows, cols = zip(
                *[((self.class_index_dict[c], self.class_index_dict[d]),
                   property_indexemb_indexsc[self.property_index_dict[p]]) for c, p, d in triples
                  if (self.class_index_dict[c], self.class_index_dict[d]) in sub_filler_idx and p
                  in properties])
            rows = [sub_filler_idx[r] for r in rows]
            self.filtered_scores[rows, cols] = 10000000

        diagonal = [(sub_filler_idx[(c, d)], property_indexemb_indexsc[p]) for c, d in
                    sub_filler_idx for p in property_indexemb_indexsc if c == d]

        rows, cols = zip(*diagonal)
        self.filtered_scores[rows, cols] = 1000000
        infer_model = InferGCI2Module(self.method)

        for subclass_idxs, filler_idxs, batch in tqdm(dataloader):
            res = infer_model(batch.to(self.device))
            res = res.cpu().detach().numpy()
            idxs = [sub_filler_idx[(c.detach().item(), d.detach().item())] for c, d in
                    zip(subclass_idxs, filler_idxs)]

            self.preds_property[idxs, :] = res

        self.preds_property *= self.filtered_scores

        self.ranks = rankdata(self.preds_property, method='ordinal')
        self.ranks = self.ranks.reshape(self.preds_property.shape)

    def infer_superclass_filler(
            self, subclass_condition=None, property_condition=None, filler_condition=None,
            axioms_to_filter=None):
        if subclass_condition is None:
            def subclass_condition(x):
                return True

        if property_condition is None:
            def property_condition(x):
                return True

        if filler_condition is None:
            def filler_condition(x):
                return True

        subclasses = {x for x in self.class_index_dict if subclass_condition(x)}
        subclasses = sorted(subclasses)
        properties = {x for x in self.property_index_dict if property_condition(x)}
        properties = sorted(properties)
        fillers = {x for x in self.class_index_dict if filler_condition(x)}
        fillers = sorted(fillers)

        filler_name_indexemb = {f: self.class_index_dict[f] for f in fillers}
        filler_indexemb_indexsc = {v: k for k, v in enumerate(filler_name_indexemb.values())}
        self.filler_name_indexemb_inverse = invert_dict(filler_name_indexemb)
        self.filler_indexemb_indexsc_inverse = invert_dict(filler_indexemb_indexsc)

        subclass_property = {(self.class_index_dict[s], self.property_index_dict[p]) for s in
                             subclasses for p in properties}  # (embed_index, embed_index)

        # (embed_idx, embed_idx) -> score_idx
        sub_prop_idx = {v: k for k, v in enumerate(subclass_property)}

        # score_idx -> (embed_idx, embed_idx)
        self.sub_prop_idx_inverse = {v: k for k, v in sub_prop_idx.items()}

        dataset = InferGCI2Dataset(
            self.class_index_dict, self.property_index_dict, sub_prop_idx,
            filler_indexemb_indexsc, mode="infer_filler")
        dataloader = DataLoader(dataset, batch_size=4)

        nb_fillers = len(fillers)

        self.preds_filler = np.zeros((len(subclass_property), nb_fillers), dtype=np.float32)

        self.filtered_scores = np.ones((len(subclass_property), nb_fillers), dtype=np.int32)

        if axioms_to_filter is not None:
            triples = process_axioms(
                axioms_to_filter, "taxonomy_rels", taxonomy=False, properties=properties)
            rows, cols = zip(*[
                ((self.class_index_dict[c], self.property_index_dict[p]),
                 filler_indexemb_indexsc[self.class_index_dict[d]]) for c, p, d
                in triples if (self.class_index_dict[c], self.property_index_dict[p]) in
                sub_prop_idx and d in fillers])
            rows = [sub_prop_idx[r] for r in rows]
            self.filtered_scores[rows, cols] = 10000000

        diagonal = [(sub_prop_idx[(c, p)], filler_indexemb_indexsc[d]) for c, p in sub_prop_idx
                    for d in filler_indexemb_indexsc if c == d]

        rows, cols = zip(*diagonal)
        self.filtered_scores[rows, cols] = 1000000
        infer_model = InferGCI2Module(self.method)
        for subclass_idxs, property_idxs, batch in tqdm(dataloader):
            res = infer_model(batch.to(self.device))
            res = res.cpu().detach().numpy()
            idxs = [sub_prop_idx[(c.detach().item(), r.detach().item())] for c, r in
                    zip(subclass_idxs, property_idxs)]

            self.preds_filler[idxs, :] = res

        self.preds_filler *= self.filtered_scores
        print("before ranks")
        self.ranks = rankdata(self.preds_filler, method='ordinal').reshape(self.preds_filler.shape)
        print("after ranks")

    def get_inferences(
            self, top_k=float("inf"), infer_mode="subclass", subclasses=None, properties=None,
            fillers=None):
        if infer_mode == "subclass":
            preds = self.preds_subclass
        elif infer_mode == "property":
            preds = self.preds_property
        elif infer_mode == "filler":
            preds = self.preds_filler
        else:
            raise ValueError()

        idxsA_idxsB, missing_entities = np.where(self.ranks <= top_k)

        zipped_idxs = zip(idxsA_idxsB, missing_entities)
        axioms = dict()
        for idxA_idxB, missing_entity in tqdm(zipped_idxs):
            score = preds[idxA_idxB, missing_entity]

            if infer_mode == "filler":
                idxA_idxB_inverse = self.sub_prop_idx_inverse
                idxA, idxB = idxA_idxB_inverse[idxA_idxB]
                sub = self.index_class_dict[idxA]
                prop = self.index_property_dict[idxB]

                missing_ent_idxsc = self.filler_indexemb_indexsc_inverse[missing_entity]
                filler = self.filler_name_indexemb_inverse[missing_ent_idxsc]
            elif infer_mode == "subclass":
                idxA_idxB_inverse = self.prop_filler_idx_inverse
                idxA, idxB = idxA_idxB_inverse[idxA_idxB]

                missing_ent_idxsc = self.subclass_indexemb_indexsc_inverse[missing_entity]
                sub = self.subclass_name_indexemb_inverse[missing_ent_idxsc]

                prop = self.index_property_dict[idxA]
                filler = self.index_class_dict[idxB]
            elif infer_mode == "property":
                idxA_idxB_inverse = self.sub_filler_idx_inverse
                idxA, idxB = idxA_idxB_inverse[idxA_idxB]
                sub = self.index_class_dict[idxA]

                missing_ent_idxsc = self.property_indexemb_indexsc_inverse[missing_entity]
                prop = self.property_name_indexemb_inverse[missing_ent_idxsc]
                filler = self.index_class_dict[idxB]
            axioms[(sub, prop, filler)] = score

        axioms = dict(sorted(axioms.items(), key=lambda x: x[1]))
        return axioms


class InferGCI2Module(nn.Module):
    def __init__(self, method):
        super().__init__()

        self.method = method

    def forward(self, x):
        bs, num_classes, ents = x.shape
        assert 3 == ents
        x = x.reshape(-1, ents)

        x = self.method(x)

        x = x.reshape(bs, num_classes)

        return x


class InferGCI2Dataset(IterableDataset):
    def __init__(
            self, class_name_indexemb, property_name_indexemb, mixed_indexemb_dict,
            missing_entity_indexemb_indexsc, mode="infer_filler"):
        super().__init__()
        self.data = list(mixed_indexemb_dict.keys())[:1000]
        self.class_name_indexemb = class_name_indexemb
        self.property_name_indexemb = property_name_indexemb

        self.mixed_indexemb_indexsc = {v: k for k, v in enumerate(mixed_indexemb_dict)}

        self.missing_entity_indexemb_indexsc = missing_entity_indexemb_indexsc
        self.len_data = len(self.data)
        self.mode = mode  # this could be "infer_subclass" or "infer_superclass"

        if mode == "infer_subclass":
            self.predata = np.array([[x, -1, -1] for x in missing_entity_indexemb_indexsc.keys()])
        elif mode == "infer_property":
            self.predata = np.array([[-1, x, -1] for x in missing_entity_indexemb_indexsc.keys()])
        elif mode == "infer_filler":
            self.predata = np.array([[-1, -1, x] for x in
                                     self.missing_entity_indexemb_indexsc.keys()])

    def get_data(self):
        for a, b in self.data:

            new_array = np.array(self.predata, copy=True)

            if self.mode == "infer_subclass":
                new_array[:, 1] = a
                new_array[:, 2] = b
            elif self.mode == "infer_property":
                new_array[:, 0] = a
                new_array[:, 2] = b
            elif self.mode == "infer_filler":
                new_array[:, 0] = a
                new_array[:, 1] = b

            tensor = new_array
            yield a, b, tensor

    def __iter__(self):
        return self.get_data()

    def __len__(self):
        return self.len_data


def process_axioms(
        ontology, projection_method, taxonomy=True, bidirectional_taxonomy=False, properties=None):

    projector = projector_factory(
        projection_method, taxonomy=taxonomy, bidirectional_taxonomy=bidirectional_taxonomy,
        relations=properties)

    edges = projector.project(ontology)
    return [e.astuple() for e in edges]


def flatten(list_):
    return list(itertools.chain(*list_))


def invert_dict(dict_):
    return {v: k for k, v in dict_.items()}
