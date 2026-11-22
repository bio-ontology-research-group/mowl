"""Shared model code of the hierarchy-embedding methods (hyperbolic cones, shadow cones,
disk embeddings): they embed the told subclass order between *named* classes."""

import numpy as np
import torch as th

from mowl.base_models.elmodel import EmbeddingELModel
from mowl.nn.order.manifolds import RiemannianSGD

_THING = "http://www.w3.org/2002/07/owl#Thing"
_NOTHING = "http://www.w3.org/2002/07/owl#Nothing"


class _PairDataset:
    """The ``gci0`` rows kept for training, with the interface the training loop needs."""

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class OrderEmbeddingModel(EmbeddingELModel):
    r"""Base of models that embed the subclass order. Trains on the axioms
    :math:`C \sqsubseteq D` between named classes (the ``gci0`` normal form) and ignores
    every other axiom of the ontology; reflexive axioms, axioms :math:`\bot \sqsubseteq D`
    and axioms with an auxiliary class introduced by normalisation are dropped.

    ``num_neg`` negatives per positive are drawn by corrupting the child and the parent in
    turn (as in the papers), and a corruption that is a told training axiom is re-drawn.
    The generic training loop of :class:`~mowl.base_models.EmbeddingELModel` is used, so any
    :class:`~mowl.training.TrainingRegime` (for example
    :class:`~mowl.training.PerNormalFormMiniBatches`, a loss from :mod:`mowl.training.losses`
    or :class:`~mowl.training.TransitiveClosurePositives`) can be attached to
    ``model.training_regime``. Scores are the energies of the module (lower: more plausible),
    declared with ``axiom_scoring = "module"`` for the rank-based evaluation regimes.
    """

    axiom_scoring = "module"
    scored_forms = ("gci0",)

    def __init__(self, dataset, embed_dim, learning_rate, batch_size, num_neg, device,
                 model_filepath, neg_sampling_gcis=None, seed=None):
        if num_neg < 1:
            raise ValueError("num_neg must be at least 1.")
        super().__init__(dataset, embed_dim, batch_size, extended=False,
                         model_filepath=model_filepath, device=device,
                         learning_rate=learning_rate, neg_sampling_gcis=neg_sampling_gcis)
        self.num_neg = num_neg
        self.seed = seed
        self._order_datasets = None
        self._DEFAULT_NEG_SAMPLING_CONFIG = {
            "gci0": {"index_pool": "classes",
                     "corrupt_column": [i % 2 for i in range(num_neg)]}}
        self._eval_gci_name = "gci0"

    # -- data ----------------------------------------------------------------------------
    @property
    def training_datasets(self):
        """Only ``gci0``, restricted to pairs of distinct named classes (see the class)."""
        if self._order_datasets is None:
            base = EmbeddingELModel.training_datasets.fget(self)
            data = base["gci0"][:] if "gci0" in base else th.zeros((0, 2), dtype=th.long)
            n_named = len(self.named_class_index_dict)
            keep = (data < n_named).all(dim=1) & (data[:, 0] != data[:, 1])
            nothing = self.class_index_dict.get(_NOTHING)
            if nothing is not None:
                keep &= data[:, 0] != nothing
            data = th.unique(data[keep], dim=0)
            self._order_datasets = {"gci0": _PairDataset(data.to(self.device))}
            self._known_pairs = set((data[:, 0] * n_named + data[:, 1]).tolist())
        return self._order_datasets

    def generate_negatives(self, gci_name, gci_dataset, max_resample=10):
        neg = super().generate_negatives(gci_name, gci_dataset)
        if neg is None or len(neg) == 0:
            return neg
        n = len(gci_dataset)
        n_named = len(self.named_class_index_dict)
        ids = np.array(list(self.named_class_index_dict.values()))
        columns = [c for c in self.get_negative_sampling_config()[gci_name]["corrupt_column"]]
        for _ in range(max_resample):
            codes = (neg[:, 0] * n_named + neg[:, 1]).tolist()
            bad = [i for i, c in enumerate(codes)
                   if c in self._known_pairs or neg[i, 0] == neg[i, 1]]
            if not bad:
                break
            for i in bad:
                col = columns[i // n]
                neg[i, col] = int(np.random.choice(ids))
        return neg

    # -- training ------------------------------------------------------------------------
    def get_optimizer(self):
        return RiemannianSGD(self.module.param_groups(), lr=self.learning_rate)

    def get_embeddings(self):
        """Class points (and, for disks, radii) as ``({iri: vector}, None, None)``."""
        w = self.module.class_embed.weight.detach().cpu().numpy()
        return {k: w[i] for k, i in self.class_index_dict.items()}, None, None

    def energies(self, child_iris, parent_iris):
        """Energy (lower: more plausible) of the axioms ``child ⊑ parent`` given as lists
        of class IRIs."""
        cid = self.class_index_dict
        rows = th.tensor([[cid[c], cid[p]] for c, p in zip(child_iris, parent_iris)],
                         device=self.device)
        with th.no_grad():
            self.module.eval()
            return self.module(rows, "gci0").reshape(len(rows))  # one energy per axiom
