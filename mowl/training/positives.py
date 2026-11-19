"""Regimes that change which atomic subsumptions are trained on as positives.

Hierarchy embeddings disagree on whether to train on the *transitive reduction* of a
subsumption hierarchy (the "basic" edges, from which the rest follows) or on its
*transitive closure*:

* RegD (Yang and Chen, Section 5.1), HiT (He et al. 2024, "Multi-hop Inference") and
  Gumbel boxes (Dasgupta et al. 2020, Section 5.3) train on the transitive reduction and
  test on the edges it implies;
* order embeddings (Vendrov et al. 2016, Section 4.1) and the transitive-closure
  saturation of CatE (Zhapa-Camacho and Hoehndorf, Section 4.5, "Tr") train on the
  closure.

Both regimes act on the ``gci0`` dataset (:math:`C \\sqsubseteq D`, both atomic) and only
on the told hierarchy: they use the asserted ``gci0`` axioms and no reasoner. For the
deductive closure of all normal forms as positives see
:class:`mowl.training.dele.ClosurePositives`.
"""

import torch as th

from mowl.training.regime import TrainingRegime


class _TensorDataset(th.utils.data.Dataset):
    """A minimal axiom dataset over a tensor (``data``, indexing, ``len``)."""

    def __init__(self, data):
        super().__init__()
        self._data = data

    @property
    def data(self):
        return self._data

    def __getitem__(self, idx):
        return self._data[idx]

    def __len__(self):
        return len(self._data)


def _edges(rows, exclude=()):
    up = {}
    for c, d in rows:
        if c == d or c in exclude or d in exclude:
            continue
        up.setdefault(c, set()).add(d)
    return up


def _descendants_closure(up):
    """``{c: set of all (transitive) superclasses of c}`` for every node with edges."""
    memo = {}

    def sups(c, stack=()):
        if c in memo:
            return memo[c]
        out = set()
        todo = list(up.get(c, ()))
        seen = set()
        while todo:
            x = todo.pop()
            if x in seen:
                continue
            seen.add(x)
            out.add(x)
            todo.extend(up.get(x, ()))
        memo[c] = out
        return out

    for c in list(up):
        sups(c)
    return memo


class _HierarchyRegime(TrainingRegime):
    TOP = "http://www.w3.org/2002/07/owl#Thing"
    BOTTOM = "http://www.w3.org/2002/07/owl#Nothing"

    def __init__(self, gci_name="gci0"):
        super().__init__()
        self.gci_name = gci_name
        self.num_before = None
        self.num_after = None

    def _special(self):
        cid = self.model.class_index_dict
        return {cid[i] for i in (self.TOP, self.BOTTOM) if i in cid}

    def training_datasets(self, datasets):
        if self.gci_name not in datasets:
            return datasets
        data = datasets[self.gci_name][:]
        self.num_before = len(data)
        new = self.transform(data)
        self.num_after = len(new)
        out = dict(datasets)
        out[self.gci_name] = _TensorDataset(new)
        return out

    def transform(self, data):
        raise NotImplementedError


class TransitiveReductionPositives(_HierarchyRegime):
    """Train only on the transitive reduction of the asserted atomic subsumptions.

    An axiom :math:`C \\sqsubseteq D` of ``gci0`` is removed if :math:`D` is reachable
    from :math:`C` through a path of length at least two of the other asserted
    ``gci0`` axioms. Axioms involving :math:`\\top` or :math:`\\bot` are kept. On cycles
    (equivalent classes) every edge of the cycle is kept, so the reduction is not
    minimal there.

    RegD (Section 5.1): "All models are trained exclusively on basic edges, which are
    edges not implied transitively by other edges in the graph"; HiT (Section 4,
    Multi-hop Inference) trains on one-hop subsumptions; Gumbel boxes (Section 5.3)
    "train ... on the transitive reduction edges".
    """

    key = "transitive-reduction-positives"

    def transform(self, data):
        special = self._special()
        rows = data.tolist()
        up = _edges(rows, exclude=special)
        closure = _descendants_closure(up)
        keep = []
        for c, d in rows:
            if c in special or d in special or c == d:
                keep.append(True)
                continue
            redundant = False
            for x in up.get(c, ()):
                if x != d and d in closure.get(x, ()) and c not in closure.get(x, ()):
                    redundant = True
                    break
            keep.append(not redundant)
        return data[th.tensor(keep, dtype=th.bool, device=data.device)]


class TransitiveClosurePositives(_HierarchyRegime):
    """Train on the transitive closure of the asserted atomic subsumptions: every
    :math:`C \\sqsubseteq E` with a path :math:`C \\sqsubseteq \\dots \\sqsubseteq E` of
    asserted ``gci0`` axioms is added as a positive (reflexive pairs excluded).

    Order embeddings (Vendrov et al. 2016, Section 4.1) train and test on the transitive
    closure of WordNet; CatE (Zhapa-Camacho and Hoehndorf, Section 4.5) reports the effect
    of saturating its lattice with the transitive closure ("Tr").

    :param max_new: Optional cap on the number of added axioms (the closure of a large
        hierarchy can be quadratic).
    """

    key = "transitive-closure-positives"

    def __init__(self, gci_name="gci0", max_new=None):
        super().__init__(gci_name=gci_name)
        self.max_new = max_new

    def transform(self, data):
        special = self._special()
        rows = data.tolist()
        up = _edges(rows, exclude=special)
        closure = _descendants_closure(up)
        present = set(map(tuple, rows))
        new = []
        for c in sorted(closure):
            for e in sorted(closure[c]):
                if c != e and (c, e) not in present:
                    new.append((c, e))
        if self.max_new is not None:
            new = new[:int(self.max_new)]
        if not new:
            return data
        extra = th.tensor(new, dtype=data.dtype, device=data.device)
        return th.cat([data, extra], dim=0)
