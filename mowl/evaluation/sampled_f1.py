"""Threshold classification of sampled subsumptions, the evaluation of RegD (Yang and
Chen, "RegD: Hierarchical Embeddings via Dissimilarity between Arbitrary Euclidean
Regions", Sections 5.1-5.3).

RegD evaluates ontology embeddings as classifiers. The evaluation of the authors' code
(``EXP_ontology/evaluate.py`` of HuiYang1997/RegD, which produced the published numbers
of Tables 3 and 4) is:

* a *score matrix* per query row: for every candidate class X, the rectified boundary
  dissimilarity :math:`\\mathrm{relu}(d_{bd})` of a fixed *child* region inside the
  region of X. For ``C ⊑ D`` (``gci0``) the child is the region of C and the candidates
  replace the superclass D; for ``A ⊑ ∃r.B`` (``gci2``, the prediction task) the child is
  the region of B translated by ``-r`` and the candidates replace A. The depth
  dissimilarity is *not* part of the score, whatever the model trained on.
* the class of the query itself is filtered out of the matrix (score ``+inf``).
* 10 negatives per positive, drawn with replacement from the matrix row: random
  classes, not checked against the ontology (a drawn class may be the true superclass
  or otherwise entailed).
* the positive score of a row is the matrix entry of **column 1**. For ``gci0`` that is
  the superclass D, the axiom's own score; for ``gci2`` column 1 is the *role index*,
  so the code scores an arbitrary class (the one whose index equals the role's) instead
  of the axiom -- :attr:`SampledF1Regime.positive_scores` reproduces this; it is what
  the published prediction-task numbers are computed with.
* the threshold is the best-F1 threshold on the validation split, evaluated the same
  way; F1, precision and recall of the test split at that threshold.

The matrix entry ``relu(d_bd)`` of the boxes is computed from the raw embedding
parameters (absolute offsets), for balls from the absolute radii; this matches the code
for boxes. The ball branch of the code computes boundary terms with absolute radii too
(its training uses softplus radii; the difference is documented on the method card).
"""

import numpy as np
import torch as th

from mowl.evaluation.regime import EvaluationRegime
from mowl.evaluation.axiom_ranking import el_rows
from mowl.evaluation.ranking_metrics import binary_f1, best_threshold

#: Column of the score matrix whose entry is the positive score of a row, per normal
#: form, as in the reference code (``compute_nf1_ranks`` / ``compute_nf3_ranks`` of
#: ``EXP_ontology/evaluate.py``). For ``gci0`` (``[C, D]``) column 1 is the superclass
#: -- the axiom's own score. For ``gci2`` (``[C, r, D]``) column 1 is the role, read as
#: a *class* index: an arbitrary class instead of the query, a slip the published
#: prediction-task numbers (Table 4) nevertheless carry.
REFERENCE_POSITIVE_COLUMN = {"gci0": 1, "gci2": 1}


def class_regions(module):
    """Centres and sizes of all class regions, from the module's embeddings.

    Boxes (``class_offset``): centres and absolute offsets. Balls (``class_rad``):
    centres and absolute radii.

    :return: ``(centers, sizes, is_ball)`` tensors on the module's device.
    """
    if hasattr(module, "class_offset"):
        return module.class_embed.weight, module.class_offset.weight.abs(), False
    if hasattr(module, "class_rad"):
        return module.class_embed.weight, module.class_rad.weight.abs().squeeze(-1), True
    raise TypeError(f"Module {type(module).__name__} has neither boxes nor balls.")


def _child_regions(module, gci_name, rows):
    """The fixed child region of every query row, as in the reference code."""
    centers, sizes, is_ball = class_regions(module)
    if gci_name == "gci0":
        return centers[rows[:, 0]], sizes[rows[:, 0]]
    if gci_name == "gci2":
        translated = centers[rows[:, 2]] - module.rel_embed.weight[rows[:, 1]]
        return translated, sizes[rows[:, 2]]
    raise ValueError(f"Unsupported normal form {gci_name!r}.")


def _boundary(child_center, child_size, centers, sizes, is_ball):
    """Boundary dissimilarity of the child inside every candidate region."""
    if is_ball:
        return (th.linalg.norm(child_center[:, None, :] - centers[None], dim=-1)
                + child_size[:, None] - sizes[None])  # noqa: W503
    diff = (child_center[:, None, :] - centers[None]).abs() \
        + child_size[:, None, :] - sizes[None]
    contained = diff.max(dim=-1).values
    outside = th.linalg.norm(th.relu(diff), dim=-1)
    return th.where(contained < 0, contained, outside)


def candidate_dists(module, gci_name, rows, row_batch=64, candidate_chunk=16384):
    """The score matrix of the RegD code's evaluation.

    Entry ``[i, X]`` is ``relu(d_bd)`` of the fixed child region of row ``i`` inside the
    region of class ``X``; the entry of the row's column-0 class is ``+inf`` (the code
    filters ``C ⊑ C``).

    :param module: An EL module with boxes (``class_offset``) or balls (``class_rad``)
        and translations (``rel_embed``): ELBE, ELEm, RegD.
    :param gci_name: ``"gci0"`` (candidates replace the superclass) or ``"gci2"``
        (candidates replace the subclass; the child is the filler translated by ``-r``).
    :param rows: Query rows in mOWL's column layout, on any device.
    :return: A float tensor of shape ``(len(rows), number of classes)`` on the CPU.
    """
    centers, sizes, is_ball = class_regions(module)
    device = centers.device
    n_classes = centers.shape[0]
    rows = rows.to(device)
    out = th.empty((len(rows), n_classes), dtype=th.float32)
    for i in range(0, len(rows), row_batch):
        chunk = rows[i:i + row_batch]
        child_center, child_size = _child_regions(module, gci_name, chunk)
        for j in range(0, n_classes, candidate_chunk):
            cc = centers[j:j + candidate_chunk]
            cs = sizes[j:j + candidate_chunk]
            bd = _boundary(child_center, child_size, cc, cs, is_ball)
            out[i:i + len(chunk), j:j + candidate_chunk] = th.relu(bd).cpu()
        # filter "C ⊑ C": the class of the query column is not a candidate
        out[i:i + len(chunk)].scatter_(1, chunk[:, 0:1].cpu(), float("inf"))
    return out


def pair_scores(module, gci_name, rows):
    """The axiom's own score: ``relu(d_bd)`` of the child region inside the region of
    the kept class -- the superclass D for ``gci0``, the subclass A for ``gci2``."""
    centers, sizes, is_ball = class_regions(module)
    device = centers.device
    rows = rows.to(device)
    head_col = 1 if gci_name == "gci0" else 0
    child_center, child_size = _child_regions(module, gci_name, rows)
    head_center = centers[rows[:, head_col]]
    head_size = sizes[rows[:, head_col]]
    if is_ball:
        bd = th.linalg.norm(head_center - child_center, dim=-1) + child_size - head_size
    else:
        diff = (head_center - child_center).abs() + child_size - head_size
        contained = diff.max(dim=-1).values
        outside = th.linalg.norm(th.relu(diff), dim=-1)
        bd = th.where(contained < 0, contained, outside)
    return th.relu(bd).cpu()


def _split_scores(model, gci_name, rows, num_rows, num_negatives, gen,
                  positive_scores, row_batch=64):
    """Positive and negative scores of one split, sampled as the reference code does.

    :param num_rows: Number of rows to take from the front of ``rows`` (all if
        ``None``), as the code takes ``val_data[nf][:1000]``.
    :param positive_scores: ``"reference"`` (matrix column 1, as in the code) or
        ``"pair"`` (the axiom's own score).
    :return: ``(pos, neg)`` numpy arrays; ``neg`` has ``num_negatives`` entries per row.
    """
    module = model.module
    if len(rows) == 0:
        return np.zeros(0), np.zeros(0)
    if num_rows is not None:
        rows = rows[:int(num_rows)]
    pos, neg = [], []
    with th.no_grad():
        for i in range(0, len(rows), row_batch):
            chunk = rows[i:i + row_batch]
            dists = candidate_dists(module, gci_name, chunk, row_batch=row_batch)
            idx = th.randint(dists.shape[1], (len(chunk), num_negatives), generator=gen)
            neg.append(th.gather(dists, 1, idx).reshape(-1).numpy())
            if positive_scores == "pair":
                pos.append(pair_scores(module, gci_name, chunk).numpy())
            else:
                pos.append(th.gather(dists, 1, chunk[:, 1:2].cpu()).reshape(-1).numpy())
    return np.concatenate(pos), np.concatenate(neg)


def validation_best_f1(model, gci_name="gci0", num_positives=1000, num_negatives=10,
                       positive_scores="reference", seed=0):
    """Best achievable F1 (with its threshold tuned on the split itself) on the first
    ``num_positives`` validation axioms -- the model-selection quantity of the RegD code
    (``train`` in ``EXP_ontology/baseline_sweep.py``).

    :return: ``(f1, threshold)``; ``(0.0, None)`` if the validation split is empty.
    """
    empty = th.zeros((0, 2), dtype=th.long)
    try:
        rows = model.validation_datasets.get(gci_name)
    except AttributeError:
        rows = None
    rows = empty if rows is None else rows[:]
    if len(rows) == 0:
        return 0.0, None
    gen = th.Generator().manual_seed(seed)
    pos, neg = _split_scores(model, gci_name, rows.cpu(), num_positives, num_negatives,
                             gen, positive_scores)
    labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    threshold, f1 = best_threshold(np.concatenate([pos, neg]), labels)
    return f1, threshold


class SampledF1Regime(EvaluationRegime):
    """Threshold classification of query axioms and random negatives, as in the RegD
    code (see the module docstring).

    The score is the rectified boundary dissimilarity read from the model's regions
    (boxes: absolute offsets; balls: absolute radii), *not* the model's loss: the depth
    dissimilarity and any margins of the training loss are not part of the score. The
    threshold maximising F1 on the validation split is applied to the test split.

    :param gci_name: ``"gci0"`` (:math:`C \\sqsubseteq D`, RegD's inference task) or
        ``"gci2"`` (:math:`A \\sqsubseteq \\exists r.B`, RegD's prediction task, whose
        child region is the filler translated by ``-r``).
    :param num_negatives: Random candidate classes per query (10 in the code).
    :param valid_positives: Validation queries to take from the front of the split
        (the code takes the first 1,000); ``None`` takes all.
    :param test_positives: Test queries to take from the front of the split; ``None``
        (the default) takes **all**, as the code does (the paper says 1,000 sampled).
    :param positive_scores: ``"reference"`` (default; the positive score of a query is
        score-matrix column 1, as in the code) or ``"pair"`` (the axiom's own score).
        The two agree for ``gci0``; for ``gci2`` column 1 is the role index read as a
        class index, so ``"reference"`` scores an arbitrary class -- the published
        prediction-task numbers (Table 4) are computed this way, and
        :attr:`REFERENCE_POSITIVE_COLUMN` documents it.
    :param seed: Seed of the negative sampling (independent of the training seed).
    """

    key = "regd-sampled-f1"
    metrics = ("f1", "precision", "recall")
    subsets = ("all",)

    def __init__(self, gci_name="gci0", num_negatives=10, valid_positives=1000,
                 test_positives=None, positive_scores="reference", seed=0):
        if gci_name not in REFERENCE_POSITIVE_COLUMN:
            raise ValueError(f"Unsupported normal form {gci_name!r}.")
        if positive_scores not in ("reference", "pair"):
            raise ValueError("positive_scores must be 'reference' or 'pair'.")
        self.gci_name = gci_name
        self.num_negatives = int(num_negatives)
        self.valid_positives = valid_positives
        self.test_positives = test_positives
        self.positive_scores = positive_scores
        self.seed = seed
        self.info = {}

    _WIDTH = {"gci0": 2, "gci2": 3}

    def _rows(self, model, ontology):
        empty = th.zeros((0, self._WIDTH[self.gci_name]), dtype=th.long)
        rows = el_rows(model, ontology)[0].get(self.gci_name, empty)
        return rows.cpu()

    def evaluate(self, model, dataset):
        gen = th.Generator().manual_seed(self.seed)
        vrows = self._rows(model, dataset.validation)
        trows = self._rows(model, dataset.testing)
        vpos, vneg = _split_scores(model, self.gci_name, vrows, self.valid_positives,
                                   self.num_negatives, gen, self.positive_scores)
        tpos, tneg = _split_scores(model, self.gci_name, trows, self.test_positives,
                                   self.num_negatives, gen, self.positive_scores)
        if len(vpos):
            vlabels = np.concatenate([np.ones(len(vpos)), np.zeros(len(vneg))])
            threshold, _ = best_threshold(np.concatenate([vpos, vneg]), vlabels)
            threshold_source = "validation"
        else:
            threshold, threshold_source = None, "test"
        ts = np.concatenate([tpos, tneg])
        tl = np.concatenate([np.ones(len(tpos)), np.zeros(len(tneg))]).astype(bool)
        if threshold is None:
            threshold, _ = best_threshold(ts, tl)
        pred = ts <= threshold
        tp = float(np.sum(pred & tl))
        precision = tp / max(float(np.sum(pred)), 1.0)
        recall = tp / max(float(np.sum(tl)), 1.0)
        self.info = {"scoring": "relu-boundary", "positive_scores": self.positive_scores,
                     "threshold": float(threshold),
                     "threshold_source": threshold_source,
                     "test_positives": int(len(tpos)),
                     "test_negatives": int(len(tneg)),
                     "valid_positives": int(len(vpos))}
        return {"all": {"f1": binary_f1(tl, pred), "precision": precision,
                        "recall": recall}}
