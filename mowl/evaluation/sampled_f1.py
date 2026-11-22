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
  so the code scores the class whose index in the reference data equals the role's
  index instead of the axiom (for GO, *biological_process* for every ``part_of``
  query). :func:`reference_role_classes` rebuilds that class from the reference index
  files and ``SampledF1Regime(positive_scores="reference")`` reproduces the slip; the
  published prediction-task numbers are computed with it. The regime reports the F1
  of the axiom's own score (``"pair"``) next to it.
* the threshold is the best-F1 threshold on the validation split, evaluated the same
  way and searched over the sorted scores as the code does
  (:func:`reference_best_threshold`); F1, precision and recall of the test split at
  that threshold (``score <= threshold`` is positive).

The matrix entry ``relu(d_bd)`` of the boxes is computed from the regions of the
geometry of the model (:func:`mowl.representation.geometry_of`: absolute offsets, absolute
radii), so it applies to every method whose classes are boxes or balls; this matches the
code for boxes. The child region of ``gci2`` is the region of ``exists r.B`` of the
geometry (the filler shifted by ``-r`` for a translation).
The ball branch of the code computes boundary terms with absolute radii too
(its training uses softplus radii; the difference is documented on the method card).
"""

import numpy as np
import torch as th

from mowl.evaluation.regime import EvaluationRegime
from mowl.evaluation.axiom_ranking import el_rows
from mowl.evaluation.region_access import all_class_regions, centre_and_size

#: Column of the score matrix whose entry is the positive score of a row, per normal
#: form, as in the reference code (``compute_nf1_ranks`` / ``compute_nf3_ranks`` of
#: ``EXP_ontology/evaluate.py``). For ``gci0`` (``[C, D]``) column 1 is the superclass
#: -- the axiom's own score. For ``gci2`` (``[C, r, D]``) column 1 is the role, read as
#: a *class* index: an arbitrary class instead of the query, a slip the published
#: prediction-task numbers (Table 4) nevertheless carry.
REFERENCE_POSITIVE_COLUMN = {"gci0": 1, "gci2": 1}


_PURPOSE = "The sampled-F1 score"


def _regions(module):
    """``(geometry, centers, sizes, is_ball)`` of the module, read through its geometry."""
    region, geometry = all_class_regions(module, _PURPOSE)
    return (geometry,) + centre_and_size(region, _PURPOSE)


def class_regions(module):
    """Centres and sizes of all class regions, from the geometry of the module
    (:func:`mowl.representation.geometry_of`).

    Boxes: centres and (non-negative) half-widths. Balls: centres and radii.

    :return: ``(centers, sizes, is_ball)`` tensors on the module's device.
    :raises NotImplementedError: If the module has no geometry yet.
    :raises TypeError: If its classes are neither balls nor boxes.
    """
    return _regions(module)[1:]


def _child_regions(geometry, centers, sizes, gci_name, rows):
    """The fixed child region of every query row, as in the reference code: the region
    of ``C`` for ``C ⊑ D``; the region of ``∃r.B`` (the filler shifted back by ``r`` for
    a translation) for ``A ⊑ ∃r.B``."""
    if gci_name == "gci0":
        return centers[rows[:, 0]], sizes[rows[:, 0]]
    if gci_name == "gci2":
        region = geometry.exists(rows[:, 1], geometry.classes(rows[:, 2]))
        centre, size, _ = centre_and_size(region, _PURPOSE)
        return centre, size
    raise ValueError(f"Unsupported normal form {gci_name!r}.")


SCORES = ("boundary", "energy")


def _score(child_center, child_size, cand_center, cand_size, is_ball, score="boundary"):
    """Score of child regions inside candidate regions, on broadcast tensors (centres of
    shape ``(..., n)``; box offsets ``(..., n)``, ball radii ``(...)``).

    ``"boundary"``: ``relu(d_bd)``, the released code's score (``depth_type`` is never
    ``'norm'`` in ``baseline_sweep.py``). ``"energy"``: the code's other branch
    (``depth_type == 'norm'`` in ``compute_nf1_ranks`` / ``compute_nf3_ranks``):
    ``d_bd + d_dep`` unrectified, the depth term always added, for balls with the code's
    broadcast of the radius difference over the ``n`` coordinates
    (:math:`\\sum_i \\Delta c_i^2 + n\\,\\Delta r^2`).
    """
    dc = child_center - cand_center
    if is_ball:
        dr = child_size - cand_size
        bd = th.linalg.norm(dc, dim=-1) + dr
        if score == "boundary":
            return th.relu(bd)
        depth = (dc.square().sum(-1) + dc.shape[-1] * dr.square()) \
            / (child_size.abs() * cand_size.abs())
        return bd + depth
    do = child_size - cand_size
    diff = dc.abs() + do
    contained = diff.max(dim=-1).values
    outside = th.linalg.norm(th.relu(diff), dim=-1)
    bd = th.where(contained < 0, contained, outside)
    if score == "boundary":
        return th.relu(bd)
    size_product = th.linalg.norm(child_size, dim=-1) * th.linalg.norm(cand_size, dim=-1)
    depth = (dc.abs() + do.abs()).sum(-1) / size_product
    return bd + depth


def candidate_dists(module, gci_name, rows, row_batch=64, candidate_chunk=16384,
                    score="boundary"):
    """The score matrix of the RegD code's evaluation.

    Entry ``[i, X]`` is the score (:func:`_score`; by default ``relu(d_bd)``) of the
    fixed child region of row ``i`` inside the region of class ``X``; the entry of the
    row's column-0 class is ``+inf`` (the code filters ``C ⊑ C``).

    :param module: A model or module with a geometry of boxes or balls
        (:func:`mowl.representation.geometry_of`): ELBE, ELEm, RegD, Box2EL, TransBox, ...
    :param gci_name: ``"gci0"`` (candidates replace the superclass) or ``"gci2"``
        (candidates replace the subclass; the child is the filler translated by ``-r``).
    :param rows: Query rows in mOWL's column layout, on any device.
    :param score: ``"boundary"`` or ``"energy"`` (see :func:`_score`).
    :return: A float tensor of shape ``(len(rows), number of classes)`` on the CPU.
    """
    geometry, centers, sizes, is_ball = _regions(module)
    device = centers.device
    n_classes = centers.shape[0]
    rows = rows.to(device)
    out = th.empty((len(rows), n_classes), dtype=th.float32)
    for i in range(0, len(rows), row_batch):
        chunk = rows[i:i + row_batch]
        child_center, child_size = _child_regions(geometry, centers, sizes, gci_name, chunk)
        for j in range(0, n_classes, candidate_chunk):
            cc = centers[j:j + candidate_chunk]
            cs = sizes[j:j + candidate_chunk]
            val = _score(child_center[:, None], child_size[:, None], cc[None], cs[None],
                         is_ball, score)
            out[i:i + len(chunk), j:j + candidate_chunk] = val.cpu()
        # filter "C ⊑ C": the class of the query column is not a candidate
        out[i:i + len(chunk)].scatter_(1, chunk[:, 0:1].cpu(), float("inf"))
    return out


def pair_scores(module, gci_name, rows, score="boundary"):
    """The axiom's own score (:func:`_score`) of the child region inside the region of
    the kept class -- the superclass D for ``gci0``, the subclass A for ``gci2``."""
    geometry, centers, sizes, is_ball = _regions(module)
    device = centers.device
    rows = rows.to(device)
    head_col = 1 if gci_name == "gci0" else 0
    child_center, child_size = _child_regions(geometry, centers, sizes, gci_name, rows)
    return _score(child_center, child_size, centers[rows[:, head_col]],
                  sizes[rows[:, head_col]], is_ball, score).cpu()


def reference_role_classes(model, ontology, data_root=None, zip_path=None,
                           index_files=None):
    """The class the RegD code scores as the "positive" of a prediction query, per role.

    ``compute_nf3_ranks`` of the code reads the positive score of a query ``[C, r, D]``
    from score-matrix column ``batch_data[:, 1]`` -- the *role* index of the reference
    data, read as a *class* index. The class it reads is therefore the class whose index
    in the reference ``classes.json`` equals the role's index in ``relations.json``
    (for GO, ``part_of`` has index 1 and class 1 is ``GO:0008150``, *biological_process*;
    45% of the GO test queries are scored against it). This function rebuilds that map
    from the index files of the pinned data release that the RegD code ships
    (identical to the Box\\ :sup:`2`\\ EL release,
    :func:`~mowl.datasets.builtin.jackermeier2024_reference.fetch_box2el_release`).

    Class IRIs are matched by name; the ANATOMY dataset of mOWL names its classes
    ``.../aux/<i>`` with ``i`` the reference index, which is matched by number.

    :param model: An :class:`~mowl.base_models.EmbeddingELModel`.
    :param ontology: ``"GALEN"``, ``"GO"`` or ``"ANATOMY"``.
    :param index_files: ``(classes, relations)``: the two index dictionaries (IRI in
        angle brackets to index) instead of reading them from the release.
    :return: A ``LongTensor`` indexed by the model's role index: the model's class index
        of the class the code scores, ``-1`` where there is none.
    """
    import json
    import re
    import zipfile
    if index_files is None:
        from mowl.datasets.builtin.jackermeier2024_reference import fetch_box2el_release
        ontology = ontology.upper()
        with zipfile.ZipFile(fetch_box2el_release(data_root, zip_path)) as zf:
            classes = json.loads(zf.read(f"data/{ontology}/prediction/classes.json"))
            relations = json.loads(zf.read(f"data/{ontology}/prediction/relations.json"))
    else:
        classes, relations = index_files
    cid = model.class_index_dict
    rid = model.object_property_index_dict
    aux = {}
    for iri, i in cid.items():
        m = re.search(r"/aux/(\d+)$", iri)
        if m:
            aux[int(m.group(1))] = i
    by_ref_index = {}
    for iri, idx in classes.items():
        key = iri.strip("<>")
        if key in cid:
            by_ref_index[int(idx)] = cid[key]
        elif int(idx) in aux:
            by_ref_index[int(idx)] = aux[int(idx)]
    n_roles = (max(rid.values()) + 1) if rid else 0
    out = th.full((n_roles,), -1, dtype=th.long)
    for iri, ridx in relations.items():
        r = rid.get(iri.strip("<>"))
        if r is not None and int(ridx) in by_ref_index:
            out[r] = by_ref_index[int(ridx)]
    return out


def reference_best_threshold(pos, neg):
    """Best-F1 threshold as ``compute_best_threshold`` of the RegD code computes it.

    The scores are sorted (ascending) and every *position* of the sorted list is tried:
    the F1 of predicting the first ``i`` scores positive, with ties broken by the sort
    order. The first position with the largest F1 gives the threshold (its score); applied
    as ``score <= threshold``, all scores tied with it count as positive.

    :return: ``(threshold, f1)``; ``(None, 0.0)`` if no position has a positive F1.
    """
    pos = np.asarray(pos, dtype=float)
    neg = np.asarray(neg, dtype=float)
    scores = np.concatenate([pos, neg])
    labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    if len(pos) == 0:
        return None, 0.0
    order = np.argsort(scores, kind="quicksort")
    sorted_scores, sorted_labels = scores[order], labels[order]
    tp = np.cumsum(sorted_labels)
    k = np.arange(1, len(scores) + 1)
    precision = tp / k
    recall = tp / len(pos)
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = np.where(tp > 0, 2 * precision * recall / (precision + recall), 0.0)
    best = int(np.argmax(f1))
    if f1[best] <= 0:
        return None, 0.0
    return float(sorted_scores[best]), float(f1[best])


def _apply_threshold(pos, neg, threshold):
    """F1, precision and recall of ``score <= threshold``, as the code computes them."""
    pos = np.asarray(pos, dtype=float)
    neg = np.asarray(neg, dtype=float)
    tp = float(np.sum(pos <= threshold))
    fp = float(np.sum(neg <= threshold))
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / len(pos) if len(pos) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return f1, precision, recall


def _split_scores(model, gci_name, rows, num_rows, num_negatives, gen,
                  role_classes=None, row_batch=64, score="boundary"):
    """Positive and negative scores of one split, sampled as the reference code does.

    :param num_rows: Number of rows to take from the front of ``rows`` (all if
        ``None``), as the code takes ``val_data[nf][:1000]``.
    :param role_classes: For ``gci2``, the map of :func:`reference_role_classes`; the
        reference positive score of a row is then the matrix entry of that class.
    :return: ``(pos_reference, pos_pair, neg)`` numpy arrays: the positive score as the
        code reads it, the axiom's own score, and ``num_negatives`` negatives per row.
        For ``gci0`` the two positive scores agree; for ``gci2`` without ``role_classes``
        the reference scores are ``None``.
    """
    module = model.module
    if len(rows) == 0:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    if num_rows is not None:
        rows = rows[:int(num_rows)]
    ref, pair, neg = [], [], []
    with th.no_grad():
        for i in range(0, len(rows), row_batch):
            chunk = rows[i:i + row_batch]
            dists = candidate_dists(module, gci_name, chunk, row_batch=row_batch,
                                    score=score)
            idx = th.randint(dists.shape[1], (len(chunk), num_negatives), generator=gen)
            neg.append(th.gather(dists, 1, idx).reshape(-1).numpy())
            if gci_name == "gci0":
                p = th.gather(dists, 1, chunk[:, 1:2].cpu()).reshape(-1).numpy()
                ref.append(p)
                pair.append(p)
                continue
            pair.append(pair_scores(module, gci_name, chunk, score=score).numpy())
            if role_classes is not None:
                cls = role_classes[chunk[:, 1].cpu()]
                vals = th.gather(dists, 1, cls.clamp_min(0).reshape(-1, 1)).reshape(-1)
                vals[cls < 0] = float("inf")
                ref.append(vals.numpy())
    pos_ref = np.concatenate(ref) if ref else None
    return pos_ref, np.concatenate(pair), np.concatenate(neg)


def _role_classes_for(model, gci_name, positive_scores, reference_ontology, cache):
    if gci_name != "gci2" or positive_scores != "reference":
        return None
    if reference_ontology is None:
        raise ValueError(
            "positive_scores='reference' on gci2 reproduces the RegD code's positive score "
            "(the class indexed by the role in the reference data) and needs "
            "reference_ontology ('GALEN', 'GO' or 'ANATOMY').")
    key = (id(model), reference_ontology)
    if key not in cache:
        cache[key] = reference_role_classes(model, reference_ontology)
    return cache[key]


_ROLE_CLASS_CACHE = {}


def validation_best_f1(model, gci_name="gci0", num_positives=1000, num_negatives=10,
                       positive_scores="reference", reference_ontology=None, seed=0,
                       score="boundary"):
    """Best achievable F1 (with its threshold tuned on the split itself) on the first
    ``num_positives`` validation axioms -- the model-selection quantity of the RegD code
    (``train`` in ``EXP_ontology/baseline_sweep.py``).

    :param positive_scores: ``"reference"`` (the code's positive score, see
        :class:`SampledF1Regime`) or ``"pair"`` (the axiom's own score).
    :return: ``(f1, threshold)``; ``(0.0, None)`` if the validation split is empty.
    """
    try:
        rows = model.validation_datasets.get(gci_name)
    except AttributeError:
        rows = None
    if rows is None or len(rows) == 0:
        return 0.0, None
    rows = rows[:]
    role_classes = _role_classes_for(model, gci_name, positive_scores, reference_ontology,
                                     _ROLE_CLASS_CACHE)
    gen = th.Generator().manual_seed(seed)
    pos_ref, pos_pair, neg = _split_scores(model, gci_name, rows.cpu(), num_positives,
                                           num_negatives, gen, role_classes, score=score)
    pos = pos_pair if positive_scores == "pair" else pos_ref
    threshold, f1 = reference_best_threshold(pos, neg)
    return f1, threshold


class SampledF1Regime(EvaluationRegime):
    """Threshold classification of query axioms and random negatives, as in the RegD
    code (see the module docstring).

    The score is the rectified boundary dissimilarity read from the model's regions
    (boxes: absolute offsets; balls: absolute radii), *not* the model's loss: the depth
    dissimilarity and any margins of the training loss are not part of the score. The
    threshold maximising F1 on the validation split (searched as the code does,
    :func:`reference_best_threshold`) is applied to the test split.

    :param gci_name: ``"gci0"`` (:math:`C \\sqsubseteq D`, RegD's inference task) or
        ``"gci2"`` (:math:`A \\sqsubseteq \\exists r.B`, RegD's prediction task, whose
        child region is the filler translated by ``-r``).
    :param num_negatives: Random candidate classes per query (10 in the code).
    :param valid_positives: Validation queries to take from the front of the split
        (the code takes the first 1,000); ``None`` takes all.
    :param test_positives: Test queries to take from the front of the split; ``None``
        (the default) takes **all**, as the code does (the paper says 1,000 sampled).
    :param positive_scores: ``"reference"`` (default) or ``"pair"``. They agree for
        ``gci0``. For ``gci2``, ``"reference"`` reproduces the code's slip: the positive
        score of ``A ⊑ ∃r.B`` is the score of the class whose reference index equals the
        role's reference index (:func:`reference_role_classes`), not of ``A``; the
        published Table 4 numbers are computed this way. ``"pair"`` scores the axiom
        itself. The regime always reports both (``f1`` follows ``positive_scores``;
        ``info["f1_pair"]`` and ``info["f1_reference"]``).
    :param reference_ontology: ``"GALEN"``, ``"GO"`` or ``"ANATOMY"``: the reference data
        whose index files define the slip (needed for ``gci2`` with ``"reference"``).
    :param seed: Seed of the negative sampling (independent of the training seed).
    :param score: ``"boundary"`` (default: ``relu(d_bd)``, what the released code
        computes) or ``"energy"`` (``d_bd + d_dep``, the code's unused
        ``depth_type='norm'`` branch; see :func:`_score`).
    """

    key = "regd-sampled-f1"
    metrics = ("f1", "precision", "recall")
    subsets = ("all",)

    def __init__(self, gci_name="gci0", num_negatives=10, valid_positives=1000,
                 test_positives=None, positive_scores="reference", reference_ontology=None,
                 seed=0, score="boundary"):
        if gci_name not in REFERENCE_POSITIVE_COLUMN:
            raise ValueError(f"Unsupported normal form {gci_name!r}.")
        if positive_scores not in ("reference", "pair"):
            raise ValueError("positive_scores must be 'reference' or 'pair'.")
        self.gci_name = gci_name
        self.num_negatives = int(num_negatives)
        self.valid_positives = valid_positives
        self.test_positives = test_positives
        self.positive_scores = positive_scores
        self.reference_ontology = reference_ontology
        if score not in SCORES:
            raise ValueError(f"score must be one of {SCORES}.")
        self.score = score
        self.seed = seed
        self.info = {}

    _WIDTH = {"gci0": 2, "gci2": 3}

    def _rows(self, model, ontology):
        empty = th.zeros((0, self._WIDTH[self.gci_name]), dtype=th.long)
        rows = el_rows(model, ontology)[0].get(self.gci_name, empty)
        return rows.cpu()

    @staticmethod
    def _classify(vpos, vneg, tpos, tneg):
        threshold, source = None, "validation"
        if vpos is not None and len(vpos):
            threshold, _ = reference_best_threshold(vpos, vneg)
        if threshold is None:
            # As in the code: without a validation threshold, the best one on the test
            # split itself.
            threshold, _ = reference_best_threshold(tpos, tneg)
            source = "test"
        if threshold is None:
            return 0.0, 0.0, 0.0, None, source
        f1, precision, recall = _apply_threshold(tpos, tneg, threshold)
        return f1, precision, recall, threshold, source

    def evaluate(self, model, dataset):
        gen = th.Generator().manual_seed(self.seed)
        want_reference = self.gci_name == "gci2" and (
            self.positive_scores == "reference" or self.reference_ontology is not None)
        role_classes = None
        if want_reference:
            role_classes = _role_classes_for(model, self.gci_name, "reference",
                                             self.reference_ontology, _ROLE_CLASS_CACHE)
        vrows = self._rows(model, dataset.validation)
        trows = self._rows(model, dataset.testing)
        v_ref, v_pair, vneg = _split_scores(model, self.gci_name, vrows,
                                            self.valid_positives, self.num_negatives, gen,
                                            role_classes, score=self.score)
        t_ref, t_pair, tneg = _split_scores(model, self.gci_name, trows,
                                            self.test_positives, self.num_negatives, gen,
                                            role_classes, score=self.score)
        pair = self._classify(v_pair, vneg, t_pair, tneg)
        ref = self._classify(v_ref, vneg, t_ref, tneg) if t_ref is not None else None
        chosen = pair if self.positive_scores == "pair" else ref
        f1, precision, recall, threshold, source = chosen
        self.info = {"scoring": "relu-boundary" if self.score == "boundary" else "energy",
                     "positive_scores": self.positive_scores,
                     "threshold": None if threshold is None else float(threshold),
                     "threshold_source": source,
                     "test_positives": int(len(t_pair)),
                     "test_negatives": int(len(tneg)),
                     "valid_positives": int(len(v_pair)),
                     "f1_pair": pair[0]}
        if ref is not None:
            self.info["f1_reference"] = ref[0]
            if role_classes is not None:
                self.info["reference_role_classes_found"] = int((role_classes >= 0).sum())
        return {"all": {"f1": f1, "precision": precision, "recall": recall}}
