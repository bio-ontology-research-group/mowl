"""Shared machinery of rank-based evaluation regimes for :math:`\\mathcal{EL}^{++}` models:
turning test ontologies into indexed query rows, scoring every candidate for every query,
filtering known positives, and computing deductive closures used as filters.

The regimes themselves are in :mod:`mowl.evaluation.normal_form_ranking` and
:mod:`mowl.evaluation.filtered_ranking`.
"""

import logging
import sys
import time

import numpy as np
import torch as th

from mowl.evaluation.ranking_metrics import compute_ranks

logger = logging.getLogger(__name__)

#: Column layout of mOWL's normal-form datasets (see :mod:`mowl.datasets.el`).
COLUMNS = {
    "gci0": ("C", "D"),
    "gci1": ("C", "D", "E"),
    "gci2": ("C", "r", "D"),
    "gci3": ("r", "C", "D"),
    "class_assertion": ("a", "C"),
    "object_property_assertion": ("a", "r", "b"),
}


def el_rows(model, ontology):
    """Index the axioms of ``ontology`` with the model's entity indices, per normal form.

    Axioms that mention an entity the model does not know are dropped (and counted).

    :param model: An :class:`~mowl.base_models.EmbeddingELModel`.
    :param ontology: An ``OWLOntology`` (e.g. ``dataset.testing``).
    :return: ``(rows, skipped)``: a dict from normal form name to a ``LongTensor`` on the
        CPU, and the number of dropped axioms.
    """
    from mowl.datasets.el import ELDataset
    if ontology is None:
        return {}, 0
    ds = ELDataset(ontology, model.class_index_dict, model.object_property_index_dict,
                   individual_index_dict=getattr(model, "individual_index_dict", None),
                   extended=True, device="cpu")
    datasets = ds.get_gci_datasets()
    rows = {}
    for name, d in datasets.items():
        data = d.data if hasattr(d, "data") else d[:]
        if data is None or len(data) == 0:
            continue
        rows[name] = th.as_tensor(data).long().cpu()
    return rows, int(getattr(ds, "num_skipped_axioms", 0) or 0)


def candidate_indices(model, iris, keep_order=False):
    """Model class indices of the candidate IRIs (unknown IRIs are dropped, with a
    warning).

    :param keep_order: Keep the order of ``iris`` (duplicates removed) instead of sorting
        by model index. The order decides ties under ``ties="ordinal"``.
    """
    cid = model.class_index_dict
    iris = list(iris)
    idx = [cid[i] for i in iris if i in cid]
    if len(idx) < len(iris):
        logger.warning("%d candidate classes are not known to the model and were dropped.",
                       len(iris) - len(idx))
    if keep_order:
        return th.tensor(list(dict.fromkeys(idx)), dtype=th.long)
    return th.tensor(sorted(set(idx)), dtype=th.long)


def build_filter(rows, target_col):
    """Map every query key (the row without its target column) to the set of target
    entity indices that are known to be true for it."""
    out = {}
    if rows is None:
        return out
    keep = [i for i in range(rows.shape[1]) if i != target_col]
    for r in rows.tolist():
        out.setdefault(tuple(r[i] for i in keep), set()).add(r[target_col])
    return out


def _progress_line(gci_name, done, total, num_c, started):
    spent = time.time() - started
    left = spent * (total - done) / max(done, 1)
    print(f"ranking {gci_name}: {done} of {total} queries ({num_c} candidates each) in "
          f"{spent:.0f} s, about {left:.0f} s left", file=sys.stderr, flush=True)


def rank_queries(model, gci_name, rows, target_col, candidates, scorer, ties="ordinal",
                 exclude_cols=(), filters=None, max_rows=2 ** 22, device=None,
                 progress=60.0):
    """Rank the true target of every query among the candidates.

    For every row, the column ``target_col`` is replaced by every candidate class, the
    resulting candidate axioms are scored (lower is better), and the rank of the true
    candidate is recorded.

    :param rows: ``LongTensor`` of query rows in mOWL column layout.
    :param target_col: Column of the entity that is ranked.
    :param candidates: ``LongTensor`` of candidate class indices.
    :param scorer: ``f(model, gci_name, rows) -> scores``.
    :param ties: ``"ordinal"`` or ``"average"`` (see
        :func:`mowl.evaluation.ranking_metrics.compute_ranks`).
    :param exclude_cols: Columns whose entity is removed from the candidates of its own
        query (Box2EL removes the trivial answers :math:`C \\sqsubseteq C` and
        :math:`C \\sqcap D \\sqsubseteq C, D`).
    :param filters: Optional dict from :func:`build_filter`; candidates listed for a query
        (other than its true target) are moved to the end before ranking.
    :return: ``(raw_ranks, filtered_ranks, kept)``: float tensors of ranks (filtered is
        ``None`` without filters) and a boolean mask of the rows whose target is a
        candidate (only those are ranked).
    :param progress: Seconds between progress lines on standard error (queries done,
        time spent, time left); the first comes after this long, so a short ranking prints
        nothing. ``None`` or 0: none.
    """
    device = device or getattr(model, "device", "cpu")
    num_c = len(candidates)
    position = {int(c): i for i, c in enumerate(candidates.tolist())}
    kept = th.tensor([int(r[target_col]) in position for r in rows.tolist()], dtype=th.bool)
    rows = rows[kept]
    raw, filt = [], []
    if len(rows) == 0:
        return th.zeros(0), (th.zeros(0) if filters is not None else None), kept
    batch = max(1, max_rows // max(num_c, 1))
    cand_dev = candidates.to(device)
    keep_cols = [i for i in range(rows.shape[1]) if i != target_col]
    module = getattr(model, "module", None)
    was_training = module.training if module is not None else False
    if module is not None:
        module.eval()
    started = last = time.time()
    with th.no_grad():
        start = 0
        while start < len(rows):
            b = rows[start:start + batch].to(device)
            n = len(b)
            try:
                expanded = b.repeat_interleave(num_c, dim=0)
                expanded[:, target_col] = cand_dev.repeat(n)
                scores = scorer(model, gci_name, expanded).reshape(n, num_c).float()
            except th.OutOfMemoryError:
                # Some scorers need far more memory per candidate than others (and the
                # device may be shared): retry with half the queries per pass.
                if batch == 1:
                    raise
                del b
                expanded = None
                th.cuda.empty_cache()
                batch = max(1, batch // 2)
                logger.warning("Out of memory while ranking; %d queries per pass.", batch)
                continue
            start += n
            targets = th.tensor([position[int(t)] for t in b[:, target_col].tolist()],
                                device=scores.device)
            for col in exclude_cols:
                for i, ent in enumerate(b[:, col].tolist()):
                    p = position.get(int(ent))
                    if p is not None and p != int(targets[i]):
                        scores[i, p] = float("inf")
            raw.append(compute_ranks(scores, targets, ties=ties).double().cpu())
            if filters is not None:
                fscores = scores.clone()
                for i, r in enumerate(b.tolist()):
                    known = filters.get(tuple(r[j] for j in keep_cols), ())
                    for ent in known:
                        p = position.get(int(ent))
                        if p is not None and p != int(targets[i]):
                            fscores[i, p] = float("inf")
                filt.append(compute_ranks(fscores, targets, ties=ties).double().cpu())
            if progress and time.time() - last >= progress and start < len(rows):
                last = time.time()
                _progress_line(gci_name, start, len(rows), num_c, started)
    if progress and last > started:
        _progress_line(gci_name, len(rows), len(rows), num_c, started)
    if module is not None and was_training:
        module.train()
    raw = th.cat(raw)
    filt = th.cat(filt) if filters is not None else None
    return raw, filt, kept


# ---------------------------------------------------------------------------------------
# Deductive closures used as filters
# ---------------------------------------------------------------------------------------

def _elk_superclasses(ontology):
    """Map every named class IRI of ``ontology`` to the IRIs of its (reflexive) named
    superclasses, computed with ELK."""
    from org.semanticweb.elk.owlapi import ElkReasonerFactory
    reasoner = ElkReasonerFactory().createReasoner(ontology)
    try:
        reasoner.precomputeInferences()
        out = {}
        for cls in ontology.getClassesInSignature():
            if cls.isOWLNothing():
                continue
            iri = str(cls.toStringID())
            sups = {str(c.toStringID()) for c in reasoner.getSuperClasses(cls, False)
                    .getFlattened() if not c.isOWLThing()}
            sups |= {str(c.toStringID()) for c in reasoner.getEquivalentClasses(cls)
                     .getEntities()}
            out[iri] = sups
        return out
    finally:
        reasoner.dispose()


def _merged_ontology(*ontologies):
    from mowl.owlapi import OWLAPIAdapter
    adapter = OWLAPIAdapter()
    merged = adapter.create_ontology("http://mowl.borg/benchmarks/merged")
    for ont in ontologies:
        if ont is not None:
            adapter.owl_manager.addAxioms(merged, ont.getAxioms())
    return merged


def elk_closure_rows(model, dataset, gci_name, include_testing=True):
    """A sound under-approximation of the deductive closure, as indexed rows, computed
    with ELK.

    * ``gci0``: all entailed atomic subsumptions :math:`C \\sqsubseteq D`.
    * ``gci2``: every asserted :math:`C \\sqsubseteq \\exists r.D` (training, and testing
      when ``include_testing``) extended to all :math:`C' \\sqsubseteq \\exists r.D'`
      with :math:`C' \\sqsubseteq C` and :math:`D \\sqsubseteq D'` entailed.

    DELE (Mashkova et al.) computes a larger closure with its own algorithm. This
    function is the stand-in used until :mod:`mowl.evaluation.deductive` provides that
    closure; regimes take the closure as a parameter, so the switch is local.
    """
    onts = [dataset.ontology] + ([dataset.testing] if include_testing else [])
    merged = _merged_ontology(*onts)
    sups = _elk_superclasses(merged)
    cid = model.class_index_dict
    rows = set()
    if gci_name == "gci0":
        for c, ds in sups.items():
            if c not in cid:
                continue
            for d in ds:
                if d in cid and d != c:
                    rows.add((cid[c], cid[d]))
        return th.tensor(sorted(rows), dtype=th.long).reshape(-1, 2)
    if gci_name == "gci2":
        subs = {}
        for c, ds in sups.items():
            for d in ds:
                subs.setdefault(d, set()).add(c)
        inv = {v: k for k, v in cid.items()}
        asserted = []
        for ont in onts:
            r, _ = el_rows(model, ont)
            if "gci2" in r:
                asserted.append(r["gci2"])
        if not asserted:
            return th.zeros((0, 3), dtype=th.long)
        for c, r, d in th.cat(asserted).tolist():
            c_iri, d_iri = inv.get(c), inv.get(d)
            for c2 in subs.get(c_iri, {c_iri}) | {c_iri}:
                for d2 in sups.get(d_iri, {d_iri}) | {d_iri}:
                    if c2 in cid and d2 in cid:
                        rows.add((cid[c2], r, cid[d2]))
        return th.tensor(sorted(rows), dtype=th.long).reshape(-1, 3)
    if gci_name == "class_assertion":
        # C(a) asserted in the training ontology entails D(a) for every superclass D.
        inv = {v: k for k, v in cid.items()}
        r, _ = el_rows(model, dataset.ontology)
        for a, c in r.get("class_assertion", th.zeros((0, 2), dtype=th.long)).tolist():
            c_iri = inv.get(c)
            for d in sups.get(c_iri, set()) | {c_iri}:
                if d in cid:
                    rows.add((a, cid[d]))
        return th.tensor(sorted(rows), dtype=th.long).reshape(-1, 2)
    raise NotImplementedError(f"No closure for {gci_name}.")


def closure_rows(model, dataset, gci_name, closure="auto"):
    """Rows of the deductive closure used for filtering.

    This is the single switch point for closure computation in the regimes of this
    package.

    :param closure: ``"auto"`` (the dataset's ``deductive_closure_ontology`` if it has
        one, otherwise :func:`elk_closure_rows`), ``"elk"``, or a callable
        ``f(model, dataset, gci_name) -> LongTensor``.
    """
    if callable(closure):
        return closure(model, dataset, gci_name)
    if closure == "auto" and getattr(dataset, "deductive_closure_ontology", None) is not None:
        rows, _ = el_rows(model, dataset.deductive_closure_ontology)
        return rows.get(gci_name)
    if closure in ("auto", "elk"):
        return elk_closure_rows(model, dataset, gci_name)
    raise ValueError(f"Unknown closure {closure!r}.")


def as_numpy(ranks):
    return np.asarray(ranks.tolist() if hasattr(ranks, "tolist") else ranks, dtype=float)
