"""Comparison of the canonical mixture (``canonical_mixture``) built from parts with its direct
implementation.

The constructed part (the canonical model, its support and the admissible supports of the
individuals) must be the same on both sides; the trained tables ``U``, ``V``, ``b`` are copied.
Variants: with the entropy regulariser, and with an individual that occurs only in the
validation data (its support is all of the elements: the dense path).
"""
import os
import pathlib
import tempfile

import torch as th

from tests.baselines.harness import Pair

_OPTIONS = dict(rank=4, lr=0.01, batch_size=2, use_cache=False)


def _dataset(unseen=False):
    def build():
        from tests.models.test_canonical_mixture import _write_family_abox_dataset
        base = os.environ.get("TMPDIR") or None
        folder = pathlib.Path(tempfile.mkdtemp(prefix="mowl_mixture_", dir=base))
        return _write_family_abox_dataset(folder, include_unseen_val_ind=unseen)
    return build


def _new(**options):
    def build(dataset):
        from mowl.models import CanonicalMixture
        return CanonicalMixture(dataset, **{**_OPTIONS, **options})
    return build


def _old(**options):
    def build(dataset):
        from mowl.baselines.models.canonical_mixture.model import CanonicalMixture
        return CanonicalMixture(dataset, **{**_OPTIONS, **options})
    return build


def _tables(model):
    """``U``, ``V``, ``b`` by a name common to both sides (the module of the direct
    implementation holds them itself, the combination in its individual model)."""
    m = model.module
    return {"U": m.U, "V": m.V, "b": m.b}


def _copy(old, new):
    a, b = old.module, new.module
    assert old.structure.element_names == new.structure.element_names
    for name in ("support_offsets", "support_indices", "own_logit_offset", "is_full_support",
                 "individual_element"):
        assert th.equal(getattr(a, name), getattr(b, name)), name
    with th.no_grad():
        for name, table in _tables(new).items():
            table.copy_(_tables(old)[name])


def _rows(model, form):
    ds = model.training_datasets.get(form)
    return ds[:] if ds is not None and len(ds) else None


def _losses(model):
    """The positive and negative losses of every training assertion, sampled negatives of
    every form and the regulariser of all individuals."""
    out, m = {}, model.module
    for form in ("class_assertion", "object_property_assertion"):
        rows = _rows(model, form)
        if rows is None:
            continue
        out[form] = m(rows, form).reshape(-1)
        out[form + "/neg"] = m(rows, form, neg=True).reshape(-1)
    ca, opa = _rows(model, "class_assertion"), _rows(model, "object_property_assertion")
    out["negatives/class"] = m.sample_class_negatives(ca).double()
    out["negatives/role"] = m.sample_role_negatives(opa).double()
    out["regularization"] = m.regularization_loss(th.arange(len(model.individual_index_dict)))
    return out


def _scores(model):
    """What the evaluation regimes rank: the loss of every (individual, class) pair and of
    every role triple, the exact scores, the ranking of all classes, the log-probabilities."""
    m = model.module
    k, n = len(model.individual_index_dict), len(model.class_index_dict)
    r = len(model.object_property_index_dict)
    cls = th.tensor([[a, c] for a in range(k) for c in range(n)])
    role = th.tensor([[a, s, b] for a in range(k) for s in range(r) for b in range(k)])
    ids = th.arange(k)
    with th.no_grad():
        return {"class_assertion": m(cls, "class_assertion").reshape(-1),
                "object_property_assertion": m(role, "object_property_assertion").reshape(-1),
                "class_assertion_score": m.class_assertion_score(cls),
                "object_property_assertion_score": m.object_property_assertion_score(role),
                "rank_classes": m.rank_classes(ids), "log_p": m.log_p(ids)}


def _pair(key, unseen=False, **options):
    return Pair(key=key, new=_new(**options), old=_old(**options), copy=_copy,
                scores=_scores, losses=_losses,
                parameters=lambda model: {k: v.detach() for k, v in _tables(model).items()},
                dataset=_dataset(unseen), epochs=3)


PAIRS = [_pair("canonical_mixture"),
         _pair("canonical_mixture/entropy", entropy_reg=0.05, init_self_mass=0.6),
         _pair("canonical_mixture/unseen", unseen=True, batch_size=3, entropy_reg=0.01)]
EXEMPT = {}
