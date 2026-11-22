"""Box2EL, MultiBoxEL, MultiELBE and the volume-based MultiBoxEL: the combination of parts that
:mod:`mowl.models` builds, against the direct implementation of :mod:`mowl.baselines`.

Every pair starts the two models from the same numbers (the state of the direct module is loaded
into the combination, which also checks that models saved by the direct module load), and
compares the losses of every axiom form of the training data (and the negative losses of the
forms that have one), the scores that the evaluation ranks, and a run of three epochs.
"""
import os

import torch as th

from tests.baselines.harness import Pair

FIXTURES = os.path.join(os.path.dirname(os.path.dirname(__file__)), "geometric", "fixtures")

#: The tables of the modules, under the names the earlier modules give them.
BOX_TABLES = ("class_center", "class_offset", "bump_classes", "head_center", "head_offset",
              "tail_center", "tail_offset", "ind_center", "ind_offset", "bump_individuals")
ELBE_TABLES = ("class_embed", "class_offset", "rel_embed", "ind_embed", "ind_offset")
VOLUME_TABLES = ("class_embed", "rel_embed", "ind_embed")


def _t3():
    from mowl.datasets import PathDataset
    return PathDataset(os.path.join(FIXTURES, "t3.owl"))


def _copy(old, new):
    new.module.load_state_dict(old.module.state_dict())


def _tables(names):
    def parameters(model):
        """The tables of the module under the names both implementations give them."""
        out = {}
        for name in names:
            table = getattr(model.module, name, None)
            if table is not None:
                out[name] = table.weight if hasattr(table, "weight") else table
        return out
    return parameters


def _losses(model, forms=None):
    """The loss of every axiom form of the training data and the negative loss of the forms
    that have one."""
    out = {}
    module = model.module
    for form, dataset in model.training_datasets.items():
        if len(dataset) == 0 or (forms is not None and form not in forms):
            continue
        rows = dataset[:]
        out[form] = module(rows, form)
        if form in getattr(module, "neg_capable_gcis", ()):
            out[form + "/neg"] = module(rows, form, neg=True)
    return out


def _ranking_scores(model):
    """What a ranking evaluation reads: the loss of every pair of classes as ``C subclass of
    D``, of every role between two classes as ``C subclass of exists r.D``, and of every role
    filler as ``exists r.C subclass of D``, in evaluation mode."""
    module = model.module.eval()
    n = min(len(model.class_index_dict), 12)
    ids = th.arange(n)
    pairs = th.cartesian_prod(ids, ids)
    out = {"gci0": module(pairs, "gci0")}
    if len(model.object_property_index_dict):
        r = th.zeros(n * n, dtype=th.long)
        out["gci2"] = module(th.stack([pairs[:, 0], r, pairs[:, 1]], dim=1), "gci2")
        out["gci3"] = module(th.stack([r, pairs[:, 0], pairs[:, 1]], dim=1), "gci3")
    return out


def _with_negatives(model):
    """The DELE negative losses of all forms switched on (Box2EL, ``all_negatives``)."""
    model.module.enable_all_negative_losses()
    return model


# -- Box2EL --------------------------------------------------------------------------------

def _box2el(cls_path):
    def build(dataset, **options):
        import importlib
        module, name = cls_path.rsplit(".", 1)
        cls = getattr(importlib.import_module(module), name)
        return cls(dataset, embed_dim=6, batch_size=64, margin=0.05, delta=2.0, reg_factor=0.2,
                   **options)
    return build


def _both(path_new, path_old, wrap=None, **options):
    new, old = _box2el(path_new), _box2el(path_old)
    if wrap is None:
        return (lambda d: new(d, **options)), (lambda d: old(d, **options))
    return (lambda d: wrap(new(d, **options))), (lambda d: wrap(old(d, **options)))


_B2_NEW = "mowl.models.boxsquaredel.model.BoxSquaredEL"
_B2_OLD = "mowl.baselines.models.boxsquaredel.model.BoxSquaredEL"

_b2_new, _b2_old = _both(_B2_NEW, _B2_OLD)
_b2all_new, _b2all_old = _both(_B2_NEW, _B2_OLD, wrap=_with_negatives)

# -- MultiBoxEL ----------------------------------------------------------------------------

_MB_NEW = "mowl.models.multiboxel.model.MultiBoxEL"
_MB_OLD = "mowl.baselines.models.multiboxel.model.MultiBoxEL"


def _multi(path, **options):
    def build(dataset):
        import importlib
        module, name = path.rsplit(".", 1)
        cls = getattr(importlib.import_module(module), name)
        return cls(dataset, embed_dim=4, batch_size=64, **options)
    return build


def _structural_losses(model):
    """The structural reading needs the definition table and the existence targets of the
    ontology: installed, as ``train`` does, before the losses are read; the penalties are the
    regularisation and the extra loss."""
    from mowl.nn.el.structural import build_structural_tables
    table, targets, top = build_structural_tables(
        model, k=model.n_components, max_components=model.max_components,
        enabled=model.structural_definitions)
    model.module.set_structural_tables(table, targets, top_idx=top)
    out = _losses(model)
    out["regularization"] = th.as_tensor(model.module.regularization_loss())
    out["structural_extra"] = th.as_tensor(model.module.structural_extra_loss())
    return out


def _point_losses(model):
    """The point readings: the losses of the forms (zero in training mode, as the objective
    reads them) and the loss of the points of the first epoch."""
    out = _losses(model)
    model.module.train()
    out.update({f"train/{k}": v for k, v in _losses(model).items()})
    out["points"] = th.as_tensor(model.additional_training_loss(0))
    return out


_mb_new = _multi(_MB_NEW, n_components=2, margin=0.05, delta=2.0, reg_factor=0.2)
_mb_old = _multi(_MB_OLD, n_components=2, margin=0.05, delta=2.0, reg_factor=0.2)
_mbs_new = _multi(_MB_NEW, n_components=2, margin=0.05, delta=2.0, reg_factor=0.2,
                  component_aggregation="soft", tau_start=0.5, diversity_reg=0.1,
                  component_reg=0.1)
_mbs_old = _multi(_MB_OLD, n_components=2, margin=0.05, delta=2.0, reg_factor=0.2,
                  component_aggregation="soft", tau_start=0.5, diversity_reg=0.1,
                  component_reg=0.1)
_mbstruct = dict(n_components=2, objective="structural", margin=0.1, max_components=8,
                 neg_sampling_gcis=[], witness_weight=0.5)
_mbt_new, _mbt_old = _multi(_MB_NEW, **_mbstruct), _multi(_MB_OLD, **_mbstruct)
_mbpoint = dict(n_components=2, objective="mixed", points_per_epoch=64, points_seed=1,
                margin=0.05)
_mbp_new, _mbp_old = _multi(_MB_NEW, **_mbpoint), _multi(_MB_OLD, **_mbpoint)


def _supervised(build):
    def make(dataset):
        model = build(dataset)
        model._ensure_point_supervision()
        return model
    return make


# -- MultiELBE -----------------------------------------------------------------------------

_ME_NEW = "mowl.models.multielbe.model.MultiELBE"
_ME_OLD = "mowl.baselines.models.multielbe.model.MultiELBE"
_me_new = _multi(_ME_NEW, n_components=2, margin=0.1)
_me_old = _multi(_ME_OLD, n_components=2, margin=0.1)
_mes_new = _multi(_ME_NEW, n_components=2, margin=0.1, component_aggregation="soft",
                  tau_start=0.5, diversity_reg=0.1, component_reg=0.1)
_mes_old = _multi(_ME_OLD, n_components=2, margin=0.1, component_aggregation="soft",
                  tau_start=0.5, diversity_reg=0.1, component_reg=0.1)

# -- the volume-based MultiBoxEL --------------------------------------------------------------

_MC_NEW = "mowl.models.mcmultibox.model.VolumeMultiBoxEL"
_MC_OLD = "mowl.baselines.models.mcmultibox.model.VolumeMultiBoxEL"


def _volume(path):
    def build(dataset):
        import importlib
        module, name = path.rsplit(".", 1)
        cls = getattr(importlib.import_module(module), name)
        return cls(dataset, embed_dim=3, num_boxes=2, num_points=256, num_batches=2,
                   temperature=0.5, point_chunk_size=64, learning_rate=0.01)
    return build


def _volume_scores(model):
    """The evaluation distance of Section 7.1.2 for every pair of classes, and the training
    loss that ranks ``C subclass of exists r.D``."""
    module = model.module.eval()
    n = min(len(model.class_index_dict), 10)
    ids = th.arange(n)
    pairs = th.cartesian_prod(ids, ids)
    out = {"gci0": module.distance(pairs, "gci0")}
    if len(model.object_property_index_dict):
        r = th.zeros(n * n, dtype=th.long)
        out["gci2"] = module.distance(th.stack([pairs[:, 0], r, pairs[:, 1]], dim=1), "gci2")
    return out


def _volume_losses(model):
    """The losses of the forms the thesis defines (the others are not trained on)."""
    return _losses(model, forms=model.TRAINABLE_GCIS)


PAIRS = [
    Pair("box2el", _b2_new, _b2_old, _copy, _ranking_scores, losses=_losses,
         parameters=_tables(BOX_TABLES)),
    Pair("box2el/all-negatives", _b2all_new, _b2all_old, _copy, _ranking_scores, losses=_losses,
         parameters=_tables(BOX_TABLES)),
    Pair("multiboxel", _mb_new, _mb_old, _copy, _ranking_scores, losses=_losses,
         parameters=_tables(BOX_TABLES)),
    Pair("multiboxel/soft-aggregation", _mbs_new, _mbs_old, _copy, _ranking_scores,
         losses=_losses, parameters=_tables(BOX_TABLES)),
    Pair("multiboxel/structural", _mbt_new, _mbt_old, _copy, _ranking_scores,
         losses=_structural_losses, dataset=_t3, parameters=_tables(BOX_TABLES)),
    Pair("multiboxel/point", _supervised(_mbp_new), _supervised(_mbp_old), _copy,
         _ranking_scores, losses=_point_losses, dataset=_t3, parameters=_tables(BOX_TABLES)),
    Pair("multielbe", _me_new, _me_old, _copy, _ranking_scores, losses=_losses,
         parameters=_tables(ELBE_TABLES)),
    Pair("multielbe/soft-aggregation", _mes_new, _mes_old, _copy, _ranking_scores,
         losses=_losses, parameters=_tables(ELBE_TABLES)),
    Pair("mcmultibox", _volume(_MC_NEW), _volume(_MC_OLD), _copy, _volume_scores,
         losses=_volume_losses, parameters=_tables(VOLUME_TABLES), epochs=2),
]

EXEMPT = {}
