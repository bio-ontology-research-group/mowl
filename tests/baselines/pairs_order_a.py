"""Order and box embeddings of the class taxonomy and of lattices and graphs, as a combination of
parts and as the direct implementation: ``order_embeddings``, ``cate``, ``box_lattice``,
``gumbel_box``, ``bcbox`` (group ``recomp-order-a``)."""
import os
import tempfile

import torch as th

import mowl
mowl.init_jvm("2g")

import mowl.models as new_models  # noqa: E402
from mowl.baselines.models.bcbox import BinaryCodeBox as OldBinaryCodeBox  # noqa: E402
from mowl.baselines.models.box_lattice import BoxLattice as OldBoxLattice  # noqa: E402
from mowl.baselines.models.cate import CatE as OldCatE  # noqa: E402
from mowl.baselines.models.gumbel_box import GumbelBox as OldGumbelBox  # noqa: E402
from mowl.baselines.models.order_embeddings import OrderEmbeddings as OldOrderEmbeddings  # noqa: E402,E501
from tests.baselines.harness import Pair  # noqa: E402

EXEMPT = {}


def _path():
    return os.path.join(tempfile.mkdtemp(dir=os.environ.get("TMPDIR")), "model.pt")


def _copy(pairs):
    """``copy`` that sets each tensor of the first model to that of the second one."""
    def copy(old, new):
        with th.no_grad():
            for get in pairs:
                get(new).copy_(get(old))
    return copy


# -- the class taxonomy (order embeddings, box lattice, Gumbel boxes) -----------------------
def _named_pairs(model):
    n = len(model.named_class_index_dict)
    sub, sup = th.meshgrid(th.arange(n), th.arange(n), indexing="ij")
    return th.stack([sub.reshape(-1), sup.reshape(-1)], dim=1)


def _taxonomy_scores(model):
    rows = _named_pairs(model)
    with th.no_grad():
        model.module.eval()
        return {"gci0": model.module(rows, "gci0").reshape(-1),
                "gci0/neg": model.module(rows, "gci0", neg=True).reshape(-1)}


def _taxonomy_losses(model):
    out = {}
    data = model.training_datasets["gci0"]
    with th.no_grad():
        out["pos"] = model.module(data[:], "gci0").reshape(-1)
        neg = model.generate_negatives("gci0", data)
        out["neg"] = model.module(neg, "gci0", neg=True).reshape(-1)
        regularization = getattr(model.module, "regularization_loss", None)
        # a module without regularisers has no such method (the direct ones), or returns 0
        out["regularization"] = th.as_tensor(
            0.0 if regularization is None else regularization()).reshape(-1)
        extra = model.additional_training_loss(0)
        if extra is not None:
            out["unary"] = extra.reshape(-1)
    return out


def _taxonomy_parameters(model):
    return {"class_embed": model.module.class_embed.weight}


def _taxonomy(key, old, new, **options):
    options = dict(embed_dim=8, learning_rate=0.05, **options)
    return Pair(
        key=key,
        old=lambda ds: old(ds, model_filepath=_path(), **options),
        new=lambda ds: new(ds, model_filepath=_path(), **options),
        copy=_copy([lambda m: m.module.class_embed.weight]),
        scores=_taxonomy_scores, losses=_taxonomy_losses,
        parameters=_taxonomy_parameters)


PAIRS = [
    _taxonomy("order_embeddings", OldOrderEmbeddings, new_models.OrderEmbeddings, margin=1.0),
    _taxonomy("order_embeddings/margin-half", OldOrderEmbeddings, new_models.OrderEmbeddings,
              margin=0.5),
    _taxonomy("box_lattice", OldBoxLattice, new_models.BoxLattice),
    _taxonomy("box_lattice/wordnet-setting", OldBoxLattice, new_models.BoxLattice,
              min_delta=1e-6, max_regularization=0.005, unary_weight=1.0),
    _taxonomy("box_lattice/earlier-guard", OldBoxLattice, new_models.BoxLattice,
              zero_at_inclusion=False, freeze_delta_negatives=False),
    _taxonomy("gumbel_box", OldGumbelBox, new_models.GumbelBox),
    _taxonomy("gumbel_box/earlier-guard", OldGumbelBox, new_models.GumbelBox,
              log_guard="epsilon", beta=0.05, max_regularization=0.01),
]


# -- CatE ------------------------------------------------------------------------------------
def _cate_ids(model):
    n = len(model.lattice)
    sub, sup = th.meshgrid(th.arange(n), th.arange(n), indexing="ij")
    return sub.reshape(-1), sup.reshape(-1)


def _cate_scores(model):
    sub, sup = _cate_ids(model)
    with th.no_grad():
        return {"edge": model.module.order_violation(sub, sup),
                "edge-rows": model.module(th.stack([sub, sup], 1), "edge").reshape(-1)}


def _cate_losses(model):
    edges = model.lattice.edges.to(model.device)
    with th.no_grad():
        pos = model.module.order_violation(edges[:, 0], edges[:, 1])
        return {"edges": pos, "loss": model.compute_loss(edges).reshape(-1)}


def _cate_copy(old, new):
    assert th.equal(old.lattice.edges, new.lattice.edges), "the two lattices differ"
    with th.no_grad():
        new.module.node_embedding.weight.copy_(old.module.node_embedding.weight)


def _cate(key, **options):
    options = dict(embed_dim=8, learning_rate=0.05, **options)
    return Pair(
        key=key,
        old=lambda ds: OldCatE(ds, model_filepath=_path(), **options),
        new=lambda ds: new_models.CatE(ds, model_filepath=_path(), **options),
        copy=_cate_copy, scores=_cate_scores, losses=_cate_losses,
        parameters=lambda m: {"node_embedding": m.module.node_embedding.weight})


PAIRS += [
    _cate("cate"),
    _cate("cate/reference-code-loss", negative_aggregation="mean", filter_negatives=False,
          cyclic_lr=False, p=1),
]


# -- BCBox -----------------------------------------------------------------------------------
def _bcbox_scores(model):
    n = model.num_nodes
    return {"energies": model.energy_matrix().reshape(-1),
            "evaluation_model": model.evaluation_model(_named_rows(model), "gci0").reshape(-1)}


def _named_rows(model):
    n = len(model.dataset.classes)
    sub, sup = th.meshgrid(th.arange(n), th.arange(n), indexing="ij")
    return th.stack([sub.reshape(-1), sup.reshape(-1)], dim=1)


def _bcbox_losses(model):
    edges = model.edge_index.to(model.device)
    with th.no_grad():
        return {"energy": model.module(edges).reshape(-1),
                "loss": model.compute_loss(edges).reshape(-1)}


def _bcbox_copy(old, new):
    with th.no_grad():
        new.module.box_min.copy_(old.module.box_min)
        new.module.box_max.copy_(old.module.box_max)
        if old.module.code_logits is not None:
            new.module.code_logits.copy_(old.module.code_logits)


def _bcbox_parameters(model):
    out = {"box_min": model.module.box_min, "box_max": model.module.box_max}
    if model.module.code_logits is not None:
        out["code_logits"] = model.module.code_logits
    return out


def _bcbox(key, **options):
    options = dict(embed_dim=6, learning_rate=0.05, num_negs=4, **options)
    return Pair(
        key=key,
        old=lambda ds: OldBinaryCodeBox(ds, model_filepath=_path(), **options),
        new=lambda ds: new_models.BinaryCodeBox(ds, model_filepath=_path(), **options),
        copy=_bcbox_copy, scores=_bcbox_scores, losses=_bcbox_losses,
        parameters=_bcbox_parameters)


PAIRS += [
    _bcbox("bcbox", variant="vbc", code_reg=0.1),
    _bcbox("bcbox/gbc", variant="gbc", num_codes=3, code_dims=4, code_reg=0.1),
    _bcbox("bcbox/gbox", variant="gbox", self_adversarial=False),
]
