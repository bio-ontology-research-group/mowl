"""``init_boxes`` on an ontology whose normalisation introduces auxiliary classes.

The models index the dataset classes followed by the auxiliary classes created by
normalisation, so the module tables have more rows than ``dataset.class_to_id``.
``ExactBoxInit.apply_to_module`` must match rows by IRI and leave the auxiliary
rows at their random initialisation.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

import mowl
mowl.init_jvm("2g")

from org.semanticweb.owlapi.model import IRI

from mowl.datasets import PathDataset
from mowl.geometric import ExactBoxInit
from mowl.models import BoxSquaredEL, ELBE, MultiBoxEL
from mowl.owlapi import OWLAPIAdapter
from tests.geometric.helpers import NS

AUX_PREFIX = "http://mowl.borg/aux/"


def _nested_dataset(directory: Path) -> PathDataset:
    """Write ``A ⊑ ∃r.(B ⊓ C)`` and ``B ⊓ ∃r.C ⊑ D`` to ``directory`` and load it."""
    adapter = OWLAPIAdapter()
    mgr = adapter.owl_manager
    ont = adapter.create_ontology("http://mowl.borg/geo/nested")
    cls = {n: adapter.create_class(NS + n) for n in "ABCD"}
    r = adapter.create_object_property(NS + "r")
    mgr.addAxiom(ont, adapter.create_subclass_of(
        cls["A"],
        adapter.create_object_some_values_from(
            r, adapter.create_object_intersection_of(cls["B"], cls["C"]))))
    mgr.addAxiom(ont, adapter.create_subclass_of(
        adapter.create_object_intersection_of(
            cls["B"], adapter.create_object_some_values_from(r, cls["C"])),
        cls["D"]))
    path = directory / "nested.owl"
    mgr.saveOntology(ont, IRI.create(path.resolve().as_uri()))
    return PathDataset(str(path))


def _tables(model):
    module = model.module
    center = module.class_embed if isinstance(model, ELBE) else module.class_center
    return center.weight.detach().clone(), module.class_offset.weight.detach().clone()


@pytest.mark.parametrize("model_cls,k", [(BoxSquaredEL, 1), (ELBE, 1), (MultiBoxEL, 2)])
def test_init_boxes_with_auxiliary_classes(scratch_dir: Path, model_cls, k: int) -> None:
    ds = _nested_dataset(scratch_dir)
    d = 4
    kwargs = {"n_components": k} if model_cls is MultiBoxEL else {}
    init = ExactBoxInit.from_dataset(ds, d, n_components=k)

    torch.manual_seed(7)
    plain = model_cls(ds, embed_dim=d, model_filepath=str(scratch_dir / "plain.pt"), **kwargs)
    torch.manual_seed(7)
    model = model_cls(ds, embed_dim=d, init_boxes=init, anchor_weight=0.5,
                      model_filepath=str(scratch_dir / "init.pt"), **kwargs)

    cid = model.class_index_dict
    aux_rows = [idx for iri, idx in cid.items() if str(iri).startswith(AUX_PREFIX)]
    assert aux_rows, "fixture must produce auxiliary classes"
    assert len(cid) > len(ds.class_to_id)

    c_init, o_init = _tables(model)
    c_plain, o_plain = _tables(plain)
    centers, offsets, mask = init.as_tensors()
    n_named = 0
    for iri, init_row in init.class_to_id.items():
        if not bool(mask[init_row]):
            continue
        row = cid[iri]
        assert torch.equal(c_init[row], centers[init_row].reshape(-1)), iri
        assert torch.equal(o_init[row], offsets[init_row].reshape(-1)), iri
        n_named += 1
    assert n_named == int(mask.sum())
    assert torch.equal(c_init[aux_rows], c_plain[aux_rows])
    assert torch.equal(o_init[aux_rows], o_plain[aux_rows])

    anchor = init.anchor_loss(model.module)
    assert torch.isfinite(anchor)
    assert float(anchor.item()) == 0.0

    model.train(epochs=1, validate_every=10)
    for param in model.module.parameters():
        assert torch.isfinite(param).all()
    assert torch.isfinite(init.anchor_loss(model.module))
    assert torch.isfinite(model.get_regularization_loss())


def test_apply_without_class_index_dict_rejects_row_mismatch(scratch_dir: Path) -> None:
    from mowl.geometric.errors import UnsupportedRegion

    ds = _nested_dataset(scratch_dir)
    init = ExactBoxInit.from_dataset(ds, 4)
    model = BoxSquaredEL(ds, embed_dim=4, model_filepath=str(scratch_dir / "m.pt"))
    with pytest.raises(UnsupportedRegion, match="class_index_dict"):
        init.apply_to_module(model.module)
    fresh = ExactBoxInit.from_dataset(ds, 4)
    with pytest.raises(UnsupportedRegion, match="class_index_dict"):
        fresh.anchor_loss(model.module)
