"""Internals of the direct implementation of MultiBoxEL (:mod:`mowl.baselines`)."""
import pytest
import torch

import mowl.baselines.nn.el.multiboxel.losses as MBL


def test_resolve_n_components_rejects_ambiguous_embedding() -> None:
    bare_emb = torch.nn.Embedding(6, 8)
    with pytest.raises(ValueError):
        MBL._resolve_n_components(bare_emb)
    with pytest.raises(ValueError):
        MBL._resolve_n_components(bare_emb, embed_dim=4)
    assert MBL._resolve_n_components(bare_emb, embed_dim=8) == 1
    assert MBL._resolve_n_components(bare_emb, n_components=2, embed_dim=4) == 2
    bare_emb.embed_dim = 8
    assert MBL._resolve_n_components(bare_emb) == 1
