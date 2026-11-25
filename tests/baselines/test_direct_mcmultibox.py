"""Internals of the direct implementation of the volume-based MultiBoxEL
(:mod:`mowl.baselines`)."""
import torch as th

from mowl.baselines.nn.el.mcmultibox.module import VolumeMultiBoxELModule


def make_module(n_classes=4, n_rels=2, d=2, n=2, points=4096, **kw):
    kw.setdefault("temperature", 0.02)
    kw.setdefault("bound", 4.0)
    return VolumeMultiBoxELModule(n_classes, n_rels, embed_dim=d, num_boxes=n,
                                  num_points=points, **kw)


def test_ranking_computes_each_distinct_operand_once():
    m = make_module(n_classes=40, n_rels=2, d=2, n=2, points=512, point_chunk_size=128)
    calls = []
    original = m._chi

    def counting(p, c, o):
        calls.append(len(c))
        return original(p, c, o)

    m._chi = counting
    data = th.stack([th.zeros(40, dtype=th.long), th.zeros(40, dtype=th.long),
                     th.arange(40)], dim=1)
    m.distance(data, "gci2")
    # source operand: one distinct class, evaluated once per point chunk (4), not 40 times
    assert sum(calls) == 4
