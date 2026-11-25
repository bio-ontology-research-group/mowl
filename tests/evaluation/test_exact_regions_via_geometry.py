"""The exact-model check reads the class regions of every region family through the
geometry of the module; the per-module readers that it replaced are the oracles, and the
numbers are identical (``array_equal``, not ``allclose``)."""
import numpy as np
import pytest
import torch as th

from mowl.evaluation import exact_model
from mowl.evaluation.region_access import evaluation_regions, geometry_evaluation_regions
from mowl.evaluation.regions import (BoxRegion, OctagonRegion, PrismRegion, UnionRegion)
from mowl.nn import (BoxELModule, ELKEmModule, MultiBoxELModule, MultiELBEModule,
                     OctagonELModule, PrismModule)
from mowl.nn.el.mcmultibox.module import VolumeMultiBoxELModule
from mowl.representation import Box, CornerBox
from mowl.representation.library.boxrel_boxes import WidthBox

N = 9


def _perturb(module, seed=0, scale=0.4):
    g = th.Generator().manual_seed(seed)
    with th.no_grad():
        for p in module.parameters():
            p.add_(scale * th.randn(p.shape, generator=g, dtype=p.dtype))
    return module


def _elkem():
    th.manual_seed(1)
    module = _perturb(ELKEmModule(N, 2, 2, embed_dim=4))
    module.bot_mask[1] = True
    module.top_mask[2] = True
    return module


def _boxel():
    th.manual_seed(2)
    return _perturb(BoxELModule(N, 2, 2, embed_dim=4))


MODULES = {
    "ELK-Em": _elkem,
    "BoxEL": _boxel,
    "MultiBoxEL": lambda: _perturb(MultiBoxELModule(N, 2, embed_dim=3, n_components=2)),
    "MultiELBE": lambda: _perturb(MultiELBEModule(N, 2, embed_dim=3, n_components=2)),
    "VolumeMultiBoxEL": lambda: _perturb(VolumeMultiBoxELModule(N, 2, embed_dim=3,
                                                                num_boxes=3, num_points=32)),
    "Prism": lambda: _perturb(PrismModule(N, 2, 2, embed_dim=5)),
    "Octagon": lambda: _perturb(OctagonELModule(N, 2, 2, embed_dim=4)),
}


def _same(new, old):
    assert type(new) is type(old)
    if isinstance(new, BoxRegion):
        assert np.array_equal(new.lo, old.lo) and np.array_equal(new.hi, old.hi)
    elif isinstance(new, UnionRegion):
        assert len(new.pieces) == len(old.pieces)
        for a, b in zip(new.pieces, old.pieces):
            _same(a, b)
    elif isinstance(new, PrismRegion):
        assert np.array_equal(new.h, old.h)
        assert np.array_equal(new.lo, old.lo) and np.array_equal(new.hi, old.hi)
    elif isinstance(new, OctagonRegion):
        assert np.array_equal(new.h, old.h)
    else:
        raise AssertionError(type(new))


def _legacy_boxel(module):
    lo = module.min_embedding.weight.detach().double().numpy()
    hi = lo + th.exp(module.delta_embedding.weight).detach().double().numpy()
    return [BoxRegion(a, b) for a, b in zip(lo, hi)]


def _legacy_elkem(module):
    with th.no_grad():
        lower, upper = module.class_box(th.arange(N))
    return [BoxRegion(lo, hi) for lo, hi in zip(lower.double().numpy(), upper.double().numpy())]


def _legacy_union(cen, off, active=None):
    cen, off = cen.detach().double().numpy(), off.detach().double().numpy()
    return [UnionRegion([BoxRegion(c[k] - o[k], c[k] + o[k]) for k in range(c.shape[0])
                         if active is None or active[i][k]], dim=c.shape[1])
            for i, (c, o) in enumerate(zip(cen, off))]


def _legacy_prism(module):
    r = module.all_class_regions()
    h, cen, w = (x.detach().double().numpy() for x in (r.h, r.cen, r.w))
    return [PrismRegion(h[i], cen[i] - w[i], cen[i] + w[i]) for i in range(h.shape[0])]


LEGACY = {
    "ELK-Em": _legacy_elkem,
    "BoxEL": _legacy_boxel,
    "MultiBoxEL": lambda m: _legacy_union(*(lambda r: (r.cen, r.w, r.active))(
        m.all_class_regions())),
    "MultiELBE": lambda m: _legacy_union(*m.class_boxes(th.arange(N))),
    "VolumeMultiBoxEL": lambda m: _legacy_union(*m.concept_boxes(th.arange(N))),
    "Prism": _legacy_prism,
    "Octagon": lambda m: [OctagonRegion(h) for h in m.all_class_bounds().detach()
                          .double().numpy()],
}


@pytest.mark.parametrize("name", sorted(MODULES))
def test_the_geometry_gives_the_numbers_of_the_module_readers(name):
    # the readers that the exact-model check used before it read the geometry
    module = MODULES[name]()
    with th.no_grad():
        old = LEGACY[name](module)
    new = geometry_evaluation_regions(module)
    assert len(new) == len(old)
    for a, b in zip(new, old):
        _same(a, b)


@pytest.mark.parametrize("name", sorted(MODULES))
def test_find_reader_gives_a_reader_for_every_module(name):
    module = MODULES[name]()
    regions = exact_model.find_reader(module)(module)
    assert len(regions) == N


def test_no_module_has_a_reader_of_its_own_any_more():
    # MultiBoxEL with the structural objective had one (its defined classes are the
    # intersections of their definers); its geometry has them now
    assert exact_model.REGION_READERS == {}


def _structural_multibox(n_components=2, max_components=4, perturb=0.4):
    from mowl.nn.el.structural import DefinitionTable, ExistenceTargets
    th.manual_seed(3)
    module = MultiBoxELModule(N, 2, embed_dim=3, n_components=n_components,
                              objective="structural", max_components=max_components)
    gci0 = th.tensor([[3, 0], [3, 1], [4, 1], [4, 2], [5, 3], [5, 2], [6, 0]])
    gci1 = th.tensor([[0, 1, 3], [1, 2, 4], [3, 2, 5]])
    table = DefinitionTable.from_normal_forms(
        gci0=gci0, gci1=gci1, n_classes=N, k=n_components, max_components=max_components,
        top_idx=None, bottom_idx=None, enabled=True)
    module.set_structural_tables(table, ExistenceTargets(
        classes=th.arange(N), pairs=th.tensor([[0, 1]])))
    assert table.n_defined >= 2     # 5 = 3 and 2 needs 8 slots
    _perturb(module, seed=5, scale=perturb)
    return module


def _legacy_structural(module):
    # the reader that exact_model.py had for MultiBoxELModule: the regions of the module with
    # the definitions resolved, the active slots only
    with th.no_grad():
        r = module.all_class_regions()
    return _legacy_union(r.cen, r.w, r.active.numpy())


def test_structural_multibox_defined_classes_are_read_through_the_geometry():
    # the oracle is the old reader of the module; the numbers are identical
    for components, slots in ((2, 4), (1, 1), (2, 8)):
        module = _structural_multibox(components, slots)
        old = _legacy_structural(module)
        new = geometry_evaluation_regions(module)
        assert len(new) == len(old) == N
        for a, b in zip(new, old):
            _same(a, b)
        # a defined class (3 = 0 and 1) is not its own row: it has the intersection of the
        # regions of its definers
        row = module.class_boxes(th.tensor([3]))
        own = _legacy_union(*row)[0]
        assert not (len(own.pieces) == len(new[3].pieces) and all(
            np.array_equal(p.lo, q.lo) for p, q in zip(own.pieces, new[3].pieces)))


def test_structural_multibox_exact_model_reads_the_defined_classes():
    module = _structural_multibox()
    reader = exact_model.find_reader(module)
    assert reader is not None
    regions = reader(module)
    old = _legacy_structural(module)
    for a, b in zip(regions, old):
        _same(a, b)


def test_elkem_empty_and_full_boxes_say_so():
    module = _elkem()
    regions = geometry_evaluation_regions(module)
    assert regions[1].is_empty()
    assert not regions[2].is_empty() and regions[2].hi.min() > 1e5


def test_boxel_corners_are_exact():
    # the lower corner is the parameter itself and the upper one its sum with exp(delta) in
    # double precision, which a centre and a half-width cannot give back
    module = _boxel()
    region = module_region(module)
    assert isinstance(region, WidthBox)
    new = geometry_evaluation_regions(module)
    low = module.min_embedding.weight.detach().double().numpy()
    assert np.array_equal(np.stack([r.lo for r in new]), low)


def module_region(module):
    from mowl.representation import geometry_of
    return geometry_of(module).classes(th.arange(N))


def test_boxes_by_centre_and_offset_are_read_in_double_precision():
    centre = th.tensor([[0.1, 0.2]])
    offset = th.tensor([[0.3, 0.05]])
    (region,) = evaluation_regions(Box(centre, offset))
    assert np.array_equal(region.lo, (centre.double() - offset.double()).numpy()[0])
    corner = CornerBox.from_corners(th.tensor([[0.0, 1.0]]), th.tensor([[2.0, 3.0]]))
    (region,) = evaluation_regions(corner)
    assert region.lo.tolist() == [0.0, 1.0] and region.hi.tolist() == [2.0, 3.0]


def test_a_family_without_a_converter_is_an_error():
    from mowl.representation.library.logic_fuzzy import FuzzySet, Product
    with pytest.raises(TypeError, match="FuzzySet"):
        evaluation_regions(FuzzySet(th.rand(2, 3), Product()))
