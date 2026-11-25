"""The formulas that live in the library as parts of a region kind are used by more than the
method that publishes them.

* ``OctagonBounds`` (the measure ``"octagon bounds"`` of octagon products) on boxes written
  as octagons lies between the box measure ``Bounds`` and ``sqrt(5)`` times it (the
  documented relation to ELBE's box inclusion loss);
* ``boundary_depth`` (the measure ``"boundary and depth"``) is a swap for ELBE: the module
  of ELBE with that measure trains on its own objective, with a different loss;
* ``octagon_geometry`` is a library module (the region operations), importable from the
  library and, for the old path, from ``mowl.nn.el.octagon.geometry``;
* the lattice operations of boxes (``meet``, ``join``, ``volume`` of ``hierarchy_regions``)
  are library functions: the volume-ratio measure (``hierarchy_measures``) computes with them
  and so does the box lattice method (``mowl.nn.el.box_lattice`` re-exports them).
"""
import math

import torch as th

import mowl
mowl.init_jvm("1g")

from mowl.representation.library import octagon_geometry as G  # noqa: E402
from mowl.representation.library.hierarchy_regions import join, meet, volume  # noqa: E402
from mowl.representation.library.shapes_measures import OctagonBounds  # noqa: E402
from mowl.representation.library.shapes_regions import OctagonProduct  # noqa: E402
from mowl.representation.measures import Bounds  # noqa: E402
from mowl.representation.regions import Box  # noqa: E402
from mowl.representation.swap import build_method  # noqa: E402


def test_octagon_bounds_on_boxes_is_between_the_box_measure_and_sqrt5_times_it():
    th.manual_seed(0)
    ca, oa = th.randn(50, 2, 2), th.rand(50, 2, 2) + 0.1
    cb, ob = th.randn(50, 2, 2), th.rand(50, 2, 2) + 0.1
    boxes = Box(ca.reshape(50, 4), oa.reshape(50, 4)), Box(cb.reshape(50, 4), ob.reshape(50, 4))
    octagons = [OctagonProduct(G.from_box(c, o)) for c, o in ((ca, oa), (cb, ob))]
    octagon_cost = OctagonBounds().inside(*octagons)
    box_cost = Bounds().inside(*boxes)
    assert (box_cost > 0).all()
    assert (octagon_cost >= box_cost - 1e-6).all()
    assert (octagon_cost <= math.sqrt(5) * box_cost + 1e-6).all()


def test_boundary_and_depth_is_a_measure_swap_of_elbe():
    rows = th.tensor([[0, 1], [1, 2]])
    th.manual_seed(0)
    plain = build_method("elbe", 5, 2, 0, 4)
    th.manual_seed(0)
    swapped = build_method("elbe", 5, 2, 0, 4, measure="boundary and depth")
    assert swapped.parts["measure"] == "boundary and depth"
    a, b = plain(rows, "gci0"), swapped(rows, "gci0")
    assert th.isfinite(b).all() and not th.allclose(a, b)
    b.sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in swapped.parameters())


def test_the_old_octagon_geometry_path_is_the_library_module():
    from mowl.nn.el.octagon import geometry as old
    assert old.tighten is G.tighten and old.directions_like is G.directions_like


def test_the_box_lattice_operations_are_library_functions():
    th.manual_seed(1)
    lo_a, lo_b = th.rand(4, 3) * 0.5, th.rand(4, 3) * 0.5
    hi_a, hi_b = lo_a + th.rand(4, 3) * 0.5, lo_b + th.rand(4, 3) * 0.5
    assert th.equal(meet(lo_a, hi_a, lo_b, hi_b)[0], th.maximum(lo_a, lo_b))
    assert th.equal(join(lo_a, hi_a, lo_b, hi_b)[1], th.maximum(hi_a, hi_b))
    assert th.allclose(volume(lo_a, hi_a), (hi_a - lo_a).prod(dim=-1))
    assert volume(th.ones(1, 2), th.zeros(1, 2)).item() == 0.0  # an empty box has no volume
