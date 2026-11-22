"""``margin_semantics`` of every ELModule with an inclusion margin is truthful: with a strict
margin a zero loss implies the inclusion (a gap of at least the margin), with a slack margin
it does not (a violation up to the margin costs nothing)."""
import pytest
import torch as th

from tests.nn.el_module_zoo import all_el_module_classes, make

MARGIN = 0.1
D_SIZE = 0.5
CENTRE = 6 ** -0.5  # unit norm in the 6 dimensions of the zoo (EmEL++ penalises other norms)
# stick-out of the sub-concept beyond the super-concept: half a margin, none, a gap of two
# margins inside, and two margins out
SIZES = {"half_out": D_SIZE + MARGIN / 2, "tight": D_SIZE, "gap": D_SIZE - 2 * MARGIN,
         "out": D_SIZE + 2 * MARGIN}

# module -> (parameters holding the centres, parameters holding the sizes); the sub-concept
# is class 0, the super-concept class 1; all other classes are left as they are.
GEOMETRY = {
    "ELBEModule": (("class_embed",), ("class_offset",)),
    "MultiELBEModule": (("class_embed",), ("class_offset",)),
    "RegDELBEModule": (("class_embed",), ("class_offset",)),
    "ELKEmModule": (("class_center",), ("class_offset",)),
    "TransBoxModule": (("class_center_embedding",), ("class_offset_embedding",)),
    "ELEmModule": (("class_embed",), ("class_rad",)),
    "EmELPPModule": (("class_embed",), ("class_rad",)),
    "EmELVarModule": (("class_embed",), ("class_rad",)),
    "RegDELEmModule": (("class_embed",), ("class_rad",)),
    "BoxSquaredELModule": (("class_center",), ("class_offset",)),
    "OctagonELModule": (("class_embed",), ("class_radius",)),
    "PrismModule": (("tri_center", "box_center"), ("box_offset", "tri_inradius")),
    "MultiBoxELModule": (("class_center",), ("class_offset",)),
}

# gci0 of these does not use the margin (RegD's energy has gamma1 and no margin; its margin
# serves the forms it takes over from ELBE / ELEm). RegD-ELEm is probed with gci1 and a
# repeated sub-concept (C and C are included in D when C is); RegD-ELBE takes gci1 over too, so
# it is probed with the inclusion of an individual in a class (class_assertion, ELBE's loss).
GCI1_PROBE = {"RegDELEmModule"}
ASSERTION_PROBE = {"RegDELBEModule": ("ind_embed", "ind_offset")}

# Modules built from parts whose method has no inclusion margin: the margin of the objective is
# fixed at 0 by the method, the margins of the method are its own arguments (EIKE: ``margins``,
# the margins of its ranking losses; Concept2Box: ``kge_margin`` and ``cross_margin``, the
# margins of its negative losses).
OWN_MARGINS = {"EIKEModule", "Concept2BoxModule"}

CLASSES = {c.__name__: c for c in all_el_module_classes()}


def _losses(name, margin_semantics_expected=None):
    module = make(CLASSES[name])
    setattr(module, module.margin_attribute, MARGIN)
    centres, sizes = GEOMETRY[name]
    if name in ASSERTION_PROBE:
        centres, sizes = centres + (ASSERTION_PROBE[name][0],), sizes
    out = {}
    for label, size in SIZES.items():
        with th.no_grad():
            for p in centres:
                getattr(module, p).weight.fill_(CENTRE)
            for p in sizes:
                w = getattr(module, p).weight
                w.fill_(D_SIZE)
                w[0] = size
            if name in ASSERTION_PROBE:
                w = getattr(module, ASSERTION_PROBE[name][1]).weight
                w.fill_(D_SIZE)
                w[0] = size
        if name in ASSERTION_PROBE:
            out[label] = float(module(th.tensor([[0, 1]]), "class_assertion").sum())
        elif name in GCI1_PROBE:
            out[label] = float(module(th.tensor([[0, 0, 1]]), "gci1").sum())
        else:
            out[label] = float(module(th.tensor([[0, 1]]), "gci0").sum())
    return module, out


@pytest.mark.parametrize("name", sorted(GEOMETRY))
def test_declared_margin_semantics_is_truthful(name):
    module, loss = _losses(name)
    assert module.margin_semantics in ("strict", "slack"), name
    assert loss["out"] > 0 and loss["gap"] == 0, loss  # the probe is a real inclusion test
    if module.margin_semantics == "strict":
        # zero loss only with a gap, so zero loss implies the inclusion
        assert loss["half_out"] > 0 and loss["tight"] > 0, loss
    else:
        # a sub-concept sticking out of the super-concept costs nothing
        assert loss["half_out"] == 0, loss


def test_the_survey_covers_every_module_with_an_inclusion_margin():
    declared = set()
    for n, c in CLASSES.items():
        try:
            if make(c).margin_semantics is not None:
                declared.add(n)
        except NotImplementedError:  # an abstract base
            pass
    assert declared <= set(GEOMETRY) | OWN_MARGINS, declared - set(GEOMETRY)


# The margin of these modules is a margin of the negative loss or of a ranking, and does not
# enter the loss of an inclusion, so they declare no ``margin_semantics``.
NEGATIVE_ONLY = {"DiskEmbeddingsModule": "gci0", "HyperbolicConesModule": "gci0",
                 "ShadowConesModule": "gci0", "OrderEmbeddingsModule": "gci0",
                 "BoxLatticeModule": "gci0", "GumbelBoxModule": "gci0"}


@pytest.mark.parametrize("name", sorted(NEGATIVE_ONLY))
def test_modules_whose_margin_is_not_an_inclusion_margin_declare_none(name):
    module = make(CLASSES[name])
    assert module.margin_semantics is None
    data = th.tensor([[0, 1], [2, 3]])
    before = module(data, "gci0").detach().clone()
    module.margin = module.margin + 0.5
    assert th.equal(module(data, "gci0").detach(), before)


def test_every_module_with_a_margin_is_accounted_for():
    for name, cls in CLASSES.items():
        try:
            module = make(cls)
        except NotImplementedError:
            continue
        if isinstance(getattr(module, "margin", None), (int, float)):
            assert name in GEOMETRY or name in NEGATIVE_ONLY or name in OWN_MARGINS, name


def test_multiboxel_names_the_margin_of_its_objective():
    from mowl.nn.el.multiboxel.module import MultiBoxELModule
    axiom = make(MultiBoxELModule)
    assert (axiom.margin_semantics, axiom.margin_attribute) == ("slack", "gamma")
    assert MultiBoxELModule.margin_semantics.fget(
        type("S", (), {"objective": "structural"})()) == "strict"
