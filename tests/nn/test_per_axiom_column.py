"""Box2EL and MultiBoxEL return every per-axiom loss as a column (B, 1). MultiELBE is left as it
is: its tests require the shapes of ELBE at one component, and ELBE returns (B,) for two forms."""
import pytest
import torch as th

from mowl.nn import BoxSquaredELModule, MultiBoxELModule

N, R, I, D, B = 8, 3, 4, 5, 6

ROWS = {
    "gci0": 2, "gci0_bot": 1, "gci1": 3, "gci1_bot": 2, "gci2": 3, "gci3": 3,
    "gci3_bot": 2, "class_assertion": 2, "object_property_assertion": 3,
}


def _batch(form):
    gen = th.Generator().manual_seed(7)
    high = {"class_assertion": (I, N), "object_property_assertion": (I, R, I)}
    hi = [N] * ROWS[form]
    if form in high:
        hi = list(high[form])
    elif form == "gci2":
        hi = [N, R, N]
    elif form == "gci3":
        hi = [R, N, N]
    elif form == "gci3_bot":
        hi = [R, N]
    return th.stack([th.randint(0, h, (B,), generator=gen) for h in hi], dim=1)


def _modules():
    th.manual_seed(0)
    return {
        "box2el": BoxSquaredELModule(N, R, I, embed_dim=D),
        "multiboxel": MultiBoxELModule(N, R, I, embed_dim=D, n_components=2),
    }


@pytest.mark.parametrize("name", ["box2el", "multiboxel"])
@pytest.mark.parametrize("form", list(ROWS))
def test_every_loss_is_a_column(name, form):
    module = _modules()[name]
    out = module(_batch(form), form)
    assert out.shape == (B, 1), (name, form, tuple(out.shape))
    if form in module.neg_capable_gcis:
        assert module(_batch(form), form, neg=True).shape == (B, 1)


def test_role_losses_are_columns_for_box2el():
    module = _modules()["box2el"]
    assert module(th.tensor([[0, 1], [1, 2]]), "role_inclusion").shape == (2, 1)
    assert module(th.tensor([[0, 1, 2]]), "role_chain").shape == (1, 1)


@pytest.mark.parametrize("name", ["box2el", "multiboxel"])
def test_old_flat_shape_is_available(name):
    module = _modules()[name]
    module.per_axiom_column = False
    for form in ("gci0", "gci2", "class_assertion"):
        assert module(_batch(form), form).shape == (B,), form


def test_values_do_not_change_only_the_shape():
    module = _modules()["box2el"]
    column = module(_batch("gci0"), "gci0")
    module.per_axiom_column = False
    flat = module(_batch("gci0"), "gci0")
    assert th.equal(column.squeeze(1), flat)
