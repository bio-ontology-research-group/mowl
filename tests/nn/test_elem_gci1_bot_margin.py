"""ELEm: the margin of gci1_bot tightens, the margin of the inclusions relaxes (published form)."""
import torch as th

import mowl
mowl.init_jvm("512m")

from mowl.nn.el.elem.module import ELEmModule  # noqa: E402


def test_the_margin_of_the_disjointness_loss_has_the_opposite_sense_of_the_inclusions():
    th.manual_seed(0)
    m = ELEmModule(3, 1, None, embed_dim=2, margin=0.0)
    with th.no_grad():
        m.class_embed.weight.copy_(th.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]]))
        m.class_rad.weight.copy_(th.tensor([[0.2], [0.2], [0.4]]))
    disjoint = th.tensor([[0, 1]])            # distance 1, radii 0.2 + 0.2: a gap of 0.6
    inside = th.tensor([[0, 2]])              # C inside the ball of E up to 0.2 - 0.4 + 0 = -0.2
    assert float(m.gci1_bot_loss(disjoint)) == 0.0
    assert float(m.gci0_loss(inside)) < 0.01
    m.margin = 0.7                            # more than the gap
    assert float(m.gci1_bot_loss(disjoint)) > 0.0         # tightens the disjointness
    assert m.margin_semantics == "slack"
    # the inclusion is relaxed by the same margin: zero loss with the margin where it was
    # positive without
    with th.no_grad():
        m.class_rad.weight.copy_(th.tensor([[0.5], [0.2], [0.4]]))
    m.margin = 0.0
    before = float(m.gci0_loss(inside))
    m.margin = 0.7
    assert before > 0.0 and float(m.gci0_loss(inside)) < before
