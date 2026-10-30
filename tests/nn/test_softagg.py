"""Unit tests for soft component aggregation and diversity regularisation (Objective O1)."""
from __future__ import annotations

import pytest
import torch

from mowl.nn import MultiBoxELModule, MultiELBEModule
from mowl.nn.el.softagg import diversity_penalty, soft_weights, softmin


def _random_batch(
    gci: str,
    cols: int,
    batch_size: int,
    n_classes: int,
    n_rels: int,
    n_inds: int,
    gen: torch.Generator,
) -> torch.Tensor:
    if gci == "gci2":
        c0 = torch.randint(0, n_classes, (batch_size, 1), generator=gen)
        r = torch.randint(0, n_rels, (batch_size, 1), generator=gen)
        c1 = torch.randint(0, n_classes, (batch_size, 1), generator=gen)
        return torch.cat([c0, r, c1], dim=1)
    if gci == "gci3":
        r = torch.randint(0, n_rels, (batch_size, 1), generator=gen)
        c0 = torch.randint(0, n_classes, (batch_size, 1), generator=gen)
        c1 = torch.randint(0, n_classes, (batch_size, 1), generator=gen)
        return torch.cat([r, c0, c1], dim=1)
    if gci == "gci3_bot":
        r = torch.randint(0, n_rels, (batch_size, 1), generator=gen)
        c0 = torch.randint(0, n_classes, (batch_size, 1), generator=gen)
        return torch.cat([r, c0], dim=1)
    if gci == "class_assertion":
        ind = torch.randint(0, n_inds, (batch_size, 1), generator=gen)
        cls = torch.randint(0, n_classes, (batch_size, 1), generator=gen)
        return torch.cat([ind, cls], dim=1)
    if gci == "object_property_assertion":
        i0 = torch.randint(0, n_inds, (batch_size, 1), generator=gen)
        r = torch.randint(0, n_rels, (batch_size, 1), generator=gen)
        i1 = torch.randint(0, n_inds, (batch_size, 1), generator=gen)
        return torch.cat([i0, r, i1], dim=1)
    return torch.randint(0, n_classes, (batch_size, cols), generator=gen)


def test_softmin_limits() -> None:
    gen = torch.Generator().manual_seed(0)
    v = torch.randn((16, 4), generator=gen)

    sm_low = softmin(v, tau=1e-4, dim=1)
    hard_min = torch.amin(v, dim=1)
    assert torch.allclose(sm_low, hard_min, atol=1e-3, rtol=1e-3)

    sm_high = softmin(v, tau=1e4, dim=1)
    mean_v = v.mean(dim=1)
    assert torch.allclose(sm_high, mean_v, atol=1e-3, rtol=1e-3)

    w = soft_weights(v, tau=0.5, dim=1)
    assert w.shape == v.shape
    assert (w > 0).all()
    assert torch.allclose(w.sum(dim=1), torch.ones(16), atol=1e-6)

    with pytest.raises(ValueError, match="tau"):
        softmin(v, tau=0.0, dim=1)
    with pytest.raises(ValueError, match="tau"):
        soft_weights(v, tau=-1.0, dim=1)


@pytest.mark.parametrize(
    "gci,cols",
    [
        ("gci0", 2),
        ("gci0_bot", 1),
        ("gci1", 3),
        ("gci1_bot", 2),
        ("gci2", 3),
        ("gci3", 3),
        ("gci3_bot", 1),
        ("class_assertion", 2),
        ("object_property_assertion", 3),
    ],
)
def test_soft_k1_equals_hard(gci: str, cols: int) -> None:
    n_classes, n_rels, d, n_inds = 10, 3, 5, 4
    gen = torch.Generator().manual_seed(123)
    batch = _random_batch(gci, cols, 64, n_classes, n_rels, n_inds, gen)

    for tau in (1.0, 0.05):
        torch.manual_seed(7)
        mb_hard = MultiBoxELModule(
            nb_ont_classes=n_classes,
            nb_rels=n_rels,
            nb_inds=n_inds,
            embed_dim=d,
            n_components=1,
            gamma=0.05,
            delta=2.0,
            component_aggregation="hard",
            tau=tau,
        )
        torch.manual_seed(7)
        mb_soft = MultiBoxELModule(
            nb_ont_classes=n_classes,
            nb_rels=n_rels,
            nb_inds=n_inds,
            embed_dim=d,
            n_components=1,
            gamma=0.05,
            delta=2.0,
            component_aggregation="soft",
            tau=tau,
        )
        out_mb_h = mb_hard(batch, gci, neg=False)
        out_mb_s = mb_soft(batch, gci, neg=False)
        assert torch.allclose(out_mb_s, out_mb_h, atol=1e-6, rtol=1e-5)
        if gci in mb_hard.neg_capable_gcis:
            out_mb_h_neg = mb_hard(batch, gci, neg=True)
            out_mb_s_neg = mb_soft(batch, gci, neg=True)
            assert torch.allclose(out_mb_s_neg, out_mb_h_neg, atol=1e-6, rtol=1e-5)

        torch.manual_seed(7)
        me_hard = MultiELBEModule(
            nb_ont_classes=n_classes,
            nb_rels=n_rels,
            nb_inds=n_inds,
            embed_dim=d,
            n_components=1,
            margin=0.1,
            component_aggregation="hard",
            tau=tau,
        )
        torch.manual_seed(7)
        me_soft = MultiELBEModule(
            nb_ont_classes=n_classes,
            nb_rels=n_rels,
            nb_inds=n_inds,
            embed_dim=d,
            n_components=1,
            margin=0.1,
            component_aggregation="soft",
            tau=tau,
        )
        out_me_h = me_hard(batch, gci, neg=False)
        out_me_s = me_soft(batch, gci, neg=False)
        assert torch.allclose(out_me_s, out_me_h, atol=1e-6, rtol=1e-5)
        if gci in me_hard.neg_capable_gcis:
            out_me_h_neg = me_hard(batch, gci, neg=True)
            out_me_s_neg = me_soft(batch, gci, neg=True)
            assert torch.allclose(out_me_s_neg, out_me_h_neg, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize(
    "gci,cols",
    [
        ("gci0", 2),
        ("gci0_bot", 1),
        ("gci1", 3),
        ("gci1_bot", 2),
        ("gci2", 3),
        ("gci3", 3),
        ("gci3_bot", 1),
        ("class_assertion", 2),
        ("object_property_assertion", 3),
    ],
)
def test_soft_converges_to_hard_k3(gci: str, cols: int) -> None:
    n_classes, n_rels, d, n_inds, k = 10, 3, 5, 4, 3
    gen = torch.Generator().manual_seed(456)
    batch = _random_batch(gci, cols, 32, n_classes, n_rels, n_inds, gen)

    torch.manual_seed(11)
    mb_hard = MultiBoxELModule(
        nb_ont_classes=n_classes,
        nb_rels=n_rels,
        nb_inds=n_inds,
        embed_dim=d,
        n_components=k,
        gamma=0.05,
        delta=2.0,
        component_aggregation="hard",
    )
    torch.manual_seed(11)
    mb_soft = MultiBoxELModule(
        nb_ont_classes=n_classes,
        nb_rels=n_rels,
        nb_inds=n_inds,
        embed_dim=d,
        n_components=k,
        gamma=0.05,
        delta=2.0,
        component_aggregation="soft",
        tau=1e-4,
    )
    assert torch.allclose(
        mb_soft(batch, gci, neg=False),
        mb_hard(batch, gci, neg=False),
        atol=1e-3,
        rtol=1e-3,
    )
    if gci in mb_hard.neg_capable_gcis:
        assert torch.allclose(
            mb_soft(batch, gci, neg=True),
            mb_hard(batch, gci, neg=True),
            atol=1e-3,
            rtol=1e-3,
        )

    torch.manual_seed(11)
    me_hard = MultiELBEModule(
        nb_ont_classes=n_classes,
        nb_rels=n_rels,
        nb_inds=n_inds,
        embed_dim=d,
        n_components=k,
        margin=0.1,
        component_aggregation="hard",
    )
    torch.manual_seed(11)
    me_soft = MultiELBEModule(
        nb_ont_classes=n_classes,
        nb_rels=n_rels,
        nb_inds=n_inds,
        embed_dim=d,
        n_components=k,
        margin=0.1,
        component_aggregation="soft",
        tau=1e-4,
    )
    assert torch.allclose(
        me_soft(batch, gci, neg=False),
        me_hard(batch, gci, neg=False),
        atol=1e-3,
        rtol=1e-3,
    )
    if gci in me_hard.neg_capable_gcis:
        assert torch.allclose(
            me_soft(batch, gci, neg=True),
            me_hard(batch, gci, neg=True),
            atol=1e-3,
            rtol=1e-3,
        )


def test_soft_gradient_reaches_all_components() -> None:
    d, k = 4, 3
    batch = torch.tensor([[0, 1]], dtype=torch.long)

    for cls, emb_attr in (
        (MultiBoxELModule, "class_center"),
        (MultiELBEModule, "class_embed"),
    ):
        torch.manual_seed(21)
        m_hard = cls(
            nb_ont_classes=2,
            nb_rels=1,
            embed_dim=d,
            n_components=k,
            component_aggregation="hard",
        )
        torch.manual_seed(21)
        m_soft = cls(
            nb_ont_classes=2,
            nb_rels=1,
            embed_dim=d,
            n_components=k,
            component_aggregation="soft",
            tau=0.5,
        )
        # Ensure target class D (row 1) has 3 well-separated components so one component is
        # never selected by hard min for any source component of C (row 0).
        with torch.no_grad():
            for m in (m_hard, m_soft):
                cen_emb = getattr(m, emb_attr)
                cen_c = cen_emb.weight[0].view(k, d)
                cen_c[0].fill_(0.0)
                cen_c[1].fill_(0.2)
                cen_c[2].fill_(0.4)
                cen_d = cen_emb.weight[1].view(k, d)
                cen_d[0].fill_(1.0)
                cen_d[1].fill_(2.0)
                cen_d[2].fill_(10.0)
                m.class_offset.weight.fill_(0.1)

        loss_hard = m_hard(batch, "gci0").sum()
        loss_hard.backward()
        grad_d_hard = getattr(m_hard, emb_attr).weight.grad[1].view(k, d)
        norms_hard = torch.linalg.norm(grad_d_hard, dim=-1)
        assert (norms_hard == 0.0).any()

        loss_soft = m_soft(batch, "gci0").sum()
        loss_soft.backward()
        grad_d_soft = getattr(m_soft, emb_attr).weight.grad[1].view(k, d)
        norms_soft = torch.linalg.norm(grad_d_soft, dim=-1)
        assert (norms_soft > 0.0).all()


def test_diversity_penalty_zero_when_disjoint_positive_when_nested() -> None:
    # Disjoint components: class 0 has [0, 2]^2 and [10, 12]^2; class 1 has [-5, -3]^2 and [5, 7]^2
    cen_disjoint = torch.tensor(
        [
            [[1.0, 1.0], [11.0, 11.0]],
            [[-4.0, -4.0], [6.0, 6.0]],
        ]
    )
    off_disjoint = torch.ones((2, 2, 2))
    pen_disjoint = diversity_penalty(cen_disjoint, off_disjoint)
    assert pen_disjoint.shape == ()
    assert float(pen_disjoint.item()) == 0.0

    # Nested components: class 0 has [0, 2]^2 (c=(1,1), o=(1,1)) inside [-1, 3]^2 (c=(1,1), o=(2,2))
    cen_nested = torch.tensor([[[1.0, 1.0], [1.0, 1.0]]])
    off_nested = torch.tensor([[[1.0, 1.0], [2.0, 2.0]]])
    pen_nested = diversity_penalty(cen_nested, off_nested)
    assert pen_nested.shape == ()
    assert float(pen_nested.item()) > 0.0


def test_diversity_penalty_zero_at_k1() -> None:
    cen = torch.randn((5, 1, 4))
    off = torch.ones((5, 1, 4))
    pen = diversity_penalty(cen, off)
    assert pen.shape == ()
    assert float(pen.item()) == 0.0
