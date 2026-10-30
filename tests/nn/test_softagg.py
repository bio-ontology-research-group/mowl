"""Unit tests for soft component aggregation and diversity regularisation (Objective O1)."""
from __future__ import annotations

import pytest
import torch

from mowl.nn import MultiBoxELModule, MultiELBEModule
from mowl.nn.el.softagg import diversity_penalty, soft_gci1_terms, soft_weights, softmin


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


def test_softmin_tau_floor_underflow() -> None:
    """At tau=0.01 with gaps [0, 1, 2], float32 non-nearest gradients underflow to 0; at tau=0.05 they are > 0."""
    v_small_tau = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32, requires_grad=True)
    softmin(v_small_tau, tau=0.01, dim=0).backward()
    assert v_small_tau.grad is not None
    assert float(v_small_tau.grad[0].item()) == pytest.approx(1.0, abs=1e-6)
    assert float(v_small_tau.grad[1].item()) == 0.0
    assert float(v_small_tau.grad[2].item()) == 0.0

    v_floor_tau = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32, requires_grad=True)
    softmin(v_floor_tau, tau=0.05, dim=0).backward()
    assert v_floor_tau.grad is not None
    assert float(v_floor_tau.grad[0].item()) > 0.99
    assert float(v_floor_tau.grad[1].item()) > 0.0
    assert float(v_floor_tau.grad[2].item()) > 0.0


def test_soft_gci1_per_pair_gradient_flow_and_detached_gate() -> None:
    """Verify soft_gci1_terms detaches gate g_ij while keeping per-pair gradients in empt_term and incl_ijm."""
    empt = torch.tensor([[[0.5, 1.0], [1.5, 2.0]]], dtype=torch.float32, requires_grad=True)
    non_empty = torch.zeros_like(empt)
    incl_ijm = torch.tensor(
        [[[[0.2, 0.8], [0.4, 0.9]], [[0.3, 0.7], [0.5, 1.1]]]],
        dtype=torch.float32,
        requires_grad=True,
    )
    empt_term, gated_incl = soft_gci1_terms(empt, non_empty, incl_ijm, tau=0.5)
    assert empt_term.shape == (1,)
    assert gated_incl.shape == (1,)

    # 1) Gate g_ij is detached: backward through gated_incl alone gives no grad to empt
    gated_incl.sum().backward(retain_graph=True)
    assert empt.grad is None or torch.all(empt.grad == 0.0)
    assert incl_ijm.grad is not None
    assert torch.all(incl_ijm.grad > 0.0)

    # 2) Emptiness term keeps gradient through w_ij * empt_ij to all k^2 = 4 pairs
    incl_ijm.grad.zero_()
    (empt_term + gated_incl).sum().backward()
    assert empt.grad is not None
    assert torch.all(empt.grad.abs() > 0.0)
    w_expected = soft_weights(empt.detach().view(1, -1), tau=0.5, dim=1).view_as(empt)
    expected_empt_grad = w_expected * (1.0 - (empt.detach() - empt_term.detach()) / 0.5)
    assert torch.allclose(empt.grad, expected_empt_grad, atol=1e-6)
    assert torch.all(incl_ijm.grad > 0.0)

    # 3) Module-level k=2 soft gci1 per-pair gradient flow in both MultiBoxELModule and MultiELBEModule
    d, k = 3, 2
    batch = torch.tensor([[0, 1, 2]], dtype=torch.long)
    for cls, emb_attr in (
        (MultiBoxELModule, "class_center"),
        (MultiELBEModule, "class_embed"),
    ):
        m_hard = cls(
            nb_ont_classes=3,
            nb_rels=1,
            embed_dim=d,
            n_components=k,
            component_aggregation="hard",
        )
        m_soft = cls(
            nb_ont_classes=3,
            nb_rels=1,
            embed_dim=d,
            n_components=k,
            component_aggregation="soft",
            tau=0.5,
        )
        with torch.no_grad():
            for m in (m_hard, m_soft):
                cen_emb = getattr(m, emb_attr)
                # C (row 0): comp 0 at 0.0, comp 1 at 4.0
                cen_c = cen_emb.weight[0].view(k, d)
                cen_c[0].fill_(0.0)
                cen_c[1].fill_(4.0)
                # D (row 1): comp 0 at 1.0, comp 1 at 6.0 (all 4 pairs disjoint when offset=0.2)
                cen_d = cen_emb.weight[1].view(k, d)
                cen_d[0].fill_(1.0)
                cen_d[1].fill_(6.0)
                # E (row 2): comp 0 at 0.5, comp 1 at 5.0
                cen_e = cen_emb.weight[2].view(k, d)
                cen_e[0].fill_(0.5)
                cen_e[1].fill_(5.0)
                m.class_offset.weight.fill_(0.2)

        m_hard(batch, "gci1").sum().backward()
        grad_c_hard = getattr(m_hard, emb_attr).weight.grad[0].view(k, d)
        grad_d_hard = getattr(m_hard, emb_attr).weight.grad[1].view(k, d)
        grad_e_hard = getattr(m_hard, emb_attr).weight.grad[2].view(k, d)
        assert (torch.linalg.norm(grad_c_hard, dim=-1) == 0.0).any()
        assert (torch.linalg.norm(grad_d_hard, dim=-1) == 0.0).any()
        assert (torch.linalg.norm(grad_e_hard, dim=-1) == 0.0).any()

        m_soft(batch, "gci1").sum().backward()
        grad_c_soft = getattr(m_soft, emb_attr).weight.grad[0].view(k, d)
        grad_d_soft = getattr(m_soft, emb_attr).weight.grad[1].view(k, d)
        grad_e_soft = getattr(m_soft, emb_attr).weight.grad[2].view(k, d)
        assert (torch.linalg.norm(grad_c_soft, dim=-1) > 0.0).all()
        assert (torch.linalg.norm(grad_d_soft, dim=-1) > 0.0).all()
        assert (torch.linalg.norm(grad_e_soft, dim=-1) > 0.0).all()


def test_negative_gci2_mean_over_all_pairs_hard_and_soft() -> None:
    """At k >= 2, negative gci2 is the mean over all k^2 pairs in both hard and soft modes."""
    n_classes, n_rels, d, k = 4, 2, 3, 2
    batch = torch.tensor([[0, 0, 1], [2, 1, 3]], dtype=torch.long)

    for cls in (MultiBoxELModule, MultiELBEModule):
        torch.manual_seed(99)
        m_hard = cls(
            nb_ont_classes=n_classes,
            nb_rels=n_rels,
            embed_dim=d,
            n_components=k,
            component_aggregation="hard",
        )
        torch.manual_seed(99)
        m_soft = cls(
            nb_ont_classes=n_classes,
            nb_rels=n_rels,
            embed_dim=d,
            n_components=k,
            component_aggregation="soft",
            tau=0.3,
        )
        neg_hard = m_hard(batch, "gci2", neg=True)
        neg_soft = m_soft(batch, "gci2", neg=True)
        assert torch.equal(neg_hard, neg_soft)

        # Verify against explicit mean over the k^2 = 4 single-component evaluations
        pair_vals = []
        for i in range(k):
            for j in range(k):
                m_single = cls(
                    nb_ont_classes=n_classes,
                    nb_rels=n_rels,
                    embed_dim=d,
                    n_components=1,
                )
                with torch.no_grad():
                    if cls is MultiBoxELModule:
                        m_single.class_center.weight.copy_(
                            m_hard.class_center.weight.view(n_classes, k, d)[:, i, :]
                        )
                        m_single.class_offset.weight.copy_(
                            m_hard.class_offset.weight.view(n_classes, k, d)[:, i, :]
                        )
                        # Put target class component j into rows 1 and 3
                        m_single.class_center.weight[1].copy_(
                            m_hard.class_center.weight.view(n_classes, k, d)[1, j, :]
                        )
                        m_single.class_offset.weight[1].copy_(
                            m_hard.class_offset.weight.view(n_classes, k, d)[1, j, :]
                        )
                        m_single.class_center.weight[3].copy_(
                            m_hard.class_center.weight.view(n_classes, k, d)[3, j, :]
                        )
                        m_single.class_offset.weight[3].copy_(
                            m_hard.class_offset.weight.view(n_classes, k, d)[3, j, :]
                        )
                        m_single.bump_classes.weight.copy_(m_hard.bump_classes.weight)
                        m_single.head_center.weight.copy_(m_hard.head_center.weight)
                        m_single.head_offset.weight.copy_(m_hard.head_offset.weight)
                        m_single.tail_center.weight.copy_(m_hard.tail_center.weight)
                        m_single.tail_offset.weight.copy_(m_hard.tail_offset.weight)
                    else:
                        m_single.class_embed.weight.copy_(
                            m_hard.class_embed.weight.view(n_classes, k, d)[:, i, :]
                        )
                        m_single.class_offset.weight.copy_(
                            m_hard.class_offset.weight.view(n_classes, k, d)[:, i, :]
                        )
                        m_single.class_embed.weight[1].copy_(
                            m_hard.class_embed.weight.view(n_classes, k, d)[1, j, :]
                        )
                        m_single.class_offset.weight[1].copy_(
                            m_hard.class_offset.weight.view(n_classes, k, d)[1, j, :]
                        )
                        m_single.class_embed.weight[3].copy_(
                            m_hard.class_embed.weight.view(n_classes, k, d)[3, j, :]
                        )
                        m_single.class_offset.weight[3].copy_(
                            m_hard.class_offset.weight.view(n_classes, k, d)[3, j, :]
                        )
                        m_single.rel_embed.weight.copy_(m_hard.rel_embed.weight)
                pair_vals.append(m_single(batch, "gci2", neg=True))
        expected_mean = torch.stack(pair_vals, dim=0).mean(dim=0)
        assert torch.allclose(neg_hard, expected_mean, atol=1e-6, rtol=1e-5)


def test_diversity_penalty_zero_when_disjoint_positive_when_nested() -> None:
    # Disjoint components at d = 2: class 0 has [0, 2]^2 and [10, 12]^2; class 1 has [-5, -3]^2 and [5, 7]^2
    cen_disjoint = torch.tensor(
        [
            [[1.0, 1.0], [11.0, 11.0]],
            [[-4.0, -4.0], [6.0, 6.0]],
        ]
    )
    off_disjoint = torch.ones((2, 2, 2))
    pen_disjoint = diversity_penalty(cen_disjoint, off_disjoint)
    assert pen_disjoint.shape == ()
    assert float(pen_disjoint.item()) <= 1e-3

    # Nested components at d = 2: class 0 has [0, 2]^2 (c=(1,1), o=(1,1)) inside [-1, 3]^2 (c=(1,1), o=(2,2))
    cen_nested = torch.tensor([[[1.0, 1.0], [1.0, 1.0]]])
    off_nested = torch.tensor([[[1.0, 1.0], [2.0, 2.0]]])
    pen_nested = diversity_penalty(cen_nested, off_nested)
    assert pen_nested.shape == ()
    assert float(pen_nested.item()) == pytest.approx(1.0, abs=1e-6)


def test_diversity_penalty_d50() -> None:
    """Verify log-space diversity_penalty at d = 50: identical -> 1 +- 1e-6 with nonzero grad, nested -> vol ratio, disjoint -> <= 1e-3."""
    d = 50

    # 1) Identical boxes at d = 50 (including small offsets o = 0.01 where raw prod_d would underflow)
    for half_width in (0.01, 0.1, 1.0):
        cen_id = torch.zeros((3, 2, d), dtype=torch.float32, requires_grad=True)
        off_id = torch.full((3, 2, d), half_width, dtype=torch.float32)
        pen_id = diversity_penalty(cen_id, off_id)
        assert pen_id.shape == ()
        assert abs(float(pen_id.item()) - 1.0) <= 1e-6
        pen_id.backward()
        assert cen_id.grad is not None
        assert torch.isfinite(cen_id.grad).all()
        assert float(torch.linalg.norm(cen_id.grad).item()) > 0.0

    # 2) Nested small-in-large at d = 50 -> volume ratio of intersection to smaller box
    # (a) Concentric small box (side=1.0) inside large box (side=4.0) -> ratio = 1.0
    cen_nested_full = torch.zeros((1, 2, d), dtype=torch.float32)
    off_nested_full = torch.ones((1, 2, d), dtype=torch.float32)
    off_nested_full[:, 0, :] = 0.5
    off_nested_full[:, 1, :] = 2.0
    pen_nested_full = diversity_penalty(cen_nested_full, off_nested_full)
    assert abs(float(pen_nested_full.item()) - 1.0) <= 1e-6

    # (b) Small box (o=1, side=2) and large box (o=2, side=4): concentric in 48 coords (inter=2, min_side=2)
    # and shifted by +2 in the first 2 coords so [0, 2] cap [1, 5] = [1, 2] (inter=1, min_side=2) -> ratio = (1/2)^2 = 0.25
    cen_partial = torch.zeros((1, 2, d), dtype=torch.float32)
    cen_partial[0, 0, :] = 1.0
    cen_partial[0, 1, :] = 1.0
    cen_partial[0, 1, :2] = 3.0
    off_partial = torch.ones((1, 2, d), dtype=torch.float32)
    off_partial[0, 0, :] = 1.0
    off_partial[0, 1, :] = 2.0
    pen_partial = diversity_penalty(cen_partial, off_partial)
    assert float(pen_partial.item()) == pytest.approx(0.25, abs=1e-5)

    # 3) Disjoint boxes at d = 50 (disjoint in a single coordinate or in all coordinates) -> <= 1e-3
    cen_disjoint_1d = torch.zeros((2, 2, d), dtype=torch.float32)
    cen_disjoint_1d[:, 1, 0] = 10.0
    off_disjoint_1d = torch.ones((2, 2, d), dtype=torch.float32)
    pen_disjoint_1d = diversity_penalty(cen_disjoint_1d, off_disjoint_1d)
    assert float(pen_disjoint_1d.item()) <= 1e-3

    cen_disjoint_all = torch.zeros((2, 2, d), dtype=torch.float32)
    cen_disjoint_all[:, 1, :] = 10.0
    pen_disjoint_all = diversity_penalty(cen_disjoint_all, off_disjoint_1d)
    assert float(pen_disjoint_all.item()) <= 1e-3


def test_diversity_penalty_zero_at_k1() -> None:
    cen = torch.randn((5, 1, 4))
    off = torch.ones((5, 1, 4))
    pen = diversity_penalty(cen, off)
    assert pen.shape == ()
    assert float(pen.item()) == 0.0
