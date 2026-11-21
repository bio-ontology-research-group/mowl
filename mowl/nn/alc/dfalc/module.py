"""Differentiable fuzzy :math:`\\mathcal{ALC}` (DF-ALC, Wu et al. 2022).

The interpretation itself is the set of parameters: every concept name is a vector of
truth values over a finite domain :math:`\\Delta`, every role name an
:math:`|\\Delta| \\times |\\Delta|` matrix of truth values. Truth values are
``sigmoid(logit)`` so that they stay in :math:`[0, 1]`. Complex concepts use the Gödel
semantics of fuzzy :math:`\\mathcal{ALC}` (Eqs. 1-5 of the paper).
"""
import torch as th

from mowl.nn.alc.module import ALCModule
from mowl.owlapi import (ClassExpressionType, OWLClassAssertionAxiom,
                         OWLDisjointClassesAxiom, OWLEquivalentClassesAxiom,
                         OWLObjectPropertyAssertionAxiom, OWLSubClassOfAxiom)

_QUANTIFIERS = (ClassExpressionType.OBJECT_SOME_VALUES_FROM,
                ClassExpressionType.OBJECT_ALL_VALUES_FROM)


def has_quantifier(cexpr):
    """Whether the class expression contains ``∃`` or ``∀``."""
    t = cexpr.getClassExpressionType()
    if t in _QUANTIFIERS:
        return True
    if t in (ClassExpressionType.OBJECT_INTERSECTION_OF, ClassExpressionType.OBJECT_UNION_OF):
        return any(has_quantifier(o) for o in cexpr.getOperandsAsList())
    if t == ClassExpressionType.OBJECT_COMPLEMENT_OF:
        return has_quantifier(cexpr.getOperand())
    return False


def hierarchical_loss(c, d):
    """Eq. 8 of the paper for one axiom :math:`C \\sqsubseteq D`: the mean over the domain
    of :math:`\\max(0, C^I(a) - D^I(a))`, one value per row. It is zero iff
    :math:`C^I(a) \\le D^I(a)` for all :math:`a` (Eq. 6)."""
    return th.relu(c - d).mean(dim=-1)


def rule_based_loss(c, d):
    """Eq. 9 for axioms of normal forms 1-3 (no quantifier): :math:`\\sum_s (1 - D^I(s))
    G(C^I(s), D^I(s))` with :math:`G(v, w) = \\mathrm{ReLU}(v - w)` excluded from the
    gradient, as stated in the paper (averaged over the domain here)."""
    return ((1 - d) * th.relu(c - d).detach()).mean(dim=-1)


class DFALCModule(ALCModule):
    """DF-ALC module: truth-value tables of concepts and roles over a finite domain.

    :param nclasses: Number of classes (including ``owl:Thing``, ``owl:Nothing``).
    :param nrelations: Number of object properties.
    :param ndomain: Size :math:`|\\Delta|` of the domain.
    :param top_index: Class index of ``owl:Thing`` (constant ``1``).
    :param bottom_index: Class index of ``owl:Nothing`` (constant ``0``).
    :param loss_type: ``"hierarchical"`` (Eq. 8, default) or ``"rule"``: Eq. 9 for
        inclusions without quantifiers, Eq. 8 for the others.
    :param init_std: Standard deviation of the logit initialisation.
    """

    def __init__(self, nclasses, nrelations, ndomain, top_index=None, bottom_index=None,
                 loss_type="hierarchical", init_std=1.0):
        super().__init__()
        if ndomain < 1:
            raise ValueError("The domain of a DF-ALC interpretation must not be empty.")
        if loss_type not in ("hierarchical", "rule"):
            raise ValueError("loss_type must be 'hierarchical' or 'rule'.")
        self.ndomain = ndomain
        self.top_index = top_index
        self.bottom_index = bottom_index
        self.loss_type = loss_type
        self.c_embedding = th.nn.Embedding(nclasses, ndomain)
        self.r_embedding = th.nn.Embedding(max(nrelations, 1), ndomain * ndomain)
        for emb in (self.c_embedding, self.r_embedding):
            th.nn.init.normal_(emb.weight, std=init_std)

    # -- Gödel operators -----------------------------------------------------------------
    def _logical_and(self, x, y):
        return th.minimum(x, y)

    def _logical_or(self, x, y):
        return th.maximum(x, y)

    def _logical_not(self, x):
        return 1 - x

    def _logical_residuum(self, r_fs, c_fs):
        return th.maximum(1 - r_fs, c_fs)

    def _logical_exist(self, r_fs, c_fs):
        # (∃r.C)(a) = sup_b min(r(a, b), C(b)); r_fs: (B, n, n), c_fs: (B, n)
        return th.minimum(r_fs, c_fs.unsqueeze(1)).amax(dim=-1)

    def _logical_forall(self, r_fs, c_fs):
        # (∀r.C)(a) = inf_b max(1 - r(a, b), C(b))
        return self._logical_residuum(r_fs, c_fs.unsqueeze(1)).amin(dim=-1)

    # -- groundings ----------------------------------------------------------------------
    def concept_truth(self, idx):
        """Truth values of class indices ``idx`` (shape ``idx.shape + (n,)``)."""
        out = th.sigmoid(self.c_embedding(idx))
        if self.top_index is not None:
            out = th.where((idx == self.top_index).unsqueeze(-1), th.ones_like(out), out)
        if self.bottom_index is not None:
            out = th.where((idx == self.bottom_index).unsqueeze(-1), th.zeros_like(out), out)
        return out

    def role_truth(self, idx):
        """Truth values of role indices ``idx``, shape ``idx.shape + (n, n)``."""
        n = self.ndomain
        return th.sigmoid(self.r_embedding(idx)).view(*idx.shape, n, n)

    # -- evaluation ----------------------------------------------------------------------
    def forward_fs(self, cexpr, x, e_emb=None, cur_index=0):
        """Truth values over the domain of the pattern ``cexpr`` filled with the columns of
        ``x`` from ``cur_index``: ``(tensor of shape (B, n), next column index)``."""
        t = cexpr.getClassExpressionType()
        if t == ClassExpressionType.OWL_CLASS:
            return self.concept_truth(x[:, cur_index]), cur_index + 1
        if t in _QUANTIFIERS:
            r = self.role_truth(x[:, cur_index])
            c, nxt = self.forward_fs(cexpr.getFiller(), x, e_emb, cur_index + 1)
            if t == ClassExpressionType.OBJECT_SOME_VALUES_FROM:
                return self._logical_exist(r, c), nxt
            return self._logical_forall(r, c), nxt
        if t in (ClassExpressionType.OBJECT_INTERSECTION_OF,
                 ClassExpressionType.OBJECT_UNION_OF):
            ops = cexpr.getOperandsAsList()
            out, nxt = self.forward_fs(ops[0], x, e_emb, cur_index)
            for op in ops[1:]:
                val, nxt = self.forward_fs(op, x, e_emb, nxt)
                out = self._logical_and(out, val) \
                    if t == ClassExpressionType.OBJECT_INTERSECTION_OF \
                    else self._logical_or(out, val)
            return out, nxt
        if t == ClassExpressionType.OBJECT_COMPLEMENT_OF:
            val, nxt = self.forward_fs(cexpr.getOperand(), x, e_emb, cur_index)
            return self._logical_not(val), nxt
        raise NotImplementedError(f"Class expression type {t} is not in ALC.")

    def _inclusion(self, c_expr, d_expr, x, e_emb, start=0):
        c, nxt = self.forward_fs(c_expr, x, e_emb, start)
        d, _ = self.forward_fs(d_expr, x, e_emb, nxt)
        if self.loss_type == "rule" and not (has_quantifier(c_expr) or has_quantifier(d_expr)):
            return rule_based_loss(c, d)
        return hierarchical_loss(c, d)

    def forward(self, axiom, x, e_emb=None):
        """Per-row loss of a batch of axioms of pattern ``axiom`` (shape ``(B,)``).

        Class and role assertions are fuzzy assertions with truth value 1
        (:math:`\\varphi = 1`, Syntax section); their loss is the distance to 1. Individual
        ``i`` is domain element ``i``."""
        if isinstance(axiom, OWLSubClassOfAxiom):
            return self._inclusion(axiom.getSubClass(), axiom.getSuperClass(), x, e_emb)
        if isinstance(axiom, OWLEquivalentClassesAxiom):
            ces = axiom.getClassExpressionsAsList()
            forward = self._inclusion(ces[0], ces[1], x, e_emb)
            return forward + self._inclusion_swapped(ces, x, e_emb)
        if isinstance(axiom, OWLDisjointClassesAxiom):
            ces = axiom.getClassExpressionsAsList()
            c, nxt = self.forward_fs(ces[0], x, e_emb)
            d, _ = self.forward_fs(ces[1], x, e_emb, nxt)
            return self._logical_and(c, d).mean(dim=-1)        # C ⊓ D ⊑ ⊥
        if isinstance(axiom, OWLClassAssertionAxiom):
            c, _ = self.forward_fs(axiom.getClassExpression(), x, e_emb, 1)
            rows = th.arange(len(x), device=x.device)
            return 1 - c[rows, x[:, 0]]
        if isinstance(axiom, OWLObjectPropertyAssertionAxiom):
            # ALCDataset encodes R(a, b) as [b, R, a].
            r = self.role_truth(x[:, 1])
            rows = th.arange(len(x), device=x.device)
            return 1 - r[rows, x[:, 2], x[:, 0]]
        raise NotImplementedError(f"Axiom type not supported: {axiom}")

    def _inclusion_swapped(self, ces, x, e_emb):
        """:math:`D \\sqsubseteq C` for an equivalence ``[C, D]`` read from ``x``."""
        c, nxt = self.forward_fs(ces[0], x, e_emb)
        d, _ = self.forward_fs(ces[1], x, e_emb, nxt)
        if self.loss_type == "rule" and not (has_quantifier(ces[0]) or has_quantifier(ces[1])):
            return rule_based_loss(d, c)
        return hierarchical_loss(d, c)
