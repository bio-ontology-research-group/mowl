"""Learnable axis-aligned cone module for :math:`\\mathcal{ALC}` (adaptation).

See :mod:`mowl.baselines.nn.alc.alcone.losses` for the parametrisation and the exactness lemma,
and :mod:`mowl.baselines.geometric.core.cones` for the exact semantics of Özçep, Leemhuis & Wolter
(JAIR 2023).
"""
import numpy as np
import torch as th

from mowl.nn.alc.module import ALCModule
from mowl.baselines.nn.alc.alcone import losses as L
from mowl.owlapi import (ClassExpressionType, OWLClassAssertionAxiom,
                         OWLDisjointClassesAxiom, OWLEquivalentClassesAxiom,
                         OWLObjectPropertyAssertionAxiom, OWLSubClassOfAxiom)

#: Logit magnitude that earlier versions used for owl:Thing (all half-axes) and, negated,
#: for owl:Nothing. It makes every loss through the constants about 1e4 times larger than a
#: loss between learned classes; select it with ``constant_logit=CONSTANT_LOGIT``.
CONSTANT_LOGIT = 1e4

#: Default magnitude of the constant logits: the scale of learned logits (``init_std``).
DEFAULT_CONSTANT_LOGIT = 1.0


class ALConeModule(ALCModule):
    """Al-cone module: concepts, roles and individuals as half-axis logits.

    Every concept name has ``2 * dim`` logits (one per half-axis of
    :math:`\\mathbb{R}^{dim}`), every role a ``2 dim x 2 dim`` matrix of incidence logits
    (entry ``[j, i]``: half-axis ``i`` has successor ``j``), every individual ``2 * dim``
    support logits. Complex concepts are evaluated with Gödel operators
    (:func:`~mowl.baselines.nn.alc.alcone.losses.godel_and` and friends). Axiom losses are
    returned per row; with ``margin > 0`` a zero loss implies that the discretised
    interpretation (:meth:`discretise`) satisfies the axiom.

    The interface follows :class:`~mowl.nn.alc.module.ALCModule` and the axiom batches
    of :class:`~mowl.datasets.alc.ALCDataset`. The entity context ``e_emb`` used by
    FALCON is accepted and ignored: the half-axes themselves play the role of the domain.

    :param nclasses: Number of classes (including ``owl:Thing`` and ``owl:Nothing``).
    :param nindividuals: Number of individuals.
    :param nrelations: Number of object properties.
    :param dim: Dimension :math:`n` of the cone space (:math:`2n` half-axes).
    :param margin: Margin :math:`\\gamma > 0` of the losses.
    :param top_index: Class index of ``owl:Thing`` (interpreted as the whole space).
    :param bottom_index: Class index of ``owl:Nothing`` (interpreted as :math:`\\{0\\}`).
    :param init_std: Standard deviation of the normal initialisation of the logits.
    :param constant_logit: Magnitude of the logits of ``owl:Thing`` (``+``) and
        ``owl:Nothing`` (``-``). It only has to be at least ``margin`` for the exactness of
        the losses; a large value multiplies every loss that involves the constants.
        Defaults to ``1.0`` (the scale of learned logits). The value of earlier versions,
        ``1e4`` (:data:`CONSTANT_LOGIT`), reproduces their numbers.
    """

    def __init__(self, nclasses, nindividuals, nrelations, dim=8, margin=0.1,
                 top_index=None, bottom_index=None, init_std=0.5,
                 constant_logit=DEFAULT_CONSTANT_LOGIT):
        super().__init__()
        if constant_logit < margin:
            raise ValueError("constant_logit must be at least the margin.")
        self.constant_logit = constant_logit
        if dim < 1:
            raise ValueError("dim must be positive.")
        if margin <= 0:
            raise ValueError("margin must be positive (zero loss implies a model only "
                             "for margin > 0).")
        self.dim = dim
        self.nhalf = 2 * dim
        self.margin = margin
        self.top_index = top_index
        self.bottom_index = bottom_index
        #: Temperature of the smooth relaxation used during training; ``0`` gives the
        #: exact Gödel operators (for which the zero-loss guarantee holds).
        self.temperature = 0.0
        self.c_embedding = th.nn.Embedding(nclasses, self.nhalf)
        self.r_embedding = th.nn.Embedding(max(nrelations, 1), self.nhalf * self.nhalf)
        self.e_embedding = th.nn.Embedding(max(nindividuals, 1), self.nhalf)
        for emb in (self.c_embedding, self.r_embedding, self.e_embedding):
            th.nn.init.normal_(emb.weight, std=init_std)

    # -- ALCModule operator interface ---------------------------------------
    def _logical_and(self, x, y):
        return L.godel_and(x, y, self.temperature)

    def _logical_or(self, x, y):
        return L.godel_or(x, y, self.temperature)

    def _logical_not(self, x):
        return L.godel_not(x)

    def _logical_exist(self, r_fs, c_fs):
        return L.exists(r_fs, c_fs, self.temperature)

    def _logical_forall(self, r_fs, c_fs):
        return L.forall(r_fs, c_fs, self.temperature)

    def _logical_residuum(self, r_fs, c_fs):
        return L.godel_or(L.godel_not(r_fs), c_fs, self.temperature)

    # -- parameters ---------------------------------------------------------
    def class_logits(self, idx):
        """Half-axis logits of class indices ``idx`` (any shape); ``owl:Thing`` and
        ``owl:Nothing`` get the constant logits :math:`\\pm` ``constant_logit``."""
        out = self.c_embedding(idx)
        if self.top_index is not None:
            out = th.where((idx == self.top_index).unsqueeze(-1),
                           th.full_like(out, self.constant_logit), out)
        if self.bottom_index is not None:
            out = th.where((idx == self.bottom_index).unsqueeze(-1),
                           th.full_like(out, -self.constant_logit), out)
        return out

    def role_logits(self, idx):
        """Incidence logits of role indices ``idx``, shape ``idx.shape + (2n, 2n)``."""
        return self.r_embedding(idx).view(*idx.shape, self.nhalf, self.nhalf)

    def individual_logits(self, idx):
        """Support logits of individual indices ``idx``."""
        return self.e_embedding(idx)

    # -- evaluation ---------------------------------------------------------
    def forward_fs(self, cexpr, x, e_emb=None, cur_index=0):
        """Half-axis logits of the class expression pattern ``cexpr`` filled with the
        columns of ``x`` starting at ``cur_index``.

        :returns: ``(logits of shape (B, 2n), next column index)``.
        """
        t = cexpr.getClassExpressionType()
        if t == ClassExpressionType.OWL_CLASS:
            return self.class_logits(x[:, cur_index]), cur_index + 1
        if t in (ClassExpressionType.OBJECT_SOME_VALUES_FROM,
                 ClassExpressionType.OBJECT_ALL_VALUES_FROM):
            r = self.role_logits(x[:, cur_index])
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
            return L.godel_not(val), nxt
        raise NotImplementedError(f"Class expression type {t} is not in ALC.")

    def forward(self, axiom, x, e_emb=None):
        """Per-row loss of a batch of axioms of pattern ``axiom`` (shape ``(B,)``)."""
        m, tau = self.margin, self.temperature
        if isinstance(axiom, OWLSubClassOfAxiom):
            c, nxt = self.forward_fs(axiom.getSubClass(), x, e_emb)
            d, _ = self.forward_fs(axiom.getSuperClass(), x, e_emb, nxt)
            return L.inclusion_loss(c, d, m, tau)
        if isinstance(axiom, OWLEquivalentClassesAxiom):
            ces = axiom.getClassExpressionsAsList()
            c, nxt = self.forward_fs(ces[0], x, e_emb)
            d, _ = self.forward_fs(ces[1], x, e_emb, nxt)
            return L.inclusion_loss(c, d, m, tau) + L.inclusion_loss(d, c, m, tau)
        if isinstance(axiom, OWLDisjointClassesAxiom):
            ces = axiom.getClassExpressionsAsList()
            c, nxt = self.forward_fs(ces[0], x, e_emb)
            d, _ = self.forward_fs(ces[1], x, e_emb, nxt)
            return L.disjointness_loss(c, d, m, tau)
        if isinstance(axiom, OWLClassAssertionAxiom):
            u = self.individual_logits(x[:, 0])
            c, _ = self.forward_fs(axiom.getClassExpression(), x, e_emb, 1)
            return L.membership_loss(u, c, m, tau) + L.point_validity_loss(u, m, tau)
        if isinstance(axiom, OWLObjectPropertyAssertionAxiom):
            # ALCDataset encodes R(a, b) as [b, R, a].
            u_obj = self.individual_logits(x[:, 0])
            r = self.role_logits(x[:, 1])
            u_subj = self.individual_logits(x[:, 2])
            role = L.role_assertion_loss(u_subj, r, u_obj, m, tau)
            valid = L.point_validity_loss(u_subj, m, tau) + L.point_validity_loss(u_obj, m, tau)
            return role + valid
        raise NotImplementedError(f"Axiom type not supported: {axiom}")

    def nondegeneracy_loss(self, class_idx):
        """Mean of :func:`~mowl.baselines.nn.alc.alcone.losses.nondegeneracy_loss` over the given
        class indices."""
        return L.nondegeneracy_loss(self.c_embedding(class_idx), self.margin,
                                    self.temperature).mean()

    # -- discretisation -----------------------------------------------------
    def discretise(self, class_names=None, role_names=None, individual_names=None):
        """Return the discretised :class:`~mowl.baselines.geometric.core.cones.ALConeInterpretation`:
        half-axis ``h`` belongs to a concept (role pair, individual support) iff its logit
        is ``> 0``. An individual whose support contains both half-axes of a coordinate
        (possible only while :func:`~mowl.baselines.nn.alc.alcone.losses.point_validity_loss` is
        positive) keeps the half-axis with the larger logit.

        :param class_names: Sequence mapping class index to name (default: index as str).
        :param role_names: Sequence mapping role index to name.
        :param individual_names: Sequence mapping individual index to name.
        """
        from mowl.baselines.geometric.core.cones import ALCone, ALConeInterpretation

        with th.no_grad():
            cw = self.c_embedding.weight.detach().cpu().numpy() > 0
            rw = self.r_embedding.weight.detach().cpu().numpy().reshape(
                -1, self.nhalf, self.nhalf) > 0
            ew = self.e_embedding.weight.detach().cpu().numpy()
        cnames = list(class_names) if class_names is not None \
            else [str(i) for i in range(cw.shape[0])]
        rnames = list(role_names) if role_names is not None \
            else [str(i) for i in range(rw.shape[0])]
        inames = list(individual_names) if individual_names is not None \
            else [str(i) for i in range(ew.shape[0])]
        concepts = {}
        for i, name in enumerate(cnames):
            if i in (self.top_index, self.bottom_index):
                continue
            concepts[name] = ALCone(cw[i])
        roles = {name: rw[i] for i, name in enumerate(rnames)}
        inds = {}
        for k, name in enumerate(inames):
            u = ew[k].reshape(-1, 2)
            pt = np.zeros(self.dim)
            for i, (p, q) in enumerate(u):
                if p > 0 and (q <= 0 or p >= q):
                    pt[i] = 1.0
                elif q > 0:
                    pt[i] = -1.0
            inds[name] = pt
        return ALConeInterpretation(self.dim, concepts, roles, inds)
