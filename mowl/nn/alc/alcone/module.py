"""Learnable axis-aligned cone module for :math:`\\mathcal{ALC}` (adaptation), put together
from parts.

The al-cone papers (Özçep, Leemhuis & Wolter, IJCAI 2020; JAIR 78, 2023) give the geometry and
model constructions but no learning objective for ontology embedding (JAIR Sec. 7); the
relaxation here is **ours**. An al-cone of :math:`\\mathbb{R}^n` is a subset of the :math:`2n`
half-axes (JAIR Prop. 1; see :mod:`mowl.geometric.cones`). Every concept name gets a real
*logit* per half-axis; the discretised cone contains half-axis :math:`h` iff its logit is
:math:`> 0`. A role gets a logit per entry of its :math:`2n \\times 2n` incidence matrix (JAIR
Def. 7), an individual a logit per half-axis of its *support*.

Complex concepts are evaluated with Gödel operators on logits (the family
:class:`~mowl.representation.library.logic_fuzzy.LogitGoedel`): :math:`\\sqcap` is :math:`\\min`,
:math:`\\sqcup` is :math:`\\max`, :math:`\\neg` is :math:`-x`, :math:`(\\exists R.C)_i =
\\max_j \\min(R_{ji}, C_j)` and :math:`\\forall R.C = \\neg\\exists R.\\neg C`. If the value of a
concept at a half-axis is :math:`> 0` the discretised cone contains the half-axis, and if it is
:math:`< 0` it does not; consequently, with margin :math:`\\gamma > 0`, every axiom loss
(:class:`~mowl.representation.library.logic_fuzzy.ConeMeasure`) is zero only if the discretised
interpretation satisfies the axiom, and conversely, if the discretised interpretation
satisfies the axiom and every logit has absolute value at least :math:`\\gamma`, the loss is
zero.

The hand-written loss module that this one replaces is kept for comparison as
:mod:`mowl.baselines.nn.alc.alcone`.
"""
import numpy as np
import torch as th

from mowl.representation.library.logic_alc import ExpressionGeometry
from mowl.representation.library.logic_fuzzy import alcone_parts

__all__ = ["ALConeModule", "CONSTANT_LOGIT", "DEFAULT_CONSTANT_LOGIT"]

#: Logit magnitude that earlier versions used for owl:Thing (all half-axes) and, negated,
#: for owl:Nothing. It makes every loss through the constants about 1e4 times larger than a
#: loss between learned classes; select it with ``constant_logit=CONSTANT_LOGIT``.
CONSTANT_LOGIT = 1e4

#: Default magnitude of the constant logits: the scale of learned logits (``init_std``).
DEFAULT_CONSTANT_LOGIT = 1.0


class ALConeModule(ExpressionGeometry):
    """Al-cone module: concepts, roles and individuals as half-axis logits.

    Every concept name has ``2 * dim`` logits (one per half-axis of
    :math:`\\mathbb{R}^{dim}`), every role a ``2 dim x 2 dim`` matrix of incidence logits
    (entry ``[j, i]``: half-axis ``i`` has successor ``j``), every individual ``2 * dim``
    support logits. Axiom losses are returned per row; with ``margin > 0`` a zero loss implies
    that the discretised interpretation (:meth:`discretise`) satisfies the axiom.

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
        Defaults to ``1.0``; the value of earlier versions, ``1e4`` (:data:`CONSTANT_LOGIT`),
        reproduces their numbers.
    :param nondegeneracy: Weight of the penalty against empty classes
        (:class:`~mowl.representation.library.logic_fuzzy.NonDegeneracy`); ``0``: none.
    :param anneal: Initial temperature of the annealed smoothing of the training schedule
        (:class:`~mowl.representation.library.logic_alc.AnnealedRelaxation`); ``None``: none.
    :param noise: Scale of the parameter noise of that schedule.
    """

    def __init__(self, nclasses, nindividuals, nrelations, dim=8, margin=0.1,
                 top_index=None, bottom_index=None, init_std=0.5,
                 constant_logit=DEFAULT_CONSTANT_LOGIT, nondegeneracy=0.0, anneal=None,
                 noise=0.3):
        if constant_logit < margin:
            raise ValueError("constant_logit must be at least the margin.")
        if dim < 1:
            raise ValueError("dim must be positive.")
        if margin <= 0:
            raise ValueError("margin must be positive (zero loss implies a model only "
                             "for margin > 0).")
        super().__init__(**alcone_parts(
            nclasses, nrelations, nindividuals, dim, margin, 0.0, None, top_index,
            bottom_index, constant_logit, init_std, anneal, noise, nondegeneracy))
        self.constant_logit = constant_logit
        self.dim = dim
        self.nhalf = 2 * dim
        self.top_index = top_index
        self.bottom_index = bottom_index

    # -- parameters ---------------------------------------------------------
    def class_logits(self, idx):
        """Half-axis logits of class indices ``idx`` (any shape); ``owl:Thing`` and
        ``owl:Nothing`` get the constant logits :math:`\\pm` ``constant_logit``."""
        return self.class_model(idx).values

    def role_logits(self, idx):
        """Incidence logits of role indices ``idx``, shape ``idx.shape + (2n, 2n)``."""
        return self.r_embedding(idx).view(*idx.shape, self.nhalf, self.nhalf)

    def individual_logits(self, idx):
        """Support logits of individual indices ``idx``."""
        return self.e_embedding(idx)

    def nondegeneracy_loss(self, class_idx):
        """Mean over the given class indices of :math:`\\mathrm{relu}(\\gamma - \\max_h c_h)`
        on the (raw) half-axis logits."""
        from mowl.representation.library.logic_fuzzy import CL
        return th.relu(self.margin - CL.reduce_max(self.c_embedding(class_idx), -1,
                                                   self.temperature)).mean()

    # -- discretisation -----------------------------------------------------
    def discretise(self, class_names=None, role_names=None, individual_names=None):
        """Return the discretised :class:`~mowl.geometric.cones.ALConeInterpretation`:
        half-axis ``h`` belongs to a concept (role pair, individual support) iff its logit
        is ``> 0``. An individual whose support contains both half-axes of a coordinate
        (possible only while the well-formedness cost of the individuals is positive) keeps
        the half-axis with the larger logit.

        :param class_names: Sequence mapping class index to name (default: index as str).
        :param role_names: Sequence mapping role index to name.
        :param individual_names: Sequence mapping individual index to name.
        """
        from mowl.geometric.cones import ALCone, ALConeInterpretation

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
