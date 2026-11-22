"""Losses and module of OnT (Yang, Chen, He, Gao, Horrocks, "Language Models as Ontology
Encoders", ISWC 2025, Section 4).

OnT extends HiT (:mod:`mowl.nn.lm.hit`) from taxonomies to :math:`\\mathcal{EL}`:

* every concept, atomic or complex, is a point :math:`x_C` of the Poincaré ball obtained by
  encoding its verbalisation :math:`V(C)` (Section 4.1; see :mod:`mowl.text.verbalise`);
* every role :math:`r` is a function :math:`f_r(v) = k_r \\otimes (R(\\Theta_r) v)`, a
  block rotation followed by a Möbius scaling (Eq. 4), whose parameters are a linear
  function of the encoding of :math:`V(r)` (Section 5.1, "we apply an extra linear
  transformation on the embedding of r");
* a subsumption :math:`C \\sqsubseteq D` is the hierarchy loss :math:`L_\\prec(x_C \\prec
  x_D)` of HiT (Eq. 5), with one sampled negative :math:`D_{neg}`;
* every :math:`\\exists r.D` occurring in the ontology adds
  :math:`L_r = \\frac{1}{2}(L_\\prec(x_{\\exists r.D} \\prec f_r(x_D)) +
  L_\\prec(f_r(x_D) \\prec x_{\\exists r.D}))` (Eq. 6);
* every :math:`C \\sqcap D` occurring in the ontology adds
  :math:`L_\\sqcap = \\frac{1}{2}(L_\\prec(x_{C \\sqcap D} \\prec x_C) + L_\\prec(x_{C \\sqcap D}
  \\prec x_D))` (Eq. 7).

Scaling :math:`k_r`: Eq. 4 includes it, but the reference implementation computes
:math:`k_r = \\exp(\\cdot)` and does not apply it ("not used in final output"), and the
published models treat it as 1. :class:`RoleTransformation` follows the code by default
(``use_scaling=False``); with ``use_scaling=True`` it applies Eq. 4 literally.
"""

import torch as th

from mowl.baselines.nn.lm.hit import HiTModule, hierarchy_loss
from mowl.representation.library.hierarchy_relations import RoleTransformation


def existential_loss(x_exists, x_role_image, negative_1, negative_2, c, clustering_margin,
                     centripetal_margin, clustering_weight=1.0, centripetal_weight=1.0):
    """:math:`L_r(\\exists r.D)` of OnT Eq. 6, one value per row.

    :param x_exists: Embeddings of the verbalisations of :math:`\\exists r.D`.
    :param x_role_image: :math:`f_r(x_D)`.
    :param negative_1: Negatives for :math:`x_{\\exists r.D} \\prec f_r(x_D)`.
    :param negative_2: Negatives for :math:`f_r(x_D) \\prec x_{\\exists r.D}`.
    """
    args = (c, clustering_margin, centripetal_margin, clustering_weight, centripetal_weight)
    forward = hierarchy_loss(x_exists, x_role_image, negative_1, *args)
    backward = hierarchy_loss(x_role_image, x_exists, negative_2, *args)
    return 0.5 * (forward + backward)


def conjunction_loss(x_conjunction, conjuncts, negatives, c, clustering_margin,
                     centripetal_margin, clustering_weight=1.0, centripetal_weight=1.0):
    """:math:`L_\\sqcap(C_1 \\sqcap \\dots \\sqcap C_n)`, one value per row: the mean over
    :math:`i` of :math:`L_\\prec(x_{C_1 \\sqcap \\dots \\sqcap C_n} \\prec x_{C_i})`. For
    :math:`n = 2` this is OnT Eq. 7.

    :param x_conjunction: Embeddings of the verbalised conjunctions, ``(n, d)``.
    :param conjuncts: List of tensors ``(n, d)``, one per conjunct position.
    :param negatives: List of negative tensors, one per conjunct position.
    """
    args = (c, clustering_margin, centripetal_margin, clustering_weight, centripetal_weight)
    losses = [hierarchy_loss(x_conjunction, x_i, neg, *args)
              for x_i, neg in zip(conjuncts, negatives)]
    return th.stack(losses).mean(dim=0)


class OnTModule(HiTModule):
    """HiT module with OnT's role transformations and logical losses.

    :param encoder: An encoder module (see :mod:`mowl.nn.lm.encoders`).
    :param use_roles: ``False`` gives the ablation OnT(w/o r) (Section 5.1): only the
        hierarchy loss of Eq. 5 is used.
    :param conjunction_weight: Weight of :math:`L_\\sqcap` (1 in the paper).
    :param existential_weight: Weight of :math:`L_r` (1 in the paper).
    :param role_pairing: See :class:`RoleTransformation`.
    :param role_scaling: See :class:`RoleTransformation`.

    The remaining parameters are those of :class:`~mowl.nn.lm.hit.HiTModule`; OnT's
    defaults (Section 5.1) are :math:`\\alpha = 3.0`, :math:`\\beta = 0.5`.
    """

    def __init__(self, encoder, curvature=None, clustering_margin=3.0, centripetal_margin=0.5,
                 clustering_weight=1.0, centripetal_weight=1.0, use_roles=True,
                 conjunction_weight=1.0, existential_weight=1.0, role_pairing="half",
                 role_scaling=False):
        super().__init__(encoder, curvature, clustering_margin, centripetal_margin,
                         clustering_weight, centripetal_weight)
        self.use_roles = use_roles
        self.conjunction_weight = conjunction_weight
        self.existential_weight = existential_weight
        self.role_model = RoleTransformation(self.dim, role_pairing, role_scaling)

    def _args(self):
        return (self.c, self.clustering_margin, self.centripetal_margin,
                self.clustering_weight, self.centripetal_weight)

    def role_image(self, filler, role_texts):
        """:math:`f_r(x_D)` for filler embeddings ``(n, d)`` and role verbalisations."""
        return self.role_model(filler, self.embed(role_texts), self.c)

    def existential_loss(self, exists_texts, role_texts, filler_texts, negative_1, negative_2):
        """Per-row :math:`L_r` (Eq. 6) for verbalised :math:`\\exists r.D`, :math:`V(r)`
        and :math:`V(D)`; negatives are embeddings."""
        x_exists = self.embed(exists_texts)
        image = self.role_image(self.embed(filler_texts), role_texts)
        return existential_loss(x_exists, image, negative_1, negative_2, *self._args())

    def conjunction_loss(self, conjunction_texts, conjunct_texts, negatives):
        """Per-row :math:`L_\\sqcap` (Eq. 7). ``conjunct_texts`` is a list (one entry per
        conjunct position) of lists of texts."""
        x = self.embed(conjunction_texts)
        conjuncts = [self.embed(t) for t in conjunct_texts]
        return conjunction_loss(x, conjuncts, negatives, *self._args())
