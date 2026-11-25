"""BoxSEL: box embeddings of Statistical EL (Zhu et al., 2024)."""
from collections import namedtuple

import numpy as np
import torch as th

from mowl.baselines.base_models.elmodel import EmbeddingELModel
from mowl.baselines.nn.el.boxsel import losses as S
from mowl.baselines.nn.el.boxsel.module import BoxSELModule

_THING = "http://www.w3.org/2002/07/owl#Thing"

#: A statistical conditional :math:`(D \mid C)[l, u]`: ``conclusion`` is :math:`D`,
#: ``premise`` is :math:`C`. Concept expressions are a class IRI, ``("and", e1, e2, ...)``
#: or ``("some", role_iri, e)``.
Conditional = namedtuple("Conditional", ["conclusion", "premise", "lower", "upper"])


def _as_conditional(item):
    if isinstance(item, Conditional):
        c = item
    elif len(item) == 3:                      # (D, C, p)
        c = Conditional(item[0], item[1], item[2], item[2])
    else:
        c = Conditional(*item)
    if not (0.0 <= c.lower <= c.upper <= 1.0):
        raise ValueError(f"A conditional needs 0 <= l <= u <= 1, got [{c.lower}, {c.upper}].")
    return c


class BoxSEL(EmbeddingELModel):
    """BoxSEL [zhu2024]_: box embeddings of Statistical :math:`\\mathcal{EL}`.

    Concepts are boxes, roles affine maps and individuals points, exactly as in
    :class:`~mowl.models.BoxEL`, whose losses encode the deterministic axioms of the
    ontology. Probabilistic conditionals :math:`(D \\mid C)[l, u]`, which OWL cannot express,
    are given with ``conditionals`` and trained with
    :math:`[l\\,\\mathrm{Vol}(C) - \\mathrm{Vol}(D \\cap C)]_+ + [\\mathrm{Vol}(D \\cap C) -
    u\\,\\mathrm{Vol}(C)]_+`. A conditional probability is read off the embedding as the
    volume ratio :math:`\\mathrm{Vol}(D \\cap C)/\\mathrm{Vol}(C)` (:meth:`probability`).

    Conditionals use class IRIs, ``("and", e1, e2, ...)`` and ``("some", role_iri, e)`` in
    premise and conclusion, a generalisation of the four forms the paper's prototype supports
    ``(B|A)``, ``(B|A1⊓A2)``, ``(B|∃r.A)``, ``(∃r.B|A)``. :math:`\\top` is not allowed.

    Training adds the conditional losses to the BoxEL losses of the generic EL loop (via
    :meth:`additional_training_loss`), with the location regulariser :math:`L_{loc}` (and
    optionally :math:`L_{vol}`) of the paper.

    :param dataset: mOWL dataset (its ontology holds the deterministic axioms; it may be
        empty of axioms if everything is given as ``conditionals`` with ``l = u = 1``).
    :param conditionals: Iterable of :class:`Conditional` or ``(D, C, l, u)`` / ``(D, C, p)``.
    :param embed_dim: Dimension (paper: 8 to 128).
    :param beta: Side length of the cube the boxes lie in (paper: 1 or 10).
    :param location_weight: Weight of :math:`L_{loc}`.
    :param volume_weight: Weight of :math:`L_{vol}` as printed (Eq. 10); off by default
        because it is zero only for boxes that fill the cube (see the catalogue card).
    :param conditional_weight: Weight of the mean conditional loss.
    :param load_role_axioms: Also train on the role inclusions and role chains of the
        ontology (the module implements their losses). Default ``False``.
    :param temperature: Softplus sharpness of the volumes (``beta`` of ``softplus``).
    :param temperature_final: If given, the sharpness grows geometrically from
        ``temperature`` to this value over the epochs (the paper's cooling schedule, which
        lets the softplus volume converge to the volume).
    """

    axiom_scoring = "module"

    def __init__(self, dataset, conditionals=(), embed_dim=50, beta=1.0, location_weight=1.0,
                 volume_weight=0.0, conditional_weight=1.0, min_bounds=(1e-4, 0.2),
                 delta_bounds=(-0.1, 0), relation_bounds=(-0.1, 0.1),
                 scaling_bounds=(0.9, 1.1), temperature=1.0, temperature_final=None,
                 learning_rate=0.001, batch_size=4096 * 8, model_filepath=None, device="cpu",
                 neg_sampling_gcis=None, positive_scale=True,
                 role_chain_composition="in_order", load_role_axioms=False):
        self.beta = beta
        self.location_weight = location_weight
        self.volume_weight = volume_weight
        self.conditional_weight = conditional_weight
        self.min_bounds = list(min_bounds)
        self.delta_bounds = list(delta_bounds)
        self.relation_bounds = list(relation_bounds)
        self.scaling_bounds = list(scaling_bounds)
        self.temperature = temperature
        self.positive_scale = positive_scale
        self.role_chain_composition = role_chain_composition
        self.temperature_final = temperature_final
        super().__init__(dataset, embed_dim, batch_size, extended=True,
                         model_filepath=model_filepath, device=device,
                         learning_rate=learning_rate, neg_sampling_gcis=neg_sampling_gcis,
                         load_role_axioms=load_role_axioms)
        self.conditionals = [_as_conditional(c) for c in conditionals]
        self._groups = None
        self.init_module()

    def init_module(self):
        self.module = BoxSELModule(
            len(self.class_index_dict), len(self.object_property_index_dict),
            nb_inds=len(self.individual_index_dict) or None, embed_dim=self.embed_dim,
            min_bounds=self.min_bounds, delta_bounds=self.delta_bounds,
            relation_bounds=self.relation_bounds, scaling_bounds=self.scaling_bounds,
            temperature=self.temperature, beta=self.beta,
            location_weight=self.location_weight, volume_weight=self.volume_weight,
            positive_scale=self.positive_scale,
            role_chain_composition=self.role_chain_composition,
        ).to(self.device)

    # -- conditionals --------------------------------------------------------------------
    def _compile(self, expr):
        if expr == _THING:
            raise ValueError("owl:Thing has no finite box: conditionals over it are not "
                             "supported (the TBox must be safe, Definition 2).")
        return S.compile_expression(expr, self.class_index_dict,
                                    self.object_property_index_dict)

    def _compiled_groups(self):
        """Conditionals grouped by the structure of conclusion and premise."""
        if self._groups is None:
            groups = {}
            for c in self.conditionals:
                (sd, ld), (sc, lc) = self._compile(c.conclusion), self._compile(c.premise)
                g = groups.setdefault((sd, sc), {"d": [], "c": [], "l": [], "u": []})
                g["d"].append(ld)
                g["c"].append(lc)
                g["l"].append(c.lower)
                g["u"].append(c.upper)
            self._groups = {
                key: (th.tensor(g["d"], dtype=th.long, device=self.device),
                      th.tensor(g["c"], dtype=th.long, device=self.device),
                      th.tensor(g["l"], dtype=th.float, device=self.device),
                      th.tensor(g["u"], dtype=th.float, device=self.device))
                for key, g in groups.items()}
        return self._groups

    def add_conditionals(self, conditionals):
        """Add conditionals before training."""
        self.conditionals.extend(_as_conditional(c) for c in conditionals)
        self._groups = None

    def on_epoch_start(self, epoch, epochs):
        if self.temperature_final is not None and epochs > 1:
            frac = epoch / (epochs - 1)
            self.module.temperature = self.temperature * \
                (self.temperature_final / self.temperature) ** frac

    def additional_training_loss(self, epoch):
        """Mean conditional loss over each group of conditionals, summed over groups."""
        if not self.conditionals:
            return None
        total = th.zeros((), device=self.device)
        for (sd, sc), (ld, lc, lo, up) in self._compiled_groups().items():
            total = total + self.module.conditional_loss(sd, sc, ld, lc, lo, up).mean()
        return self.conditional_weight * total

    # -- inference -----------------------------------------------------------------------
    def probability(self, conclusion, premise):
        """:math:`p` with :math:`I_w \\models (D \\mid C)[p]` for the embedding: the proportion of
        the premise box that lies in the conclusion box (Theorem 2). ``nan`` if the premise
        has no volume."""
        (sd, ld), (sc, lc) = self._compile(conclusion), self._compile(premise)
        ld = th.tensor([ld], dtype=th.long, device=self.device)
        lc = th.tensor([lc], dtype=th.long, device=self.device)
        return float(self.module.probability(sd, sc, ld, lc)[0])

    def conditional_errors(self):
        """Per conditional, the distance of its embedded probability to ``[l, u]`` (zero
        if inside), as a list in the order of :attr:`conditionals`. The mean is the mean
        absolute error of the paper for point conditionals ``l = u``."""
        out = []
        for c in self.conditionals:
            p = self.probability(c.conclusion, c.premise)
            out.append(max(c.lower - p, p - c.upper, 0.0) if p == p else float("nan"))
        return out

    def mean_absolute_error(self):
        errors = self.conditional_errors()
        return float(np.mean(errors)) if errors else 0.0

    @staticmethod
    def probability_interval(models, conclusion, premise):
        """The interval estimate of the paper (Section 8.3): the minimum and maximum of the
        probabilities in ``models``, several embeddings of the same TBox (different seeds).
        Each is a lower/upper bound estimate of the entailed interval."""
        ps = [m.probability(conclusion, premise) for m in models]
        return min(ps), max(ps)
