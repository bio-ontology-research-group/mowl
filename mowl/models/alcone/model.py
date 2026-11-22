"""Learnable axis-aligned cone (al-cone) embeddings for :math:`\\mathcal{ALC}`."""
import torch as th

from mowl.models.composed.alc import ComposedALC
from mowl.nn.alc.alcone.module import ALConeModule

_THING = "http://www.w3.org/2002/07/owl#Thing"
_NOTHING = "http://www.w3.org/2002/07/owl#Nothing"


class ALConeModel(ComposedALC):
    """Al-cone embedding of an :math:`\\mathcal{ALC}` ontology (adaptation).

    The geometry is that of Özçep, Leemhuis & Wolter (JAIR 78, 2023): concepts are
    axis-aligned cones of :math:`\\mathbb{R}^n`, negation is the polar cone, disjunction
    the convex union, roles are incidence matrices over half-axes (Def. 7). The papers
    give no learning algorithm for ontology embedding; the training objective here is
    ours: every concept, role and individual is a vector of half-axis logits, complex
    concepts are evaluated with Gödel operators, and each axiom has a margin loss that is
    zero only if the discretised interpretation (:meth:`discretise`) satisfies it (see
    :mod:`mowl.baselines.nn.alc.alcone.losses`). The discretised result can be verified exactly
    with :class:`~mowl.geometric.cones.ALConeChecker` (:meth:`violations`).

    Differences to FALCON (:class:`~mowl.models.FALCONModel`), which shares the training
    loop: no anonymous entities and no negative sampling are used (the half-axes act as
    the domain); a non-degeneracy regulariser keeps named concepts different from
    :math:`\\{0\\}`.

    :param dataset: mOWL dataset.
    :param dim: Dimension :math:`n` of the cone space (``2 n`` half-axes). Defaults to ``8``.
    :param margin: Margin :math:`\\gamma > 0` of the losses. Defaults to ``0.1``.
    :param nondegeneracy: Weight of the regulariser that pushes every named class
        (except ``owl:Nothing``) away from :math:`\\{0\\}`. It assumes the named classes
        are satisfiable; set it to ``0`` otherwise. Defaults to ``0.1``.
    :param alpha: Weight of the TBox loss. Defaults to ``0.4``.
    :param beta: Weight of the class-assertion loss; role assertions get
        ``1 - alpha - beta``. Defaults to ``0.3``.
    :param learning_rate: Adam learning rate. Defaults to ``0.05``.
    :param batch_size: Batch size of the axiom data loaders. Defaults to ``256``.
    :param init_std: Standard deviation of the logit initialisation. Defaults to ``0.5``.
    :param temperature: Initial temperature of the smooth relaxation used in training
        (annealed to ``0``; see :meth:`train`). Defaults to ``0.3``.
    :param noise: Scale of Gaussian noise added to the parameters after each step,
        multiplied by the current temperature (so it vanishes at the end of training);
        it helps to leave local minima. Defaults to ``0.3``.
    :param constant_logit: Magnitude of the logits of ``owl:Thing`` and ``owl:Nothing``
        (see :class:`~mowl.baselines.nn.alc.alcone.module.ALConeModule`). Defaults to ``1.0``; the
        value of earlier versions, ``1e4``, made every loss through them 1e4 times larger.

    The model is a combination of parts (:class:`~mowl.models.ComposedALC` with the method
    ``"alcone"``, see :func:`mowl.representation.library.logic_fuzzy.alcone`): its module
    (:class:`~mowl.nn.alc.alcone.module.ALConeModule`) is a
    :class:`~mowl.representation.Geometry` whose class model is
    :class:`~mowl.representation.library.logic_fuzzy.HalfAxisClasses`, relation model
    :class:`~mowl.representation.library.logic_fuzzy.IncidenceMatrices`, measure
    :class:`~mowl.representation.library.logic_fuzzy.ConeMeasure` and objective
    :class:`~mowl.representation.library.logic_fuzzy.FuzzyObjective`; the penalty against empty
    classes is the regulariser :class:`~mowl.representation.library.logic_fuzzy.NonDegeneracy`
    and the annealed training loop a
    :class:`~mowl.representation.library.logic_alc.AnnealedRelaxation`. The hand-written
    implementation it replaces is :class:`mowl.baselines.models.alcone.model.ALConeModel`.
    """

    def __init__(self, dataset, dim=8, margin=0.1, nondegeneracy=0.1, alpha=0.4, beta=0.3,
                 learning_rate=0.05, batch_size=256, init_std=0.5, temperature=0.3,
                 noise=0.3, constant_logit=1.0, model_filepath=None, device="cpu"):
        self.dim = dim
        self.nondegeneracy = nondegeneracy
        self.init_std = init_std
        self.constant_logit = constant_logit
        super().__init__(dataset, method="alcone", embed_dim=dim, batch_size=batch_size,
                         alpha=alpha, beta=beta, anon_e=0, learning_rate=learning_rate,
                         margin=margin, model_filepath=model_filepath, device=device)
        self.temperature = temperature
        self.noise = noise

    @property
    def temperature(self):
        """Initial temperature of the smooth relaxation of training (the training schedule's)."""
        return self.module.schedule.temperature

    @temperature.setter
    def temperature(self, value):
        self.module.schedule.temperature = value

    @property
    def noise(self):
        """Scale of the parameter noise of training (the training schedule's)."""
        return self.module.schedule.noise

    @noise.setter
    def noise(self, value):
        self.module.schedule.noise = value

    def init_module(self):
        cid = self.class_index_dict
        self.module = ALConeModule(
            len(self.dataset.classes), len(self.dataset.individuals),
            len(self.dataset.object_properties), dim=self.dim, margin=self.margin,
            top_index=cid.get(_THING), bottom_index=cid.get(_NOTHING),
            init_std=self.init_std, constant_logit=self.constant_logit,
            nondegeneracy=self.nondegeneracy, anneal=0.3, noise=0.3).to(self.device)

    def train(self, epochs, validate_every=1, temperature=None):
        """Train with Adam on the axiom losses.

        The Gödel operators have sparse subgradients, so training uses the smooth
        relaxation of the operators with a temperature that decreases linearly from
        ``temperature`` (default: the model's ``temperature``) to ``0`` over the epochs, with
        parameter noise proportional to the temperature (see ``noise``): the training schedule
        :class:`~mowl.representation.library.logic_alc.AnnealedRelaxation` of the geometry.
        After training the module is left at temperature ``0`` (exact operators). Validation
        and checkpointing are as in :meth:`EmbeddingALCModel.train`, on the exact loss.

        :param epochs: Number of epochs (one optimisation step each).
        :param validate_every: Validate every this many epochs.
        :param temperature: Initial temperature.
        """
        schedule = self.module.schedule
        kept = schedule.temperature
        if temperature is not None:
            schedule.temperature = temperature
        try:
            super().train(epochs, validate_every)
        finally:
            schedule.temperature = kept

    def total_loss(self):
        """Current training loss with the exact operators (no gradient), as a float."""
        self.module.temperature = 0.0
        with th.no_grad():
            return float(self._epoch_loss(self.training_dataloaders, None))

    # -- exact view ----------------------------------------------------------
    def discretise(self):
        """Return the discretised :class:`~mowl.geometric.cones.ALConeInterpretation`
        with class, role and individual IRIs as names."""
        def names(index_dict):
            out = [None] * len(index_dict)
            for name, i in index_dict.items():
                out[i] = name
            return out
        return self.module.discretise(names(self.class_index_dict),
                                      names(self.object_property_index_dict),
                                      names(self.individual_index_dict))

    def alc_axioms(self, ontology=None):
        """The :math:`\\mathcal{ALC}` axioms of ``ontology`` (default: the training
        ontology) in the syntax of :mod:`mowl.geometric.cones`, and the skipped axioms."""
        from mowl.geometric.cones import axioms_from_owl
        return axioms_from_owl(self.dataset.ontology if ontology is None else ontology)

    def violations(self, ontology=None):
        """Axioms of ``ontology`` (default: training ontology) that the discretised
        interpretation violates, decided by the exact
        :class:`~mowl.geometric.cones.ALConeChecker`."""
        from mowl.geometric.cones import ALConeChecker
        axioms, _ = self.alc_axioms(ontology)
        return ALConeChecker(self.discretise()).violations(axioms)

    def _subsumption_costs(self, sub, sup, x, e_emb):
        return super()._subsumption_costs(sub, sup, x, e_emb)

    def score_subsumption(self, sub, sup):
        """Score of :math:`A \\sqsubseteq B` for class IRIs ``sub`` and ``sup``: minus the
        inclusion loss with margin 0 (``0`` iff the soft cones satisfy it)."""
        cid = self.class_index_dict
        idx = th.tensor([cid[sub], cid[sup]], device=self.device)
        with th.no_grad():
            c = self.module.classes(idx)
            return -float(self.module.measure.inside(
                type(c)(c.values[0:1], c.family), type(c)(c.values[1:2], c.family), 0.0))
