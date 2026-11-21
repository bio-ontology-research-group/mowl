"""Learnable axis-aligned cone (al-cone) embeddings for :math:`\\mathcal{ALC}`."""
import torch as th

from mowl.base_models.alcmodel import EmbeddingALCModel
from mowl.nn.alc.alcone.module import ALConeModule

_THING = "http://www.w3.org/2002/07/owl#Thing"
_NOTHING = "http://www.w3.org/2002/07/owl#Nothing"


class ALConeModel(EmbeddingALCModel):
    """Al-cone embedding of an :math:`\\mathcal{ALC}` ontology (adaptation).

    The geometry is that of Özçep, Leemhuis & Wolter (JAIR 78, 2023): concepts are
    axis-aligned cones of :math:`\\mathbb{R}^n`, negation is the polar cone, disjunction
    the convex union, roles are incidence matrices over half-axes (Def. 7). The papers
    give no learning algorithm for ontology embedding; the training objective here is
    ours: every concept, role and individual is a vector of half-axis logits, complex
    concepts are evaluated with Gödel operators, and each axiom has a margin loss that is
    zero only if the discretised interpretation (:meth:`discretise`) satisfies it (see
    :mod:`mowl.nn.alc.alcone.losses`). The discretised result can be verified exactly
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
    """

    def __init__(self, dataset, dim=8, margin=0.1, nondegeneracy=0.1, alpha=0.4, beta=0.3,
                 learning_rate=0.05, batch_size=256, init_std=0.5, temperature=0.3,
                 noise=0.3, model_filepath=None, device="cpu"):
        super().__init__(dataset, embed_dim=dim, batch_size=batch_size, alpha=alpha,
                         beta=beta, anon_e=0, learning_rate=learning_rate,
                         model_filepath=model_filepath, device=device)
        self.dim = dim
        self.margin = margin
        self.nondegeneracy = nondegeneracy
        self.init_std = init_std
        self.temperature = temperature
        self.noise = noise
        self.init_module()

    def init_module(self):
        cid = self.class_index_dict
        self.module = ALConeModule(
            len(self.dataset.classes), len(self.dataset.individuals),
            len(self.dataset.object_properties), dim=self.dim, margin=self.margin,
            top_index=cid.get(_THING), bottom_index=cid.get(_NOTHING),
            init_std=self.init_std).to(self.device)

    # The al-cone module evaluates concepts on half-axes, not on sampled entities.
    def sample_anonymous_entities(self):
        return None

    def _entity_context(self, anon_e_emb):
        return None

    def _named_class_indices(self):
        cid = self.class_index_dict
        idx = [i for name, i in cid.items() if name not in (_THING, _NOTHING)]
        return th.tensor(idx, dtype=th.long, device=self.device)

    def _epoch_loss(self, dataloaders, e_emb):
        total = super()._epoch_loss(dataloaders, e_emb)
        if self.nondegeneracy > 0:
            idx = self._named_class_indices()
            if len(idx) > 0:
                reg = self.nondegeneracy * self.module.nondegeneracy_loss(idx)
                total = reg if total is None else total + reg
        return total

    def train(self, epochs, validate_every=1, temperature=None):
        """Train with Adam on the axiom losses.

        The Gödel operators have sparse subgradients, so training uses the smooth
        relaxation of :mod:`mowl.nn.alc.alcone.losses` with a temperature that decreases
        linearly from ``temperature`` (default: the model's ``temperature``) to ``0``
        over the epochs, with parameter noise proportional to the temperature (see
        ``noise``). After training the module is left at temperature ``0`` (exact
        operators). Validation and checkpointing are as in
        :meth:`EmbeddingALCModel.train`, on the exact loss.

        :param epochs: Number of epochs (one optimisation step each).
        :param validate_every: Validate every this many epochs.
        :param temperature: Initial temperature.
        """
        tau0 = self.temperature if temperature is None else temperature
        self.module = self.module.to(self.device)
        optimizer = th.optim.Adam(self.module.parameters(), lr=self.learning_rate)
        best = float("inf")
        for epoch in range(epochs):
            self.module.train()
            self.module.temperature = tau0 * (1.0 - epoch / max(epochs, 1))
            loss = self._epoch_loss(self.training_dataloaders, None)
            if loss is None:
                raise ValueError("No trainable axioms were found in the training ontology.")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if self.noise > 0 and self.module.temperature > 0:
                with th.no_grad():
                    for p in self.module.parameters():
                        p.add_(self.noise * self.module.temperature * th.randn_like(p))
            if self.dataset.validation and (epoch + 1) % validate_every == 0:
                self.module.temperature = 0.0
                with th.no_grad():
                    self.module.eval()
                    valid = self._epoch_loss(self.validation_dataloaders, None)
                if valid is not None and float(valid) < best:
                    best = float(valid)
                    if self.model_filepath is not None:
                        th.save(self.module.state_dict(), self.model_filepath)
        self.module.temperature = 0.0
        if self.model_filepath is not None and best == float("inf"):
            th.save(self.module.state_dict(), self.model_filepath)

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
        from mowl.nn.alc.alcone.losses import inclusion_loss
        c, nxt = self.module.forward_fs(sub, x, e_emb)
        d, _ = self.module.forward_fs(sup, x, e_emb, cur_index=nxt)
        return inclusion_loss(c, d, 0.0)

    def score_subsumption(self, sub, sup):
        """Score of :math:`A \\sqsubseteq B` for class IRIs ``sub`` and ``sup``: minus the
        inclusion loss with margin 0 (``0`` iff the soft cones satisfy it)."""
        from mowl.nn.alc.alcone.losses import inclusion_loss
        cid = self.class_index_dict
        idx = th.tensor([cid[sub], cid[sup]], device=self.device)
        with th.no_grad():
            c = self.module.class_logits(idx)
            return -float(inclusion_loss(c[0:1], c[1:2], 0.0))
