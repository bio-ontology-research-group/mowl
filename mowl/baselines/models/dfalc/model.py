"""Differentiable fuzzy ALC (DF-ALC) over a finite domain of individuals."""
import numpy as np
import torch as th

from mowl.base_models.alcmodel import EmbeddingALCModel
from mowl.baselines.nn.alc.dfalc.module import DFALCModule

_THING = "http://www.w3.org/2002/07/owl#Thing"
_NOTHING = "http://www.w3.org/2002/07/owl#Nothing"


def _names(index_dict):
    out = [None] * len(index_dict)
    for name, i in index_dict.items():
        out[i] = name
    return out


class DFALCModel(EmbeddingALCModel):
    """Differentiable fuzzy :math:`\\mathcal{ALC}` (Wu et al., 2022, arXiv:2211.12006).

    The learned object is a *grounding*: the fuzzy interpretation over a finite domain
    :math:`\\Delta`, with a truth-value vector per concept name and a truth-value matrix per
    role name. The domain is the set of individuals of the ontology (the paper's
    :math:`\\Delta^I = N_I`), plus ``anon_domain`` anonymous elements. Complex concepts are
    evaluated with the Gödel semantics of fuzzy :math:`\\mathcal{ALC}` and the TBox is enforced
    by the hierarchical loss (Eq. 8) or the rule-based loss (Eq. 9, ``loss_type="rule"``). The
    grounding can be started from a *perceptual grounding* (:meth:`set_grounding`), which the
    training then revises into a model of the ontology (Proposition 2, zero loss).

    Differences to the paper (adaptations): axioms are not normalised, the hierarchical loss
    is applied to arbitrary :math:`\\mathcal{ALC}` inclusions (it only needs
    :math:`C^I(a) \\le D^I(a)`); the rule-based losses of normal forms 4-7 are
    implemented for the shapes ``L ⊑ ∃r.L``, ``L ⊑ ∀r.L``, ``∃r.L ⊑ L``, ``∀r.L ⊑ L``
    (``loss_type="rule"``; ``threshold`` is :math:`\\alpha = \\alpha'`), other quantified
    inclusions use the hierarchical loss; ABox assertions are fuzzy
    assertions with truth value 1 and the loss is the distance to 1.

    Differences to FALCON (:class:`~mowl.models.FALCONModel`): the domain is finite and
    explicit, there are no entity embeddings or membership networks, no sampled anonymous
    entities and no negative sampling.

    :param dataset: mOWL dataset.
    :param anon_domain: Number of anonymous domain elements added after the individuals.
        Required (``> 0``) if the ontology has no individuals. Defaults to ``0``.
    :param loss_type: ``"hierarchical"`` (default) or ``"rule"``.
    :param threshold: Threshold of the rule-based losses of normal forms 4-7 (default 0.8).
    :param alpha: Weight of the TBox loss. Defaults to ``0.4``.
    :param beta: Weight of the class assertion loss; role assertions get
        ``1 - alpha - beta``. Defaults to ``0.3``.
    :param learning_rate: Adam learning rate. Defaults to ``0.05``.
    :param batch_size: Batch size of the axiom loaders.
    :param init_std: Standard deviation of the logit initialisation.
    """

    scored_forms = ("gci0", "gci1", "gci2", "gci3")

    def __init__(self, dataset, anon_domain=0, loss_type="hierarchical", alpha=0.4, beta=0.3,
                 learning_rate=0.05, batch_size=256, init_std=1.0, threshold=0.8,
                 model_filepath=None,
                 device="cpu"):
        n = len(dataset.individuals) + anon_domain
        if n < 1:
            raise ValueError("The ontology has no individuals: set anon_domain > 0 to give "
                             "the fuzzy interpretation a domain.")
        super().__init__(dataset, embed_dim=n, batch_size=batch_size, alpha=alpha,
                         beta=beta, anon_e=0, learning_rate=learning_rate,
                         model_filepath=model_filepath, device=device)
        self.anon_domain = anon_domain
        self.loss_type = loss_type
        self.init_std = init_std
        self.threshold = threshold
        self.ndomain = n
        self.init_module()

    def init_module(self):
        cid = self.class_index_dict
        self.module = DFALCModule(
            len(self.dataset.classes), len(self.dataset.object_properties), self.ndomain,
            top_index=cid.get(_THING), bottom_index=cid.get(_NOTHING),
            loss_type=self.loss_type, init_std=self.init_std,
            threshold=self.threshold).to(self.device)

    # The domain is explicit: no sampled entities.
    def sample_anonymous_entities(self):
        return None

    def _entity_context(self, anon_e_emb):
        return None

    def _scoring_context(self):
        # Only its shape is used by the base class to bound the size of a scoring chunk
        # (rows x |Δ|^2 truth values for the role matrices).
        return th.empty(self.ndomain, self.ndomain)

    @property
    def domain(self):
        """Names of the domain elements: the individual IRIs, then ``anon_i``."""
        return _names(self.individual_index_dict) + [f"anon_{i}" for i in
                                                     range(self.anon_domain)]

    # -- grounding -----------------------------------------------------------------------
    def set_grounding(self, concepts=None, roles=None, eps=1e-3):
        """Initialise the interpretation with a (perceptual) grounding.

        :param concepts: Dict class IRI -> array of ``len(domain)`` truth values in
            ``[0, 1]`` (domain order of :attr:`domain`).
        :param roles: Dict role IRI -> ``(len(domain), len(domain))`` array of truth values.
        :param eps: Truth values are clipped to ``[eps, 1 - eps]`` before the logit.
        """
        def logit(a):
            a = th.as_tensor(np.asarray(a, dtype="float32")).clamp(eps, 1 - eps)
            return th.log(a / (1 - a))

        with th.no_grad():
            for iri, values in (concepts or {}).items():
                v = logit(values)
                if v.shape != (self.ndomain,):
                    raise ValueError(f"Grounding of {iri} must have shape ({self.ndomain},).")
                self.module.c_embedding.weight[self.class_index_dict[iri]] = v.to(self.device)
            for iri, values in (roles or {}).items():
                v = logit(values)
                if v.shape != (self.ndomain, self.ndomain):
                    raise ValueError(f"Grounding of {iri} must be {self.ndomain} x "
                                     f"{self.ndomain}.")
                self.module.r_embedding.weight[self.object_property_index_dict[iri]] = \
                    v.reshape(-1).to(self.device)

    def concept_grounding(self):
        """Dict class IRI -> numpy truth-value vector over :attr:`domain`."""
        idx = th.arange(len(self.class_index_dict), device=self.device)
        with th.no_grad():
            t = self.module.concept_truth(idx).cpu().numpy()
        return {name: t[i] for name, i in self.class_index_dict.items()}

    def role_grounding(self):
        """Dict role IRI -> numpy truth-value matrix over :attr:`domain`."""
        idx = th.arange(len(self.object_property_index_dict), device=self.device)
        with th.no_grad():
            t = self.module.role_truth(idx).cpu().numpy()
        return {name: t[i] for name, i in self.object_property_index_dict.items()}

    @property
    def class_embeddings(self):
        """Class IRI -> truth-value vector (the grounding is the embedding)."""
        return self.concept_grounding()

    @property
    def object_property_embeddings(self):
        """Role IRI -> truth-value matrix, flattened."""
        return {k: v.reshape(-1) for k, v in self.role_grounding().items()}

    @property
    def individual_embeddings(self):
        """Individual IRI -> its membership in every class (a column of the grounding)."""
        g = self.concept_grounding()
        return {name: np.array([g[c][i] for c in sorted(g)])
                for name, i in self.individual_index_dict.items()}

    def total_loss(self):
        """Current training loss (no gradient), as a float."""
        with th.no_grad():
            return float(self._epoch_loss(self.training_dataloaders, None))

    def violations(self, tolerance=1e-6):
        """The training axioms (as OWL axioms) whose fuzzy inclusion ``C^I(a) <= D^I(a)`` is
        violated by more than ``tolerance`` for some domain element (Eq. 6); assertions are
        violated if their truth value is below ``1 - tolerance``. Axiom patterns are
        reported with the number of violating rows."""
        out = []
        with th.no_grad():
            for axiom, loader in self.training_dataloaders.items():
                if self.axiom_category(axiom) is None:
                    continue
                bad = 0
                for batch in loader:
                    x = batch[0].to(self.device)
                    bad += int((self._row_violation(axiom, x) > tolerance).sum())
                if bad:
                    out.append((axiom, bad))
        return out

    def _row_violation(self, axiom, x):
        from mowl.owlapi import (OWLSubClassOfAxiom, OWLEquivalentClassesAxiom,
                                 OWLDisjointClassesAxiom)
        m = self.module
        if isinstance(axiom, OWLSubClassOfAxiom):
            c, nxt = m.forward_fs(axiom.getSubClass(), x)
            d, _ = m.forward_fs(axiom.getSuperClass(), x, None, nxt)
            return th.relu(c - d).amax(-1)
        if isinstance(axiom, OWLEquivalentClassesAxiom):
            ces = axiom.getClassExpressionsAsList()
            c, nxt = m.forward_fs(ces[0], x)
            d, _ = m.forward_fs(ces[1], x, None, nxt)
            return (c - d).abs().amax(-1)
        if isinstance(axiom, OWLDisjointClassesAxiom):
            ces = axiom.getClassExpressionsAsList()
            c, nxt = m.forward_fs(ces[0], x)
            d, _ = m.forward_fs(ces[1], x, None, nxt)
            return th.minimum(c, d).amax(-1)
        return m(axiom, x, None)

    # -- scores --------------------------------------------------------------------------
    def _subsumption_costs(self, sub, sup, x, e_emb):
        from mowl.baselines.nn.alc.dfalc.module import hierarchical_loss
        c, nxt = self.module.forward_fs(sub, x, e_emb)
        d, _ = self.module.forward_fs(sup, x, e_emb, cur_index=nxt)
        return hierarchical_loss(c, d)

    def score_subsumption(self, sub, sup):
        """Score of :math:`A \\sqsubseteq B` for class IRIs: minus the hierarchical loss."""
        from mowl.baselines.nn.alc.dfalc.module import hierarchical_loss
        cid = self.class_index_dict
        idx = th.tensor([cid[sub], cid[sup]], device=self.device)
        with th.no_grad():
            c = self.module.concept_truth(idx)
            return -float(hierarchical_loss(c[0:1], c[1:2]))

    def score_instance(self, individual, cls):
        """Truth value of ``cls(individual)`` in the grounding."""
        return float(self.concept_grounding()[cls][self.individual_index_dict[individual]])
