"""OnT: Ontology Transformer encoders (Yang, Chen, He, Gao, Horrocks, ISWC 2025)."""

import logging

import torch as th

from mowl.models.hit.model import HiT, DEFAULT_CENTRI_GRID

logger = logging.getLogger(__name__)


class OnT(HiT):
    """Ontology Transformer encoder (OnT) [yang2025ont]_.

    OnT extends :class:`~mowl.models.hit.HiT` from taxonomies to :math:`\\mathcal{EL}`
    (Section 4). Every concept, atomic or complex, is the point of the Poincaré ball
    obtained by encoding its verbalisation (Section 4.1, :class:`mowl.text.Verbaliser`);
    every role is a rotation (optionally followed by a Möbius scaling) whose parameters
    are a linear function of the encoding of the role's label (Section 4.2, Eq. 4). The
    training loss (Section 4.3) is the sum of

    * :math:`L_\\prec(x_C \\prec x_D)` (Eq. 5, HiT's clustering and centripetal losses)
      for every axiom :math:`C \\sqsubseteq D` of ``dataset.ontology``
      (``EquivalentClasses`` give both directions), with one random negative per axiom:
      a random class, or :math:`\\exists r.X` for a random class :math:`X` when
      :math:`D = \\exists r.D'` (as in the OnT data release);
    * :math:`L_r` (Eq. 6) for every occurrence of :math:`\\exists r.D`, aligning its
      verbalisation with :math:`f_r(x_D)`;
    * :math:`L_\\sqcap` (Eq. 7) for every occurrence of :math:`C_1 \\sqcap \\dots \\sqcap
      C_n` (mean over the conjuncts; Eq. 7 for :math:`n = 2`).

    As in the reference implementation, every training step takes a batch of
    subsumption axioms and an equally large random sample of existential restrictions
    and of conjunctions, whose losses use the negatives of the subsumption batch
    (randomly permuted); the losses are averaged over the batch (the paper writes sums).

    Defaults are those of Section 5.1 and the released configuration:
    all-MiniLM-L12-v2, :math:`\\alpha = 3.0`, :math:`\\beta = 0.5`, learning rate
    :math:`10^{-5}`, one epoch, batch size 64, one negative per axiom, 500 warm-up steps,
    :math:`\\lambda` chosen on validation subsumptions by MRR.

    The ontology is used as it is: OnT is normally trained on an ontology normalised by
    its own name-preserving normaliser (``normalization/`` of the reference code), which
    mOWL does not reimplement; the OnT benchmark datasets
    (:mod:`mowl.datasets.builtin.yang2025ont`) are already normalised. Axioms with
    ``owl:Nothing`` or with ``owl:Thing`` as superclass, and axioms outside
    :math:`\\mathcal{EL}`, are skipped (and counted in :attr:`skipped_axioms`).

    :param use_roles: ``False`` trains the ablation OnT(w/o r) (only Eq. 5).
    :param conjunction_weight: Weight of :math:`L_\\sqcap`.
    :param existential_weight: Weight of :math:`L_r`.
    :param role_pairing: ``"half"`` (reference code) or ``"adjacent"`` (paper's matrix),
        see :func:`mowl.nn.lm.hyperbolic.rotate`.
    :param role_scaling: Apply the scaling :math:`k_r` of Eq. 4 (the reference code does
        not).
    :param existential_mode: How :math:`\\exists r.D` is embedded when scoring:
        ``"text"`` (the encoding of its verbalisation; the reference evaluation's default
        "sentence" mode) or ``"role"`` (:math:`f_r(x_D)`, the "constructed" mode).

    The other parameters are those of :class:`~mowl.models.hit.HiT`.
    """

    default_margins = (3.0, 0.5)

    scored_forms = ("gci0", "gci2", "gci3")

    def __init__(self, dataset, encoder="sentence-transformers/all-MiniLM-L12-v2",
                 verbaliser=None, labels=None, verbalisation_style="paper",
                 label_fallback="fragment", clustering_margin=None, centripetal_margin=None,
                 clustering_weight=1.0, centripetal_weight=1.0, curvature=None,
                 learning_rate=1e-5, weight_decay=0.0, warmup_steps=500, batch_size=64,
                 num_negatives=1, use_roles=True, conjunction_weight=1.0,
                 existential_weight=1.0, role_pairing="half", role_scaling=False,
                 existential_mode="text", centri_weight=None,
                 centri_weight_grid=DEFAULT_CENTRI_GRID, encode_batch_size=256,
                 score_dtype="float64", seed=None, device="cpu", model_filepath=None):
        if existential_mode not in ("text", "role"):
            raise ValueError("existential_mode must be 'text' or 'role'.")
        self.use_roles = use_roles
        self.conjunction_weight = conjunction_weight
        self.existential_weight = existential_weight
        self.role_pairing = role_pairing
        self.role_scaling = role_scaling
        self.existential_mode = existential_mode
        self.skipped_axioms = 0
        super().__init__(dataset, encoder=encoder, verbaliser=verbaliser, labels=labels,
                         verbalisation_style=verbalisation_style,
                         label_fallback=label_fallback, clustering_margin=clustering_margin,
                         centripetal_margin=centripetal_margin,
                         clustering_weight=clustering_weight,
                         centripetal_weight=centripetal_weight, curvature=curvature,
                         learning_rate=learning_rate, weight_decay=weight_decay,
                         warmup_steps=warmup_steps, batch_size=batch_size,
                         num_negatives=num_negatives, centri_weight=centri_weight,
                         centri_weight_grid=centri_weight_grid,
                         encode_batch_size=encode_batch_size, score_dtype=score_dtype,
                         seed=seed, device=device,
                         model_filepath=model_filepath)
        self.existentials = []
        self.conjunctions = []

    def init_module(self):
        from mowl.nn.lm.ont import OnTModule
        return OnTModule(self.encoder, curvature=self.curvature,
                         clustering_margin=self.clustering_margin,
                         centripetal_margin=self.centripetal_margin,
                         clustering_weight=self.clustering_weight,
                         centripetal_weight=self.centripetal_weight,
                         use_roles=self.use_roles,
                         conjunction_weight=self.conjunction_weight,
                         existential_weight=self.existential_weight,
                         role_pairing=self.role_pairing, role_scaling=self.role_scaling)

    # ------------------------------------------------------------------ training data
    def training_axioms(self):
        """``(subclass, superclass)`` class-expression pairs of the training ontology."""
        from org.semanticweb.owlapi.model import AxiomType
        pairs, skipped = [], 0
        for ax in self.dataset.ontology.getAxioms():
            t = ax.getAxiomType()
            if t == AxiomType.SUBCLASS_OF:
                cand = [(ax.getSubClass(), ax.getSuperClass())]
            elif t == AxiomType.EQUIVALENT_CLASSES:
                ops = list(ax.getClassExpressionsAsList())
                cand = [(x, y) for x in ops for y in ops if not x.equals(y)]
            else:
                continue
            for sub, sup in cand:
                if sup.isOWLThing() or sub.isOWLNothing() or sup.isOWLNothing() or \
                        any(c.isOWLNothing() for c in sub.getClassesInSignature()):
                    skipped += 1
                    continue
                pairs.append((sub, sup))
        self.skipped_axioms = skipped
        return pairs

    def _collect(self, expr):
        from org.semanticweb.owlapi.model import OWLObjectIntersectionOf, \
            OWLObjectSomeValuesFrom
        if isinstance(expr, OWLObjectSomeValuesFrom):
            self.existentials.append((self.text(expr), self.verbaliser.role(expr.getProperty()),
                                      self.text(expr.getFiller())))
            self._collect(expr.getFiller())
        elif isinstance(expr, OWLObjectIntersectionOf):
            ops = list(expr.getOperandsAsList())
            self.conjunctions.append((self.text(expr), [self.text(o) for o in ops]))
            for o in ops:
                self._collect(o)

    def _negative(self, sup, pool):
        from org.semanticweb.owlapi.model import OWLObjectSomeValuesFrom
        from mowl.owlapi import OWLAPIAdapter
        x = self.rng.choice(pool)
        if isinstance(sup, OWLObjectSomeValuesFrom):
            adapter = OWLAPIAdapter()
            return self.text(adapter.create_object_some_values_from(
                sup.getProperty(), adapter.create_class(x)))
        return self.text(x)

    def training_triplets(self):
        """``(V(C), V(D), negative)`` for every training axiom (``num_negatives`` each);
        also fills :attr:`existentials` and :attr:`conjunctions`."""
        pool = self.candidate_iris
        self.existentials, self.conjunctions = [], []
        triplets, unsupported = [], 0
        for sub, sup in self.training_axioms():
            try:
                child, parent = self.text(sub), self.text(sup)
            except NotImplementedError:
                unsupported += 1
                continue
            self._collect(sub)
            self._collect(sup)
            for _ in range(self.num_negatives):
                triplets.append((child, parent, self._negative(sup, pool)))
        self.skipped_axioms += unsupported
        if self.skipped_axioms:
            logger.info("OnT skipped %d axioms (bottom, top superclass or not EL).",
                        self.skipped_axioms)
        if not triplets:
            raise ValueError("The training ontology has no usable subsumption axioms.")
        return triplets

    # ------------------------------------------------------------------ training
    def _sample(self, items, n):
        if len(items) <= n:
            return list(items)
        return self.rng.sample(items, n)

    def batch_loss(self, batch, negatives):
        """Mean of :math:`L_\\prec` over the batch plus, if roles are used,
        ``existential_weight`` times the mean :math:`L_r` and ``conjunction_weight``
        times the mean :math:`L_\\sqcap` of random samples of the size of the batch."""
        m = self.module
        child = m.embed([t[0] for t in batch])
        parent = m.embed([t[1] for t in batch])
        hier = m.hierarchy_loss(child, parent, negatives).mean()
        parts = {"hierarchy": float(hier.detach())}
        loss = hier
        n = len(batch)
        if self.use_roles and self.existentials:
            ex = self._sample(self.existentials, n)
            k = len(ex)
            neg1 = negatives[th.randperm(n, device=negatives.device)[:k]]
            neg2 = negatives[th.randperm(n, device=negatives.device)[:k]]
            l_r = m.existential_loss([e[0] for e in ex], [e[1] for e in ex],
                                     [e[2] for e in ex], neg1, neg2).mean()
            loss = loss + self.existential_weight * l_r
            parts["existential"] = float(l_r.detach())
        if self.use_roles and self.conjunctions:
            cj = self._sample(self.conjunctions, n)
            conj_texts, part_texts, weights = [], [], []
            for text, ops in cj:
                for o in ops:
                    conj_texts.append(text)
                    part_texts.append(o)
                    weights.append(1.0 / len(ops))
            idx = th.randint(0, n, (len(conj_texts),), device=negatives.device)
            rows = m.hierarchy_loss(m.embed(conj_texts), m.embed(part_texts), negatives[idx])
            w = th.tensor(weights, device=rows.device, dtype=rows.dtype)
            l_c = (w * rows).sum() / len(cj)
            loss = loss + self.conjunction_weight * l_c
            parts["conjunction"] = float(l_c.detach())
        return loss, parts

    # ------------------------------------------------------------------ roles
    def role_parameters(self, roles=None):
        """``{iri: (scaling, angles)}`` of the role transformations :math:`f_r`.

        :param roles: Object property IRIs (default: all of the dataset).
        """
        roles = list(self.object_property_index_dict) if roles is None else list(roles)
        emb = self.embed_texts([self.verbaliser.label(r) for r in roles])
        with th.no_grad():
            scaling, angles = self.module.role_model.parameters_of(emb)
        return {r: (float(s), a.cpu().numpy()) for r, s, a in zip(roles, scaling, angles)}

    @property
    def object_property_embeddings(self):
        """``{iri: numpy vector}``: the encodings of the role labels, from which the role
        transformations are computed."""
        roles = list(self.object_property_index_dict)
        emb = self.embed_texts([self.verbaliser.label(r) for r in roles]).cpu().numpy()
        return dict(zip(roles, emb))

    def _existential_embeddings(self, roles, fillers, inv_r, inv_c):
        if self.existential_mode == "text":
            return super()._existential_embeddings(roles, fillers, inv_r, inv_c)
        mat = self._class_matrix()
        role_emb = self.embed_texts([self.verbaliser.label(inv_r[int(r)])
                                     for r in roles.tolist()])
        with th.no_grad():
            return self.module.role_model(mat[fillers.to(mat.device)], role_emb, self.c)
