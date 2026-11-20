"""CatE: order embeddings of the lattice of ALC concept descriptions [zhapa2023]_."""

import logging
import os

import torch as th

from mowl.base_models.model import Model
from mowl.nn.graph.cate.losses import order_margin_loss
from mowl.nn.graph.cate.module import CatEModule
from mowl.projection.categorical import saturation as sat
from mowl.projection.categorical.lattice import CatELattice, UnseenConceptError

logger = logging.getLogger(__name__)

__all__ = ["CatE", "UnseenConceptError"]


class CatE(Model):
    """CatE (Zhapa-Camacho and Hoehndorf, *Lattice-based ALC ontology embeddings with
    saturation*, arXiv:2305.07163v3; NeSy 2024 for the version without saturation).

    **Method.** (i) The training ontology is turned into the preorder
    :math:`(\\tilde C, \\sqsubseteq)` of its concept descriptions and their
    sub-descriptions (Section 3) by
    :class:`~mowl.projection.categorical.model.CategoricalProjector`, wrapped in
    :class:`~mowl.projection.categorical.lattice.CatELattice`; (ii) optionally the lattice
    is partially saturated (Section 3.1, :mod:`mowl.projection.categorical.saturation`);
    (iii) every element gets a vector, and every morphism :math:`A \\sqsubseteq B` is
    trained to satisfy :math:`f(A) \\preceq f(B)` in the product order with the score
    :math:`s(A, B) = \\lVert \\max(0, f(A) - f(B)) \\rVert_2` (Eq. 6) and the margin loss of
    Eq. 7, whose negatives replace :math:`B` by an element drawn uniformly from the
    lattice (:class:`~mowl.nn.graph.cate.CatEModule`).

    **Scoring.** A query is scored by :math:`s` between lattice elements (lower is
    better): :math:`C \\sqsubseteq D` by :math:`s(C, D)`, :math:`C(a)` by
    :math:`s(\\{a\\}, C)`, :math:`C \\sqsubseteq \\exists r.D` by :math:`s(C, \\exists r.D)`.
    Both sides must be elements of the lattice. CatE cannot embed other concept
    descriptions (paper, Section 5); scoring one raises
    :class:`~mowl.projection.categorical.lattice.UnseenConceptError`. For ranking
    :math:`C \\sqsubseteq \\exists r.D'` over candidates :math:`D'` the elements
    :math:`\\exists r.D'` must be added before training (``existentials``), as the paper
    does for the PPI task (Section 4.4).

    **Training (from the paper and the reference code).** Adam; triangular cyclic
    learning rate between ``learning_rate / 10`` and ``learning_rate`` with 20 steps per
    half cycle, stepped per batch; weight decay :math:`10^{-4}`; Xavier-uniform
    initialisation; validation every ``validate_every`` epochs by the raw mean
    reciprocal rank of the validation axioms, keeping the best parameters and stopping
    after ``early_stopping_patience`` validations without improvement. The paper states
    only "Adam with a Cyclic Learning Rate scheduler" (Section 4.1); bounds, step size,
    weight decay, initialisation and model selection are taken from the reference code
    (``src/cat.py``).

    **Deviations.** (a) ``negative_aggregation="per_negative"`` (default) is Eq. 7; the
    reference code averages the scores of the ``num_negs`` corruptions before the hinge
    (``"mean"``). Eq. 7 sums over corruptions that are *not* morphisms; by default
    (``filter_negatives=True``) corruptions that are morphisms are discarded, while the
    reference code keeps them (``filter_negatives=False``). Keeping them makes a zero
    loss impossible: every element has the morphism :math:`X \\sqsubseteq \\top`.
    (b) Nodes are identified by unambiguous names; the reference trained on the
    projector's string rendering, which drops parentheses, so distinct descriptions
    could share a vector. (c) ``complete_negation=True`` adds the full negation diagram of
    Figure 2c for complex operands (see
    :func:`~mowl.projection.categorical.saturation.complete_negation`); ``False`` gives the
    projector's lattice. (d) The reference PPI training adds a second loss for the
    morphisms of a batch whose head is a protein, with negatives drawn among the
    :math:`\\exists
    interacts.p` elements; this is not described in the paper and is optional
    here (interaction_loss, part of the cate-code-training regime). (e) Every class and
    individual of the dataset (including those that occur only in the validation or test
    ontology) is an element, with only :math:`\\bot \\sqsubseteq X \\sqsubseteq \\top` (and
    the negation diagram for classes) if it is not in the training ontology.

    :param dataset: mOWL dataset; the lattice is built from ``dataset.ontology``.
    :type dataset: :class:`mowl.datasets.Dataset`
    :param embed_dim: Embedding dimension.
    :type embed_dim: int
    :param margin: :math:`\\gamma` of Eq. 7.
    :type margin: float
    :param num_negs: Corruptions per morphism.
    :type num_negs: int
    :param learning_rate: Upper bound of the cyclic learning rate.
    :type learning_rate: float
    :param batch_size: Morphisms per batch.
    :type batch_size: int
    :param p: Norm of Eq. 6.
    :type p: int
    :param weight_decay: Adam weight decay.
    :type weight_decay: float
    :param cyclic_lr: Use the cyclic learning rate (otherwise a constant one).
    :type cyclic_lr: bool
    :param negative_aggregation: ``"per_negative"`` (Eq. 7) or ``"mean"`` (reference
        code), see :func:`~mowl.nn.graph.cate.order_margin_loss`.
    :type negative_aggregation: str
    :param filter_negatives: Discard corruptions that are morphisms of the lattice (or
        reflexive), as Eq. 7 sums over non-morphisms only; ``False`` keeps them, as the
        reference code does.
    :type filter_negatives: bool
    :param saturation_steps: Iterations of the saturation rules (0: none, ``None``: until
        a fixed point). ``1`` is the paper's S1.
    :type saturation_steps: int or None
    :param saturation_rules: Rule names of
        :data:`mowl.projection.categorical.saturation.RULES` (default: S1).
    :type saturation_rules: tuple
    :param transitive_closure: Add the transitive closure: ``True`` for the paper's Tr as
        the reference code computes it (not through :math:`\\bot`, :math:`\\top`),
        ``"full"`` for the complete closure, under which a zero loss is attainable.
    :type transitive_closure: bool or str
    :param complete_negation: See deviation (c).
    :type complete_negation: bool
    :param existentials: Elements :math:`\\exists r.D` to add: a dict from role IRI to
        ``"classes"``, ``"evaluation_classes"`` or an iterable of class IRIs; ``None``;
        or ``"auto"`` (default): if the dataset defines ``get_evaluation_property()`` (the
        PPI and GDA datasets), :math:`\\exists r.D` for that role and every tail
        evaluation class, otherwise none.
    :param interaction_loss: Add the PPI training term of the reference code
        (``src/cat_ppi_evaluator.py``, ``train``): for every morphism of the batch whose
        head is one of the dataset's head evaluation classes (the proteins), the mean
        order violation of the morphism plus a hinge on the violation of a corruption
        whose tail is drawn uniformly among the elements :math:`\\exists r.p` of the
        dataset's evaluation property. Not described in the paper; the published Table 4
        was produced with it.
    :param early_stopping_patience: Validations without improvement before stopping
        (``None``: never stop early).
    :type early_stopping_patience: int or None
    :param model_filepath: Where the parameters are saved.
    :type model_filepath: str, optional
    :param device: Torch device.
    :type device: str
    """

    def __init__(self, dataset, embed_dim=200, margin=1.0, num_negs=4,
                 learning_rate=1e-4, batch_size=32768, p=2, weight_decay=1e-4,
                 cyclic_lr=True, negative_aggregation="per_negative", filter_negatives=True,
                 saturation_steps=0,
                 saturation_rules=sat.S1, transitive_closure=False, complete_negation=True,
                 existentials="auto", interaction_loss=False, early_stopping_patience=10,
                 model_filepath=None, device="cpu"):
        super().__init__(dataset, model_filepath=model_filepath)
        if negative_aggregation not in ("per_negative", "mean"):
            raise ValueError("negative_aggregation must be 'per_negative' or 'mean'.")
        self.embed_dim = embed_dim
        self.margin = margin
        self.num_negs = num_negs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.p = p
        self.weight_decay = weight_decay
        self.cyclic_lr = cyclic_lr
        self.negative_aggregation = negative_aggregation
        self.filter_negatives = filter_negatives
        self.saturation_steps = saturation_steps
        self.saturation_rules = tuple(saturation_rules)
        self.transitive_closure = transitive_closure
        self.complete_negation = complete_negation
        self.existentials = existentials
        self.interaction_loss = interaction_loss
        self.early_stopping_patience = early_stopping_patience
        self.device = device
        self._training_regime = None
        self._lattice = None
        self._module = None
        self._validation = None
        self._keys = None
        self._interaction = None

    # ---------------------------------------------------------------- lattice
    def _resolve_existentials(self):
        spec = self.existentials
        ds = self.dataset
        if spec is None:
            return {}
        if spec == "auto":
            prop = getattr(ds, "get_evaluation_property", None)
            if prop is None:
                return {}
            try:
                spec = {prop(): "evaluation_classes"}
            except (NotImplementedError, AttributeError, TypeError):
                return {}
        out = {}
        for role, fillers in dict(spec).items():
            if fillers == "classes":
                fillers = list(ds.classes.as_str)
            elif fillers == "evaluation_classes":
                fillers = list(ds.evaluation_classes[1].as_str)
            out[str(role)] = [str(f) for f in fillers]
        return out

    @property
    def lattice(self):
        """The :class:`~mowl.projection.categorical.lattice.CatELattice` of the training
        ontology (built on first access)."""
        if self._lattice is None:
            ds = self.dataset
            self._lattice = CatELattice(
                ds.ontology, classes=ds.classes.as_str, individuals=ds.individuals.as_str,
                existentials=self._resolve_existentials(),
                complete_negation=self.complete_negation,
                saturation_steps=self.saturation_steps, rules=self.saturation_rules,
                transitive_closure=self.transitive_closure)
            logger.info("CatE lattice: %s", self._lattice.stats)
        return self._lattice

    @property
    def class_index_dict(self):
        """Class IRI to lattice element id (including ``owl:Thing`` and ``owl:Nothing``).

        :rtype: dict
        """
        return self.lattice.class_to_id

    @property
    def individual_index_dict(self):
        """Individual IRI to the lattice element id of :math:`\\{a\\}`.

        :rtype: dict
        """
        return self.lattice.individual_to_id

    @property
    def object_property_index_dict(self):
        """Role IRI to role index (used in ``gci2`` rows).

        :rtype: dict
        """
        return self.lattice.role_to_id

    @property
    def training_regime(self):
        """A :class:`~mowl.training.TrainingRegime`, or ``None`` to train with the
        model's own behaviour.

        Assigning a regime calls its ``attach`` method with this model; the
        'cate-code-training' regime switches the model to the training design of the
        CatE reference code. (CatE's own :meth:`train` loop does not consult the
        regime's per-axiom-type hooks.)
        """
        return self._training_regime

    @training_regime.setter
    def training_regime(self, regime):
        if regime is not None:
            from mowl.training.regime import TrainingRegime
            if not isinstance(regime, TrainingRegime):
                raise TypeError("training_regime must be a mowl.training.TrainingRegime "
                                "or None.")
            regime.attach(self)
        self._training_regime = regime

    # ----------------------------------------------------------------- module
    def init_module(self):
        """Create the :class:`~mowl.nn.graph.cate.CatEModule`."""
        lat = self.lattice
        module = CatEModule(len(lat), self.embed_dim, p=self.p)
        if lat.existential_to_id:
            keys = list(lat.existential_to_id.items())
            module.set_existentials([k[0] for k, _ in keys], [k[1] for k, _ in keys],
                                    [v for _, v in keys])
        self._module = module.to(self.device)

    @property
    def module(self):
        """The order-embedding module; ``module(rows, gci_name)`` scores rows of element
        ids (see :class:`~mowl.nn.graph.cate.CatEModule`)."""
        if self._module is None:
            self.init_module()
        return self._module

    @property
    def evaluation_model(self):
        return self.module

    # --------------------------------------------------------------- training
    def _edge_keys(self):
        if self._keys is None or self._keys.device != th.device(self.device):
            e = self.lattice.edges.to(self.device)
            self._keys = th.sort(e[:, 0] * len(self.lattice) + e[:, 1])[0]
        return self._keys

    def sample_negatives(self, pos):
        """Replace the superconcept of every morphism ``num_negs`` times by an element
        drawn uniformly from the lattice.

        :return: ``(neg_dst, mask)``: corrupted superconcepts of shape ``(b, num_negs)``
            and, if ``filter_negatives``, a mask that is ``False`` where the corruption
            is itself a morphism of the lattice or reflexive (otherwise ``None``).
        """
        n = len(self.lattice)
        neg_dst = th.randint(0, n, (len(pos), self.num_negs), device=pos.device)
        if not self.filter_negatives:
            return neg_dst, None
        keys = pos[:, :1] * n + neg_dst
        mask = ~th.isin(keys, self._edge_keys()) & (neg_dst != pos[:, :1])
        return neg_dst, mask

    def compute_loss(self, pos):
        """Loss of Eq. 7 for a batch of morphisms ``pos`` (shape ``(b, 2)``), plus the
        interaction loss of the reference code if :attr:`interaction_loss` is set."""
        module = self.module
        neg_dst, mask = self.sample_negatives(pos)
        pos_scores = module.order_violation(pos[:, 0], pos[:, 1])
        neg_scores = module.order_violation(
            pos[:, 0].unsqueeze(1).expand(-1, self.num_negs), neg_dst)
        loss = order_margin_loss(pos_scores, neg_scores, self.margin,
                                 negative_aggregation=self.negative_aggregation,
                                 neg_mask=mask)
        if self.interaction_loss:
            loss = loss + self._interaction_loss(pos[:, 0], pos_scores)
        return loss

    def _interaction_ids(self):
        """(head ids, existential ids) of the interaction loss: the lattice elements of
        the dataset's head evaluation classes and the elements :math:`\\exists r.p` for
        the dataset's evaluation property."""
        if self._interaction is not None and self._interaction[0].device == th.device(self.device):
            return self._interaction
        ds = self.dataset
        prop = getattr(ds, "get_evaluation_property", None)
        if prop is None:
            raise ValueError(
                "interaction_loss needs a dataset with get_evaluation_property() "
                "(the PPI and GDA datasets).")
        role = str(prop())
        heads, tails = ds.evaluation_classes
        lat = self.lattice
        head_ids = [lat.class_to_id[c] for c in heads.as_str if c in lat.class_to_id]
        pairs = [(lat.role_to_id[role], lat.class_to_id[t]) for t in tails.as_str
                 if t in lat.class_to_id]
        ex_ids = [lat.existential_to_id[k] for k in pairs if k in lat.existential_to_id]
        if not head_ids or not ex_ids:
            raise ValueError(
                "interaction_loss found no interaction elements for role "
                f"{role}: the model needs existentials='auto' (or a matching "
                "'existentials' argument).")
        self._interaction = (
            th.tensor(head_ids, dtype=th.long, device=self.device),
            th.tensor(ex_ids, dtype=th.long, device=self.device))
        return self._interaction

    def _interaction_loss(self, heads, pos_scores):
        """The PPI term of the reference training loop: mean order violation of the
        protein-headed morphisms of the batch plus a hinge on one corruption per row
        whose tail is drawn uniformly among the elements :math:`\\exists r.p`."""
        head_ids, ex_ids = self._interaction_ids()
        sel = th.isin(heads, head_ids)
        if not sel.any():
            return pos_scores.sum() * 0.0
        neg = ex_ids[th.randint(0, len(ex_ids), (int(sel.sum()),), device=ex_ids.device)]
        s_pos = pos_scores[sel]
        s_neg = self.module.order_violation(heads[sel], neg)
        return s_pos.mean() + th.relu(self.margin - s_neg).mean()

    def train(self, epochs=100, validate_every=None, verbose=False):
        """Train on all morphisms of the lattice.

        :param epochs: Number of passes over the morphisms.
        :type epochs: int
        :param validate_every: If given and the dataset has a validation ontology, compute
            the validation MRR every that many epochs, keep the best parameters and stop
            early after ``early_stopping_patience`` validations without improvement.
        :type validate_every: int, optional
        :param verbose: Print the loss of every epoch.
        :type verbose: bool
        :return: The mean training loss of every epoch.
        :rtype: list of float
        """
        module = self.module
        edges = self.lattice.edges.to(self.device)
        optimizer = th.optim.Adam(module.parameters(), lr=self.learning_rate,
                                  weight_decay=self.weight_decay)
        scheduler = None
        if self.cyclic_lr:
            scheduler = th.optim.lr_scheduler.CyclicLR(
                optimizer, base_lr=self.learning_rate / 10, max_lr=self.learning_rate,
                step_size_up=20, cycle_momentum=False)
        validate = validate_every is not None and self.dataset.validation is not None \
            and len(self.validation_queries()) > 0
        best, bad, history = -1.0, 0, []
        self.validation_history = []
        for epoch in range(epochs):
            module.train()
            perm = th.randperm(len(edges), device=self.device)
            total, batches = 0.0, 0
            for start in range(0, len(edges), self.batch_size):
                pos = edges[perm[start:start + self.batch_size]]
                optimizer.zero_grad()
                loss = self.compute_loss(pos)
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                total += loss.item()
                batches += 1
            history.append(total / max(batches, 1))
            if verbose:
                print(f"Epoch {epoch + 1}: loss {history[-1]:.6f}")
            if validate and ((epoch + 1) % validate_every == 0 or epoch + 1 == epochs):
                mrr = self.validation_mrr()
                self.validation_history.append((epoch + 1, mrr))
                if mrr > best:
                    best, bad = mrr, 0
                    th.save(module.state_dict(), self.model_filepath)
                else:
                    bad += 1
                    if self.early_stopping_patience is not None and \
                            bad >= self.early_stopping_patience:
                        break
        if validate:
            self.load_best_model()
        else:
            th.save(module.state_dict(), self.model_filepath)
        module.eval()
        return history

    # ------------------------------------------------------------- validation
    def validation_queries(self):
        """Validation queries ``(head_id, true_id, candidate_ids)``: the released
        validation subsumptions of the dataset if it defines
        ``cate_validation_query_pairs`` (the CatE completion benchmarks), otherwise the
        named subsumptions :math:`C \\sqsubseteq D` and memberships :math:`D(a)` of
        the validation ontology (candidates: all named classes) and
        :math:`C \\sqsubseteq \\exists r.D` (candidates: the elements
        :math:`\\exists r.D'`). Axioms with elements outside the lattice are skipped."""
        if self._validation is not None:
            return self._validation
        lat = self.lattice
        bounds = {lat.top_id, lat.bot_id}
        classes = th.tensor(sorted(set(lat.class_to_id.values()) - bounds), dtype=th.long)
        pairs = getattr(self.dataset, "cate_validation_query_pairs", None)
        if pairs:
            out = []
            for head_iri, tail_iri in pairs:
                head_iri, tail_iri = str(head_iri), str(tail_iri)
                if head_iri in lat.class_to_id and tail_iri in lat.class_to_id:
                    out.append((lat.class_to_id[head_iri], lat.class_to_id[tail_iri],
                                classes))
            self._validation = out
            return out
        by_role = {}
        for (r, _), node in lat.existential_to_id.items():
            by_role.setdefault(r, []).append(node)
        by_role = {r: th.tensor(sorted(v), dtype=th.long) for r, v in by_role.items()}
        out = []
        onto = self.dataset.validation
        if onto is not None:
            from org.semanticweb.owlapi.model import AxiomType
            from org.semanticweb.owlapi.model import ClassExpressionType as CT
            for ax in onto.getAxioms():
                try:
                    if ax.getAxiomType() == AxiomType.SUBCLASS_OF:
                        sub, sup = ax.getSubClass(), ax.getSuperClass()
                        if sub.isAnonymous():
                            continue
                        h = lat.node_id(sub)
                        if not sup.isAnonymous():
                            out.append((h, lat.node_id(sup), classes))
                        elif sup.getClassExpressionType() == CT.OBJECT_SOME_VALUES_FROM \
                                and not sup.getFiller().isAnonymous():
                            r = lat.role_to_id.get(str(sup.getProperty().toStringID()))
                            if r in by_role:
                                out.append((h, lat.node_id(sup), by_role[r]))
                    elif ax.getAxiomType() == AxiomType.CLASS_ASSERTION:
                        cls = ax.getClassExpression()
                        if cls.isAnonymous():
                            continue
                        a = lat.individual_id(ax.getIndividual().toStringID())
                        out.append((a, lat.node_id(cls), classes))
                except UnseenConceptError:
                    continue
        self._validation = out
        return out

    @th.no_grad()
    def validation_mrr(self):
        """Raw mean reciprocal rank of the validation queries (ties: average rank)."""
        module = self.module
        module.eval()
        rr = []
        for head, true, cands in self.validation_queries():
            cands = cands.to(self.device)
            scores = module.order_violation(th.full_like(cands, head), cands)
            mask = cands != head
            scores, cands = scores[mask], cands[mask]
            true_score = module.order_violation(th.tensor([head], device=self.device),
                                                th.tensor([true], device=self.device))[0]
            better = (scores < true_score).sum().item()
            ties = (scores == true_score).sum().item()
            rank = 1 + better + max(ties - 1, 0) / 2
            rr.append(1.0 / rank)
        module.train()
        return sum(rr) / len(rr) if rr else 0.0

    # ---------------------------------------------------------------- scoring
    def _tensor(self, ids):
        return th.as_tensor(ids, dtype=th.long, device=self.device).reshape(-1)

    @th.no_grad()
    def score_edges(self, src_ids, dst_ids):
        """:math:`s(A, B)` (Eq. 6) for element ids; lower is better."""
        return self.module.order_violation(self._tensor(src_ids), self._tensor(dst_ids))

    def score_subsumption(self, sub_ids, sup_ids):
        """Score of :math:`C \\sqsubseteq D` for element ids (e.g. from
        :attr:`class_index_dict`); lower is better."""
        return self.score_edges(sub_ids, sup_ids)

    def score_membership(self, individual_ids, class_ids):
        """Score of :math:`C(a)` as :math:`s(\\{a\\}, C)` for ids from
        :attr:`individual_index_dict` and :attr:`class_index_dict`; lower is better."""
        return self.score_edges(individual_ids, class_ids)

    @th.no_grad()
    def score_existential(self, sub_ids, role_ids, filler_ids):
        """Score of :math:`C \\sqsubseteq \\exists r.D` as :math:`s(C, \\exists r.D)`.

        :raises UnseenConceptError: If some :math:`\\exists r.D` is not an element.
        """
        rows = th.stack([self._tensor(sub_ids), self._tensor(role_ids),
                         self._tensor(filler_ids)], dim=1)
        return self.module(rows, "gci2")

    def score(self, axiom):
        """Score of a ``SubClassOf``, ``ClassAssertion`` or ``ObjectPropertyAssertion``
        axiom (:math:`r(a, b)` as :math:`\\{a\\} \\sqsubseteq \\exists r.\\{b\\}`, as the
        projector encodes it). Lower is better; 0 means the order holds exactly.

        :raises UnseenConceptError: If a concept description of the axiom is not an
            element of the lattice.
        :rtype: float
        """
        from org.semanticweb.owlapi.model import AxiomType
        lat = self.lattice
        t = axiom.getAxiomType()
        if t == AxiomType.SUBCLASS_OF:
            pair = (lat.node_id(axiom.getSubClass()), lat.node_id(axiom.getSuperClass()))
        elif t == AxiomType.CLASS_ASSERTION:
            pair = (lat.individual_id(axiom.getIndividual().toStringID()),
                    lat.node_id(axiom.getClassExpression()))
        elif t == AxiomType.OBJECT_PROPERTY_ASSERTION:
            from mowl.owlapi import OWLAPIAdapter
            ad = OWLAPIAdapter()
            obj = ad.create_class(str(axiom.getObject().toStringID()))
            ex = ad.create_object_some_values_from(axiom.getProperty(), obj)
            pair = (lat.individual_id(axiom.getSubject().toStringID()), lat.node_id(ex))
        else:
            raise TypeError(f"CatE.score does not score {t} axioms.")
        return self.score_edges([pair[0]], [pair[1]]).item()

    # ---------------------------------------------------------- persistence
    def load_best_model(self):
        """Load the parameters saved by :meth:`train`."""
        self.module.load_state_dict(th.load(self.model_filepath, weights_only=True))
        self.module.eval()

    def from_pretrained(self, file_name):
        """Load parameters from ``file_name``."""
        if not isinstance(file_name, str):
            raise TypeError("Parameter file_name must be a string.")
        if not os.path.exists(file_name):
            raise FileNotFoundError("Pretrained model path does not exist")
        self.module.load_state_dict(th.load(file_name, weights_only=True))

    # -------------------------------------------------------------- embeddings
    def _vectors(self, mapping):
        w = self.module.node_embedding.weight.detach().cpu().numpy()
        return {name: w[i] for name, i in mapping.items()}

    @property
    def class_embeddings(self):
        """Vector of every named class.

        :rtype: dict
        """
        return self._vectors(self.lattice.class_to_id)

    @property
    def individual_embeddings(self):
        """Vector of every :math:`\\{a\\}`.

        :rtype: dict
        """
        return self._vectors(self.lattice.individual_to_id)

    @property
    def node_embeddings(self):
        """Vector of every lattice element, by element name.

        :rtype: dict
        """
        return self._vectors(self.lattice.name_to_id)


# ---------------------------------------------------------------------------------------
# Benchmark runner support
# ---------------------------------------------------------------------------------------

def _register_benchmark_family():
    """Register a :class:`mowl.benchmarks.runner.ModelFamily` for :class:`CatE`, so that
    ``python -m mowl.benchmarks run <benchmark> --method cate`` can build and train it
    (the built-in families handle only ``EmbeddingELModel`` subclasses)."""
    try:
        from mowl.benchmarks import runner
    except ImportError:  # pragma: no cover
        return
    if any(getattr(f, "name", "") == "cate" for f in runner._FAMILIES):
        return

    class CatEFamily(runner.ModelFamily):
        name = "cate"

        def matches(self, cls):
            return isinstance(cls, type) and issubclass(cls, CatE)

        def build(self, cls, dataset, model_kwargs, device, model_filepath):
            kwargs = dict(model_kwargs)
            kwargs.setdefault("device", device)
            kwargs.setdefault("model_filepath", model_filepath)
            return cls(dataset, **kwargs)

        def train(self, model, epochs, protocol, validate_every=None, select_best=True):
            validate = select_best and model.dataset.validation is not None
            if validate and validate_every is None:
                validate_every = max(1, epochs // 10)
            model.train(epochs=epochs, validate_every=validate_every if validate else None)
            validated = validate and bool(getattr(model, "validation_history", []))
            return {"selected": "best validation MRR" if validated else "last epoch",
                    "validation": getattr(model, "validation_history", []),
                    "lattice": {k: v for k, v in model.lattice.stats.items()
                                if isinstance(v, (int, float))}}

    runner.register_family(CatEFamily(), first=True)


_register_benchmark_family()
