"""HiT: Hierarchy Transformer encoders (He, Yuan, Chen, Horrocks, NeurIPS 2024)."""

import logging
import math
import random

import numpy as np
import torch as th

from mowl.base_models.model import Model
from mowl.owlapi.defaults import TOP, BOT

logger = logging.getLogger(__name__)

#: Grid of the centripetal score weight :math:`\lambda` searched on validation data (OnT,
#: Section 5.1: :math:`\lambda \in \{0, 0.1, \dots, 1\}`).
DEFAULT_CENTRI_GRID = tuple(i / 10 for i in range(11))


class HiT(Model):
    """Hierarchy Transformer encoder (HiT) [he2024hit]_.

    A text encoder (a sentence-transformers model with mean pooling and no normalisation
    layer) is re-trained so that the embeddings of class labels, read as points of the
    Poincaré ball of radius :math:`\\sqrt{d}` (curvature :math:`c = 1/d`), are clustered
    by the hyperbolic clustering loss and ordered by the hyperbolic centripetal loss
    (Section 3, Eqs. 3-5). A subsumption :math:`C \\sqsubseteq D` is scored by
    :math:`-(d_c(x_C, x_D) + \\lambda(\\lVert x_D\\rVert_c - \\lVert x_C\\rVert_c))` (Eq. 6).

    Training data: the atomic subsumptions :math:`A \\sqsubseteq B` of
    ``dataset.ontology`` (``taxonomy="asserted"``, the edges of the training hierarchy) or
    the direct subsumptions inferred by ELK (``taxonomy="direct"``, as the paper builds
    hierarchies from OWL ontologies, Appendix A). For every positive :math:`(e, e^+)`,
    ``num_negatives`` negative parents are sampled (Section 4.1): random classes
    :math:`e^-` with :math:`e \\not\\sqsubseteq e^-` under the closed-world assumption
    (not in the reflexive-transitive closure of the training edges), or, with
    ``hard_negatives=True``, siblings of :math:`e` first, completed with random ones.

    Optimisation as in the paper (Appendix B): AdamW, learning rate :math:`10^{-5}`, 500
    linear warm-up steps followed by linear decay, batch size 256. After every epoch, if
    ``dataset.validation`` has atomic subsumptions, the weight :math:`\\lambda` is chosen
    on them and the parameters with the best validation MRR are kept (the paper selects
    the model each epoch on validation F1 with a tuned threshold; mOWL uses the ranking
    MRR, as OnT does for HiT, Yang et al. 2025, Section 5.1).

    Class labels are ``rdfs:label`` annotations (fallback: IRI fragment), see
    :mod:`mowl.text.verbalise`. Complex class expressions (when scoring axioms that are
    not atomic) are verbalised with the OnT rules; HiT itself is defined for named
    entities only.

    :param dataset: A :class:`mowl.datasets.Dataset`.
    :param encoder: Hugging Face model name (loaded with sentence-transformers, an optional
        dependency), a ``SentenceTransformer`` object, or any :class:`torch.nn.Module`
        mapping a list of strings to a ``(n, d)`` tensor (for example
        :class:`mowl.nn.lm.encoders.BagOfWordsEncoder`).
    :param verbaliser: A :class:`mowl.text.Verbaliser`; default: one built from
        ``dataset.ontology`` with ``labels``, ``verbalisation_style`` and
        ``label_fallback``.
    :param labels: ``{iri: label}`` overriding the ontology's labels.
    :param verbalisation_style: ``"paper"``, ``"release"`` or ``"deeponto"``.
    :param label_fallback: ``"fragment"``, ``"camel_case"`` or ``None``.
    :param clustering_margin: :math:`\\alpha` (Eq. 3); the paper uses 5.0.
    :param centripetal_margin: :math:`\\beta` (Eq. 4); the paper uses 0.1.
    :param clustering_weight: Weight of the clustering loss.
    :param centripetal_weight: Weight of the centripetal loss.
    :param curvature: Curvature parameter :math:`c` (default :math:`1/d`).
    :param learning_rate: AdamW learning rate.
    :param weight_decay: AdamW weight decay (0, the Hugging Face default).
    :param warmup_steps: Linear warm-up steps (then linear decay to 0).
    :param batch_size: Training triplets per step.
    :param num_negatives: Negative parents per positive (10 in the paper).
    :param hard_negatives: Sample siblings as negatives first.
    :param taxonomy: ``"asserted"`` or ``"direct"`` (see above).
    :param centri_weight: Fixed :math:`\\lambda`; ``None`` selects it on validation data
        (0 if there is none).
    :param centri_weight_grid: Grid searched for :math:`\\lambda`.
    :param encode_batch_size: Batch size when encoding without gradients.
    :param score_dtype: Precision of the ranking scores, ``"float64"`` (default) or
        ``"float32"`` (as the reference code; much faster on GPUs with slow float64).
    :param seed: Seed of the negative sampling (default: Python's global generator).
    :param device: Torch device.
    :param model_filepath: Where the best parameters are saved.
    """

    #: Default margins of the method (subclasses change them).
    default_margins = (5.0, 0.1)

    axiom_scoring = "method"
    scored_forms = ("gci0",)

    def __init__(self, dataset, encoder="sentence-transformers/all-MiniLM-L12-v2",
                 verbaliser=None, labels=None, verbalisation_style="paper",
                 label_fallback="fragment", clustering_margin=None, centripetal_margin=None,
                 clustering_weight=1.0, centripetal_weight=1.0, curvature=None,
                 learning_rate=1e-5, weight_decay=0.0, warmup_steps=500, batch_size=256,
                 num_negatives=10, hard_negatives=False, taxonomy="asserted",
                 centri_weight=None, centri_weight_grid=DEFAULT_CENTRI_GRID,
                 encode_batch_size=256, score_dtype="float64", seed=None, device="cpu",
                 model_filepath=None):
        super().__init__(dataset, model_filepath=model_filepath)
        from mowl.text.verbalise import Verbaliser
        from mowl.nn.lm.encoders import as_encoder
        if taxonomy not in ("asserted", "direct"):
            raise ValueError("taxonomy must be 'asserted' or 'direct'.")
        self.device = device
        self.verbaliser = verbaliser if verbaliser is not None else Verbaliser(
            dataset.ontology, labels=labels, style=verbalisation_style,
            fallback=label_fallback)
        self.clustering_margin = (self.default_margins[0] if clustering_margin is None
                                  else clustering_margin)
        self.centripetal_margin = (self.default_margins[1] if centripetal_margin is None
                                   else centripetal_margin)
        self.clustering_weight = clustering_weight
        self.centripetal_weight = centripetal_weight
        self.curvature = curvature
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.batch_size = batch_size
        self.num_negatives = num_negatives
        self.hard_negatives = hard_negatives
        self.taxonomy = taxonomy
        self.centri_weight = centri_weight
        self._fixed_centri_weight = centri_weight is not None
        self.centri_weight_grid = tuple(centri_weight_grid)
        self.encode_batch_size = encode_batch_size
        if score_dtype not in ("float32", "float64"):
            raise ValueError("score_dtype must be 'float32' or 'float64'.")
        self.score_dtype = getattr(th, score_dtype)
        self.rng = random.Random(seed) if seed is not None else random
        self.encoder = as_encoder(encoder, device=device)
        self.module = self.init_module()
        self.module.to(device)
        self.history = []
        self.best_validation = None
        self._cache = {}
        self._training_regime = None

    @property
    def training_regime(self):
        """The named training design attached to the model
        (:class:`mowl.training.TrainingRegime`).

        HiT's training loop implements the reference design itself and does not
        consult regime hooks, so only the descriptive designs of
        :mod:`mowl.training.lm_reference` (``hit-code-training``,
        ``ont-code-training``) can be attached; attaching one records, in
        benchmark results, which published design the run follows.
        """
        return self._training_regime

    @training_regime.setter
    def training_regime(self, regime):
        from mowl.training.regime import TrainingRegime
        from mowl.training.lm_reference import LMCodeTraining
        if regime is not None and not isinstance(regime, TrainingRegime):
            raise TypeError("training_regime must be a mowl.training.TrainingRegime "
                            "instance or None.")
        if regime is not None and not isinstance(regime, LMCodeTraining):
            raise ValueError(
                f"{type(self).__name__}.train implements the reference training design "
                f"itself and does not consult regime hooks; attaching "
                f"{type(regime).__name__} would have no effect. Only the descriptive "
                f"designs of mowl.training.lm_reference can be attached.")
        self._training_regime = regime
        if regime is not None:
            regime.attach(self)

    # ------------------------------------------------------------------ set-up
    def init_module(self):
        from mowl.baselines.nn.lm.hit import HiTModule
        return HiTModule(self.encoder, curvature=self.curvature,
                         clustering_margin=self.clustering_margin,
                         centripetal_margin=self.centripetal_margin,
                         clustering_weight=self.clustering_weight,
                         centripetal_weight=self.centripetal_weight)

    @property
    def c(self):
        """Curvature parameter of the ball."""
        return self.module.c

    @property
    def candidate_iris(self):
        """IRIs of the classes ranked as candidates: ``dataset.candidate_classes`` if the
        dataset defines it, otherwise all classes except ``owl:Thing`` and
        ``owl:Nothing``."""
        cands = getattr(self.dataset, "candidate_classes", None)
        if cands is not None:
            return list(cands)
        return [c for c in self.dataset.classes.as_str if c not in (TOP, BOT)]

    def _named_iri(self, expr):
        if expr.isAnonymous():
            return None
        iri = str(expr.asOWLClass().toStringID())
        return None if iri in (TOP, BOT) else iri

    def taxonomy_edges(self, ontology=None):
        """Atomic subsumptions :math:`(A, B)` used for training, as IRI pairs."""
        from org.semanticweb.owlapi.model import AxiomType
        ontology = self.dataset.ontology if ontology is None else ontology
        edges = set()
        if self.taxonomy == "direct":
            from org.semanticweb.elk.owlapi import ElkReasonerFactory
            reasoner = ElkReasonerFactory().createReasoner(ontology)
            try:
                reasoner.precomputeInferences()
                for cls in ontology.getClassesInSignature():
                    a = self._named_iri(cls)
                    if a is None:
                        continue
                    for sup in reasoner.getSuperClasses(cls, True).getFlattened():
                        b = self._named_iri(sup)
                        if b is not None and b != a:
                            edges.add((a, b))
            finally:
                reasoner.dispose()
            return sorted(edges)
        for ax in ontology.getAxioms():
            t = ax.getAxiomType()
            if t == AxiomType.SUBCLASS_OF:
                pairs = [(ax.getSubClass(), ax.getSuperClass())]
            elif t == AxiomType.EQUIVALENT_CLASSES:
                ops = list(ax.getClassExpressionsAsList())
                pairs = [(x, y) for x in ops for y in ops if not x.equals(y)]
            else:
                continue
            for x, y in pairs:
                a, b = self._named_iri(x), self._named_iri(y)
                if a is not None and b is not None and a != b:
                    edges.add((a, b))
        return sorted(edges)

    # ------------------------------------------------------------------ text
    def text(self, expr):
        """Verbalisation of a class expression, entity or IRI."""
        return self.verbaliser.verbalise(expr)

    def embed_texts(self, texts, batch_size=None):
        """Encode texts without gradients (evaluation mode), shape ``(n, d)``."""
        batch_size = batch_size or self.encode_batch_size
        was_training = self.module.training
        self.module.eval()
        out = []
        with th.no_grad():
            for i in range(0, len(texts), batch_size):
                out.append(self.module.embed(texts[i:i + batch_size]).float())
        if was_training:
            self.module.train()
        if not out:
            return th.zeros((0, self.module.dim), device=self.device)
        return th.cat(out)

    def embed(self, expressions):
        """Embeddings of class expressions (OWL objects or IRIs), encoded from their
        verbalisations; works for classes not seen in training (inductive)."""
        return self.embed_texts([self.text(e) for e in expressions])

    def _class_matrix(self):
        """Embeddings of all classes of ``class_index_dict``, by index (cached until the
        next training step)."""
        if "classes" not in self._cache:
            iris = sorted(self.class_index_dict, key=self.class_index_dict.get)
            self._cache["classes"] = self.embed_texts([self.text(i) for i in iris])
        return self._cache["classes"]

    def _invalidate(self):
        self._cache = {}

    @property
    def class_embeddings(self):
        """``{iri: numpy vector}`` for every class of the dataset."""
        mat = self._class_matrix().cpu().numpy()
        return {iri: mat[i] for iri, i in self.class_index_dict.items()}

    @property
    def object_property_embeddings(self):
        raise NotImplementedError("HiT does not embed roles.")

    @property
    def individual_embeddings(self):
        raise NotImplementedError("HiT does not embed individuals.")

    # ------------------------------------------------------------------ training data
    @staticmethod
    def _closure(edges):
        """Strict ancestors of every node of the edge list (breadth-first search)."""
        parents = {}
        for a, b in edges:
            parents.setdefault(a, set()).add(b)
        closure = {}
        for start in parents:
            seen, frontier = set(), list(parents[start])
            while frontier:
                x = frontier.pop()
                if x in seen:
                    continue
                seen.add(x)
                frontier.extend(parents.get(x, ()))
            seen.discard(start)
            closure[start] = seen
        return closure

    def training_triplets(self):
        """``(child_text, parent_text, negative_text)`` triplets: ``num_negatives`` per
        training edge, sampled as described in the class docstring."""
        edges = self.taxonomy_edges()
        if not edges:
            raise ValueError("The training ontology has no atomic subsumptions.")
        closure = self._closure(edges)
        pool = self.candidate_iris
        children, parents = {}, {}
        for a, b in edges:
            children.setdefault(b, set()).add(a)
            parents.setdefault(a, set()).add(b)
        triplets = []
        for a, b in edges:
            excluded = closure.get(a, set()) | {a}
            negs = []
            if self.hard_negatives:
                sibs = {s for p in parents[a] for s in children.get(p, ())}
                sibs = sorted(sibs - excluded)
                self.rng.shuffle(sibs)
                negs.extend(sibs[:self.num_negatives])
            tries = 0
            while len(negs) < self.num_negatives and tries < 100 * self.num_negatives:
                n = self.rng.choice(pool)
                tries += 1
                if n not in excluded:
                    negs.append(n)
            for n in negs:
                triplets.append((self.text(a), self.text(b), self.text(n)))
        return triplets

    # ------------------------------------------------------------------ training
    def _optimiser(self, total_steps):
        params = [p for p in self.module.parameters() if p.requires_grad]
        opt = th.optim.AdamW(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        warm = max(0, int(self.warmup_steps))

        def schedule(step):
            # get_linear_schedule_with_warmup of Hugging Face transformers (the default
            # scheduler of the trainer used by the reference implementations)
            if step < warm:
                return step / max(1, warm)
            return max(0.0, (total_steps - step) / max(1, total_steps - warm))
        return opt, th.optim.lr_scheduler.LambdaLR(opt, schedule)

    def batch_loss(self, batch, negatives):
        """Mean training loss of one batch of triplets (HiT: :math:`L_{HiT}`, Eq. 5).

        :param batch: List of ``(child, parent, negative)`` texts.
        :param negatives: Embeddings of the negatives of the batch.
        :return: ``(loss, parts)`` with ``parts`` a dict of floats for logging.
        """
        child = self.module.embed([t[0] for t in batch])
        parent = self.module.embed([t[1] for t in batch])
        loss = self.module.hierarchy_loss(child, parent, negatives).mean()
        return loss, {"hierarchy": float(loss.detach())}

    def train(self, epochs=1, validate=True, verbose=False):
        """Train the encoder.

        :param epochs: Number of passes over the training triplets (20 in HiT, 1 in OnT).
        :param validate: Select :math:`\\lambda` and the best epoch on validation data
            (if ``dataset.validation`` has atomic subsumptions).
        :param verbose: Log the loss every 100 steps.
        :return: The mean training loss of the last epoch.
        """
        triplets = self.training_triplets()
        steps_per_epoch = math.ceil(len(triplets) / self.batch_size)
        opt, sched = self._optimiser(steps_per_epoch * epochs)
        has_valid = validate and self._validation_rows() is not None
        best = None
        mean_loss = float("nan")
        for epoch in range(epochs):
            self.module.train()
            self._invalidate()
            order = list(range(len(triplets)))
            self.rng.shuffle(order)
            total = 0.0
            for step, start in enumerate(range(0, len(order), self.batch_size)):
                batch = [triplets[i] for i in order[start:start + self.batch_size]]
                negatives = self.module.embed([t[2] for t in batch])
                loss, parts = self.batch_loss(batch, negatives)
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
                total += float(loss.detach())
                if verbose and step % 100 == 0:
                    logger.info("epoch %d step %d %s", epoch, step, parts)
            self._invalidate()
            mean_loss = total / max(1, steps_per_epoch)
            record = {"epoch": epoch, "loss": mean_loss}
            if has_valid:
                mrr, lam = self.validate()
                record.update({"valid_mrr": mrr, "centri_weight": lam})
                if best is None or mrr > best[0]:
                    best = (mrr, lam, epoch)
                    self._save_state()
            self.history.append(record)
            if verbose:
                logger.info("%s", record)
        if best is not None:
            self.load_best_model()
            if not self._fixed_centri_weight:
                self.centri_weight = best[1]
            self.best_validation = {"mrr": best[0], "centri_weight": best[1],
                                    "epoch": best[2]}
        elif self.centri_weight is None:
            self.centri_weight = 0.0
        return mean_loss

    def _save_state(self):
        th.save({k: v.detach().cpu() for k, v in self.module.state_dict().items()},
                self.model_filepath)

    def load_best_model(self):
        """Load the parameters saved at the best validation epoch."""
        self.module.load_state_dict(th.load(self.model_filepath, map_location=self.device))
        self._invalidate()

    def save(self, path):
        """Save the module parameters and :math:`\\lambda` to ``path``."""
        th.save({"state_dict": self.module.state_dict(), "centri_weight": self.centri_weight},
                path)

    def from_pretrained(self, file_name):
        """Load parameters written by :meth:`save`."""
        data = th.load(file_name, map_location=self.device)
        self.module.load_state_dict(data["state_dict"])
        self.centri_weight = data.get("centri_weight", self.centri_weight)
        self._invalidate()

    # ------------------------------------------------------------------ scoring
    def _validation_rows(self):
        if self.dataset.validation is None:
            return None
        from mowl.evaluation.axiom_ranking import el_rows
        rows, _ = el_rows(self, self.dataset.validation)
        gci0 = rows.get("gci0")
        return gci0 if gci0 is not None and len(gci0) else None

    def _query(self, gci_name, rows, target_col):
        """Embeddings of the fixed side of every query row and whether the candidates are
        subclasses (``True``) or superclasses (``False``)."""
        from mowl.owlapi import OWLAPIAdapter
        mat = self._class_matrix()
        inv_c = getattr(self, "_inv_classes", None)
        if inv_c is None:
            inv_c = self._inv_classes = {v: k for k, v in self.class_index_dict.items()}
        inv_r = {v: k for k, v in self.object_property_index_dict.items()}
        rows = rows.cpu()
        if gci_name == "gci0":
            fixed = rows[:, 1 - target_col]
            return mat[fixed.to(mat.device)], target_col == 0
        adapter = OWLAPIAdapter()
        if gci_name == "gci1" and target_col == 2:
            exprs = [adapter.create_object_intersection_of(
                adapter.create_class(inv_c[int(a)]), adapter.create_class(inv_c[int(b)]))
                for a, b in rows[:, :2].tolist()]
            return self._embed_cached(exprs), False
        if gci_name == "gci2" and target_col == 0:
            return self._existential_embeddings(rows[:, 1], rows[:, 2], inv_r, inv_c), True
        if gci_name == "gci3" and target_col == 2:
            return self._existential_embeddings(rows[:, 0], rows[:, 1], inv_r, inv_c), False
        raise NotImplementedError(f"Ranking column {target_col} of {gci_name} is not "
                                  f"supported.")

    def _embed_cached(self, exprs):
        texts = [self.text(e) for e in exprs]
        cache = self._cache.setdefault("texts", {})
        missing = sorted({t for t in texts if t not in cache})
        if missing:
            emb = self.embed_texts(missing)
            for t, e in zip(missing, emb):
                cache[t] = e
        return th.stack([cache[t] for t in texts])

    def _existential_embeddings(self, roles, fillers, inv_r, inv_c):
        from mowl.owlapi import OWLAPIAdapter
        adapter = OWLAPIAdapter()
        exprs = [adapter.create_object_some_values_from(
            adapter.create_object_property(inv_r[int(r)]), adapter.create_class(inv_c[int(d)]))
            for r, d in zip(roles.tolist(), fillers.tolist())]
        return self._embed_cached(exprs)

    @th.no_grad()
    def axiom_costs(self, gci_name, rows, centri_weight=None):
        """Cost of every candidate axiom in ``rows`` (model indices), lower is more
        plausible: :math:`d_c(x, y) + \\lambda(\\lVert y\\rVert_c - \\lVert x\\rVert_c)`
        for the embedding :math:`x` of the subsumed and :math:`y` of the subsuming
        expression, the negative of the score of :meth:`score_candidates`. ``gci0`` rows
        are ``[C, D]``; ``gci2`` rows ``[C, r, D]`` have :math:`y = \\exists r.D` and
        ``gci3`` rows ``[r, C, D]`` have :math:`x = \\exists r.C` (models that verbalise
        existential restrictions, i.e. OnT)."""
        from mowl.nn.lm import hyperbolic as hb
        if gci_name not in (self.scored_forms or ()):
            from mowl.evaluation.model_scoring import UnsupportedAxiomForm
            raise UnsupportedAxiomForm(
                f"{type(self).__name__} has no score for {gci_name!r} axioms.")
        lam = centri_weight if centri_weight is not None else (self.centri_weight or 0.0)
        mat = self._class_matrix()
        cpu = rows.cpu()

        def existential(roles, fillers):
            inv_c = getattr(self, "_inv_classes", None)
            if inv_c is None:
                inv_c = self._inv_classes = {v: k for k, v in self.class_index_dict.items()}
            inv_r = {v: k for k, v in self.object_property_index_dict.items()}
            pairs, inverse = th.unique(th.stack([roles, fillers], dim=1), dim=0,
                                       return_inverse=True)
            unique = self._existential_embeddings(pairs[:, 0], pairs[:, 1], inv_r, inv_c)
            return unique[inverse.to(unique.device)]

        if gci_name == "gci0":
            sub, sup = mat[cpu[:, 0].to(mat.device)], mat[cpu[:, 1].to(mat.device)]
        elif gci_name == "gci2":
            sub, sup = mat[cpu[:, 0].to(mat.device)], existential(cpu[:, 1], cpu[:, 2])
        else:
            sub, sup = existential(cpu[:, 0], cpu[:, 1]), mat[cpu[:, 2].to(mat.device)]
        dtype = self.score_dtype
        sub, sup = sub.to(dtype), sup.to(dtype)
        c = self.c
        cost = hb.distance(sub, sup, c) + lam * (hb.hyperbolic_norm(sup, c)
                                                 - hb.hyperbolic_norm(sub, c))
        return cost.to(rows.device)

    def score_candidates(self, gci_name, rows, target_col, candidates, centri_weight=None,
                         chunk_size=4096):
        """Scores of the candidate axioms obtained by replacing column ``target_col`` of
        every row with every candidate class (higher is more likely).

        :param gci_name: ``"gci0"`` (``[C, D]``), ``"gci1"`` (``[C, D, E]``, ranking
            ``E``), ``"gci2"`` (``[C, r, D]``, ranking ``C``) or ``"gci3"``
            (``[r, C, D]``, ranking ``D``).
        :param rows: ``LongTensor`` in mOWL's column layout (model indices).
        :param candidates: ``LongTensor`` of candidate class indices.
        :param centri_weight: :math:`\\lambda` (default: :attr:`centri_weight`, or 0).
        :return: ``(len(rows), len(candidates))`` tensor (of :attr:`score_dtype`).
        """
        from mowl.nn.lm import hyperbolic as hb
        lam = centri_weight if centri_weight is not None else (self.centri_weight or 0.0)
        query, candidates_are_children = self._query(gci_name, rows, target_col)
        cand = self._class_matrix()[candidates.to(self._class_matrix().device)]
        c = self.c
        dtype = self.score_dtype
        q_norm = hb.hyperbolic_norm(query.to(dtype), c).unsqueeze(1)
        out = []
        for s in range(0, len(cand), chunk_size):
            cc = cand[s:s + chunk_size]
            dist = hb.pairwise_distance(query, cc, c, dtype=dtype)
            c_norm = hb.hyperbolic_norm(cc.to(dtype), c).unsqueeze(0)
            if candidates_are_children:
                out.append(-(dist + lam * (q_norm - c_norm)))
            else:
                out.append(-(dist + lam * (c_norm - q_norm)))
        return th.cat(out, dim=1)

    def validate(self, rows=None, grid=None):
        """Rank the superclass of every validation subsumption :math:`C \\sqsubseteq D`
        among the candidate classes for every :math:`\\lambda` of the grid.

        :return: ``(best MRR, best lambda)``; with a fixed :attr:`centri_weight`, only
            that value is evaluated.
        """
        from mowl.evaluation.ont_ranking import ordinal_ranks_by_counting
        rows = self._validation_rows() if rows is None else rows
        if rows is None:
            return float("nan"), self.centri_weight
        cid = self.class_index_dict
        cands = th.tensor(sorted({cid[i] for i in self.candidate_iris if i in cid}))
        pos = {int(c): i for i, c in enumerate(cands.tolist())}
        rows = rows[[int(r[1]) in pos for r in rows.tolist()]]
        targets = th.tensor([pos[int(t)] for t in rows[:, 1].tolist()])
        grid = ((self.centri_weight,) if self._fixed_centri_weight
                else tuple(grid or self.centri_weight_grid))
        best = (-1.0, grid[0])
        for lam in grid:
            ranks = []
            for s in range(0, len(rows), 256):
                scores = self.score_candidates("gci0", rows[s:s + 256], 1, cands, lam)
                ranks.append(ordinal_ranks_by_counting(-scores, targets[s:s + 256]).cpu())
            mrr = float(np.mean(1.0 / th.cat(ranks).double().numpy()))
            if mrr > best[0]:
                best = (mrr, lam)
        return best

    def score(self, axiom):
        """Score (Eq. 6, higher is more likely) of a ``SubClassOf`` axiom with arbitrary
        :math:`\\mathcal{EL}` class expressions (verbalised), using :attr:`centri_weight`."""
        from org.semanticweb.owlapi.model import AxiomType
        if axiom.getAxiomType() != AxiomType.SUBCLASS_OF:
            raise TypeError("Only SubClassOf axioms can be scored.")
        x = self.embed([axiom.getSubClass(), axiom.getSuperClass()])
        lam = self.centri_weight or 0.0
        return float(self.module.score(x[0], x[1], lam))
