"""What HiT models share, whatever trains them: the taxonomy, the verbalisation, the training
triplets, the scoring and the ranking. :class:`mowl.models.HiT` trains with the regimes of
:mod:`mowl.training`; the direct implementation in :mod:`mowl.baselines.models.hit` keeps its
own loop for the comparison in ``tests/baselines``."""

import logging
import random

import torch as th

from mowl.base_models.model import Model
from mowl.owlapi.defaults import TOP, BOT

logger = logging.getLogger(__name__)

#: Grid of the centripetal score weight :math:`\lambda` searched on validation data (OnT,
#: Section 5.1: :math:`\lambda \in \{0, 0.1, \dots, 1\}`).
DEFAULT_CENTRI_GRID = tuple(i / 10 for i in range(11))


class HiTBase(Model):
    """The data and the scoring of HiT models (see :class:`mowl.models.HiT`). A subclass makes
    ``init_module``, the four methods of the module of its side and trains."""

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
        from mowl.representation.library.text_encoders import as_encoder
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
        from mowl.evaluation.text_ranking import encode_batched
        return encode_batched(self._embed_points, texts, batch_size or self.encode_batch_size,
                              self.module, lambda: self.module.dim, self.device)

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
        from mowl.evaluation.text_ranking import encode_cached
        return encode_cached(self.embed_texts, [self.text(e) for e in exprs],
                             self._cache.setdefault("texts", {}))

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
        if gci_name not in (self.scored_forms or ()):
            from mowl.evaluation.model_scoring import UnsupportedAxiomForm
            raise UnsupportedAxiomForm(
                f"{type(self).__name__} has no score for {gci_name!r} axioms.")
        lam = centri_weight if centri_weight is not None else (self.centri_weight or 0.0)
        mat = self._class_matrix()
        cpu = rows.cpu()

        from mowl.evaluation.text_ranking import embed_unique_pairs
        inv_c = getattr(self, "_inv_classes", None)
        if inv_c is None:
            inv_c = self._inv_classes = {v: k for k, v in self.class_index_dict.items()}
        inv_r = {v: k for k, v in self.object_property_index_dict.items()}

        def existential(roles, fillers):
            return embed_unique_pairs(
                lambda r, d: self._existential_embeddings(r, d, inv_r, inv_c), roles, fillers)

        if gci_name == "gci0":
            sub, sup = mat[cpu[:, 0].to(mat.device)], mat[cpu[:, 1].to(mat.device)]
        elif gci_name == "gci2":
            sub, sup = mat[cpu[:, 0].to(mat.device)], existential(cpu[:, 1], cpu[:, 2])
        else:
            sub, sup = existential(cpu[:, 0], cpu[:, 1]), mat[cpu[:, 2].to(mat.device)]
        dtype = self.score_dtype
        return self._cost(sub.to(dtype), sup.to(dtype), lam).to(rows.device)

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
        lam = centri_weight if centri_weight is not None else (self.centri_weight or 0.0)
        query, candidates_are_children = self._query(gci_name, rows, target_col)
        from mowl.evaluation.text_ranking import candidate_scores
        cand = self._class_matrix()[candidates.to(self._class_matrix().device)]
        return candidate_scores(
            lambda q, c: self._ranking_scores(q, c, candidates_are_children, lam),
            query, cand, chunk_size)

    def validate(self, rows=None, grid=None):
        """Rank the superclass of every validation subsumption :math:`C \\sqsubseteq D`
        among the candidate classes for every :math:`\\lambda` of the grid.

        :return: ``(best MRR, best lambda)``; with a fixed :attr:`centri_weight`, only
            that value is evaluated.
        """
        from mowl.evaluation.text_ranking import best_on_grid
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
        return best_on_grid(
            lambda lam, s, e: self.score_candidates("gci0", rows[s:e], 1, cands, lam),
            targets, grid)

    def score(self, axiom):
        """Score (Eq. 6, higher is more likely) of a ``SubClassOf`` axiom with arbitrary
        :math:`\\mathcal{EL}` class expressions (verbalised), using :attr:`centri_weight`."""
        from org.semanticweb.owlapi.model import AxiomType
        if axiom.getAxiomType() != AxiomType.SUBCLASS_OF:
            raise TypeError("Only SubClassOf axioms can be scored.")
        x = self.embed([axiom.getSubClass(), axiom.getSuperClass()])
        lam = self.centri_weight or 0.0
        return self._score_pair(x, lam)
    # -- what depends on the module of a side --------------------------------------------
    def _embed_points(self, texts):
        """Points of the ball for a list of texts."""
        raise NotImplementedError

    def _cost(self, sub, sup, lam):
        """:math:`d_c(x, y) + \\lambda(\\lVert y\\rVert_c - \\lVert x\\rVert_c)` of two
        tensors of points of the ball, one value per row."""
        raise NotImplementedError

    def _ranking_scores(self, query, candidates, candidates_are_children, lam):
        """Scores (higher is more likely) of the candidates against every query row."""
        raise NotImplementedError

    def _score_pair(self, x, lam):
        """Score of the subsumption between the two embeddings in ``x`` (higher is more
        likely)."""
        raise NotImplementedError
