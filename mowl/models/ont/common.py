"""What OnT models share, whatever trains them: the verbalised axioms, the sampling of the
triplets and the scoring of existential restrictions through the role transformations.
:class:`mowl.models.OnT` trains with the regimes of :mod:`mowl.training`; the direct
implementation in :mod:`mowl.baselines.models.ont` keeps its own loop for the comparison in
``tests/baselines``."""

import logging

import torch as th

from mowl.models.hit.common import HiTBase, DEFAULT_CENTRI_GRID

logger = logging.getLogger(__name__)


class OnTBase(HiTBase):
    """The data and the scoring of OnT models (see :class:`mowl.models.OnT`)."""

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
