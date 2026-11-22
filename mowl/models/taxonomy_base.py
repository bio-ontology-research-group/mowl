r"""Shared model base for order and hierarchy embeddings of the class taxonomy."""
import numpy as np

from mowl.base_models.elmodel import EmbeddingELModel
from mowl.evaluation.model_scoring import pair_costs
from mowl.training.taxonomy import named_pair_rows


class TaxonomyEmbeddingModel(EmbeddingELModel):
    r"""An :class:`~mowl.base_models.EmbeddingELModel` that trains on the subclass axioms
    between **named classes** (:math:`C \sqsubseteq D`, normal form ``gci0``) and nothing
    else.

    The order and lattice embeddings of this family have a meaning for the order relation
    only. The normal forms with conjunction, existential restrictions, bottom, assertions
    and role axioms are dropped from the training data (no error, they are not part of
    these methods); ``gci0`` rows that mention auxiliary classes introduced by the
    normalisation are dropped too, so what is trained is the taxonomy the ontology states
    (and what the normaliser makes of ``C ⊑ D ⊓ E``, which are the two axioms ``C ⊑ D`` and
    ``C ⊑ E``). Scoring supports ``gci0`` only (``scored_forms``).

    Training uses the generic loop of :class:`~mowl.base_models.EmbeddingELModel` (one
    full-batch step per epoch) with negatives obtained by corrupting the stated pairs:
    ``neg_columns`` lists the columns to corrupt, one set of negatives per column; the
    default corrupts the subclass and the superclass in turn, as the order-embedding paper
    does for hypernymy ("replacing one of the two concepts with a randomly chosen
    concept"). A training regime attached to ``training_regime`` can replace any of it.

    :param neg_columns: Columns (0: subclass, 1: superclass) to corrupt.
    """

    axiom_scoring = "module"
    scored_forms = ("gci0",)

    def __init__(self, dataset, embed_dim, batch_size, model_filepath=None, device="cpu",
                 learning_rate=0.001, neg_sampling_gcis=None, neg_columns=(0, 1)):
        super().__init__(dataset, embed_dim, batch_size, extended=False,
                         model_filepath=model_filepath, device=device,
                         learning_rate=learning_rate, neg_sampling_gcis=neg_sampling_gcis)
        self.neg_columns = tuple(neg_columns)
        self._DEFAULT_NEG_SAMPLING_CONFIG = {
            "gci0": {"index_pool": "classes", "corrupt_column": list(self.neg_columns)}}

    # -- data: gci0 between named classes only -------------------------------------------
    def _named_gci0(self, datasets):
        if not datasets or "gci0" not in datasets:
            return {}
        ds = datasets["gci0"]
        n = len(self.named_class_index_dict)
        if ds._data.dim() == 2:
            # The normaliser returns axioms in the iteration order of a hash set, which
            # changes between processes; negatives are drawn per row position, so the rows
            # are sorted to make a seeded run reproducible.
            ds._data = named_pair_rows(ds._data, n)
        return {"gci0": ds}

    def _load_datasets(self):
        if self._datasets_loaded:
            return
        super()._load_datasets()
        self._training_datasets = self._named_gci0(self._training_datasets)
        if self._validation_datasets:
            self._validation_datasets = self._named_gci0(self._validation_datasets)
        if self._testing_datasets:
            self._testing_datasets = self._named_gci0(self._testing_datasets)

    def train(self, epochs, validate_every=1, epoch_callback=None):
        if len(self.training_datasets.get("gci0", ())) == 0:
            raise ValueError(
                f"{type(self).__name__} trains on subclass axioms between named classes, and "
                f"the training ontology has none.")
        super().train(epochs, validate_every=validate_every, epoch_callback=epoch_callback)
        self.after_training()

    def after_training(self):
        """Hook run after the last epoch."""

    def get_optimizer(self):
        optimizer = super().get_optimizer()
        project = getattr(self.module, "project", None)
        if project is not None:
            # The boxes live in the unit cube: project after every step (paper, Sec. 4.3).
            original = optimizer.step

            def step(*args, **kwargs):
                out = original(*args, **kwargs)
                project()
                return out
            optimizer.step = step
        return optimizer

    @property
    def object_property_embeddings(self):
        return dict()

    @property
    def individual_embeddings(self):
        return dict()

    def get_embeddings(self):
        self.init_module()
        self.load_best_model()
        return self.class_embeddings, dict(), None

    # -- helpers -------------------------------------------------------------------------
    def subsumption_costs(self, sub, sup):
        """Cost (lower: more plausible) of ``sub ⊑ sup`` for IRIs or indices."""
        return pair_costs(self, sub, sup).cpu().numpy()  # one cost per axiom

    def descendant_fractions(self):
        """For each named class the number of its descendants (itself included) in the
        training taxonomy, divided by the number of named classes (the unary marginal of
        the box lattice paper, Section 5.2)."""
        n = len(self.named_class_index_dict)
        children = [[] for _ in range(n)]
        for sub, sup in self.training_datasets["gci0"].data.cpu().numpy():
            if sub != sup:
                children[sup].append(sub)
        memo = {}

        def descendants(root):
            seen, stack = {root}, [root]
            while stack:
                for c in children[stack.pop()]:
                    if c not in seen:
                        seen.add(c)
                        stack.append(c)
            return seen
        counts = np.array([len(memo.setdefault(i, descendants(i))) for i in range(n)])
        return counts / n
