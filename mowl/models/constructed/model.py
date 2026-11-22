"""A model that is constructed, not trained."""

import logging

from mowl.base_models.elmodel import EmbeddingELModel

logger = logging.getLogger(__name__)


class ConstructedModel(EmbeddingELModel):
    """A model of the training ontology that is *constructed* (by completion, by a solver,
    in closed form) and read as regions: a source, a reading and the exact measure
    (:mod:`mowl.representation.construction`).

    Either name a constructed method (a :class:`~mowl.representation.construction.Recipe`
    registered with ``trained_on="construction"``) or swap its parts:

    .. testcode::

       from mowl.datasets.builtin import FamilyDataset
       from mowl.models.constructed import ConstructedModel

       dataset = FamilyDataset()
       model = ConstructedModel(dataset, method="interval_union")
       cones = ConstructedModel(dataset, method="interval_union", reading="half-axes")

    :param method: Catalogue key of the constructed method (``"canonical_model"``,
        ``"least_model"``, ``"interval_union"``, ``"convex_polyhedral"``, ``"alcone-exact"``,
        ``"box_milp"``, ...; see :data:`mowl.representation.construction.RECIPES`).
    :param source, reading: Replace the method's source or reading (names in
        :data:`~mowl.representation.construction.SOURCES` and
        :data:`~mowl.representation.construction.READINGS`).
    :param source_options, reading_options: Options of the parts.

    :attr:`module` is the :class:`~mowl.representation.construction.ConstructionGeometry`:
    class ids are those of :attr:`class_index_dict` (classes the construction does not know
    have the empty region), role ids those of :attr:`object_property_index_dict`,
    individual ids those of :attr:`individual_index_dict`. ``module(rows, "gci0")`` is 0.0
    where an axiom holds in the constructed model and 1.0 where it does not, so the
    evaluation regimes rank with it as with a trained model. :meth:`train` builds nothing
    more: the model is constructed when it is created.
    """

    def __init__(self, dataset, method="canonical_model", source=None, reading=None,
                 source_options=None, reading_options=None, model_filepath=None,
                 device="cpu", batch_size=4096):
        from mowl.representation.construction import RECIPES
        from mowl.representation.methods import METHOD_KINDS
        if method not in RECIPES or METHOD_KINDS.get(method) != "construction":
            raise ValueError(f"Unknown constructed method {method!r}; available: "
                             f"{sorted(RECIPES)}.")
        super().__init__(dataset, 1, batch_size, extended=True,
                         model_filepath=model_filepath, device=device)
        self.method = method
        self.recipe = RECIPES[method]
        _ = self.training_datasets          # loads the axioms, adds auxiliary classes
        source_key = source if isinstance(source, str) or source is None else source.key
        least = (source_key or self.recipe.source) == "least model"
        self.nb_classes = len(self.class_index_dict)
        self.nb_rels = len(self.object_property_index_dict)
        self.nb_individuals = len(self.individual_index_dict)

        def by_id(index):
            return [name for name, _ in sorted(index.items(), key=lambda kv: kv[1])]

        names = {"names": list(range(self.nb_classes)) if least
                 else by_id(self.class_index_dict),
                 "roles": list(range(self.nb_rels)) if least
                 else by_id(self.object_property_index_dict),
                 "individuals": list(range(self.nb_individuals)) if least
                 else by_id(self.individual_index_dict)}
        self.module = self.recipe.build(self if least else dataset, source=source,
                                        reading=reading, source_options=source_options,
                                        reading_options=reading_options, **names)

    @property
    def parts(self):
        """The parts of the combination (``module.parts``)."""
        return self.module.parts

    def train(self, *args, **kwargs):
        """Nothing to train: the model was constructed when it was created."""
        logger.info("%s is constructed, not trained; train() does nothing.", self.method)
        return None
