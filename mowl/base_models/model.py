from deprecated.sphinx import deprecated, versionchanged, versionadded
import functools
import re
import tempfile
from mowl.datasets import Dataset
from mowl.owlapi import OWLAPIAdapter
from mowl.evaluation import RankingEvaluator
from mowl.error import messages as msg
from java.util import HashSet



#: The arguments that swap a part of a method (see :mod:`mowl.representation.swap`).
PART_ARGUMENTS = ("classes", "relations", "individuals", "measure", "objective",
                  "objective_options", "regularizers", "intersection", "distance",
                  "negation", "source", "reading", "encoder", "scorer", "walker", "view")


def _refusing_parts(init):
    """``init`` that explains a part argument it does not take: the named classes are one
    combination of parts each; the parts are swapped in the general models."""
    @functools.wraps(init)
    def wrapper(self, *args, **kwargs):
        try:
            init(self, *args, **kwargs)
        except TypeError as error:
            match = re.search(r"unexpected keyword argument '(\w+)'", str(error))
            if match and match.group(1) in PART_ARGUMENTS and match.group(1) in kwargs:
                raise TypeError(
                    f"{type(self).__name__} is a named combination of parts and takes only "
                    f"its own options, not the part {match.group(1)!r}. To replace a part, "
                    f"use the general model of its kind with the method key: "
                    f"ComposedEL / ComposedALC / ComposedSequenceModel / "
                    f"QueryEmbeddingModel / ConstructedModel(dataset, method=<key>, "
                    f"{match.group(1)}=...) (see the page 'Combine parts').") from error
            raise
    return wrapper


@versionchanged(version="0.1.0", reason="Parameter ``model_filepath`` added in the base class for \
    all models. Optional parameter that will use temporary files in case it is not set.")
class Model():
    """Abstract model class.

    :param dataset: Dataset object.
    :type dataset: mowl.datasets.base.Dataset
    :param model_filepath: Path for saving the model. Defaults to a temporary file path.
    :type model_filepath: str, optional
    """


    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if "__init__" in cls.__dict__:
            cls.__init__ = _refusing_parts(cls.__dict__["__init__"])

    #: How the rank-based evaluation regimes obtain axiom scores from the model:
    #: ``"module"``, ``"evaluation_model"``, ``"method"`` or ``"vectors"``; ``None`` if
    #: the model has none (see :mod:`mowl.evaluation.model_scoring`).
    axiom_scoring = None
    #: The normal forms the model has a score for (``None``: not restricted).
    scored_forms = None

    def __init__(self, dataset, model_filepath=None):
        if not isinstance(dataset, Dataset):
            raise TypeError("Parameter dataset must be a mOWL Dataset.")

        if model_filepath is not None and not isinstance(model_filepath, str):
            raise TypeError("Optional parameter model_filepath must be of type str.")

        self.dataset = dataset
        self._model_filepath = model_filepath
        self._testing_set = None
        self._evaluator = None
        self._evaluation_model = None
        self._metrics = None
        self.expressivity_report = None
        if Model.check_expressivity_on_init:
            self.check_expressivity()

    #: If ``True`` (default), every model compares its ontology with the expressivity its
    #: method supports when it is created, and warns if axioms go beyond it (they are
    #: ignored or only partly used). Set ``mowl.base_models.model.Model
    #: .check_expressivity_on_init = False`` to switch the check off.
    check_expressivity_on_init = True

    #: The :data:`mowl.training.regime.HOOKS` the training loop of this model consults.
    #: Empty: the model trains with an objective of its own and takes no training regime.
    training_hooks = frozenset()

    @property
    def training_regime(self):
        """The :class:`~mowl.training.TrainingRegime` used by :meth:`train`, or ``None``.

        A regime can be assigned only if the training loop of the model consults every
        hook the regime overrides (:attr:`training_hooks`); otherwise assigning it raises
        :class:`TypeError`, so that a regime is never silently ignored.
        """
        return getattr(self, "_training_regime", None)

    @training_regime.setter
    def training_regime(self, regime):
        if regime is not None:
            from mowl.training.regime import TrainingRegime, check_compatible
            if not isinstance(regime, TrainingRegime):
                raise TypeError("training_regime must be a mowl.training.TrainingRegime "
                                "or None.")
            check_compatible(regime, self)
            regime.attach(self)
        self._training_regime = regime

    @property
    def card(self):
        """The catalogue card (:class:`mowl.catalog.MethodCard`) of this method, or
        ``None`` if the class is not catalogued."""
        from mowl.catalog import card_for
        return card_for(type(self))

    @property
    def expressivity(self):
        """The :class:`mowl.logic.Expressivity` this method supports, or ``None`` if it is
        not catalogued or not defined on a description logic."""
        card = self.card
        return None if card is None else card.supported_expressivity

    def check_expressivity(self, warn=True):
        """Compare the training ontology with the expressivity this method supports.

        Axioms that use constructs outside the supported logic are ignored, or only partly
        used, by the method. The result is stored in :attr:`expressivity_report`.

        :param warn: Emit a :class:`UserWarning` that lists the axiom types and constructs
            outside the supported logic.
        :returns: A :class:`mowl.logic.ExpressivityReport`, or ``None`` if the method
            states no expressivity.
        """
        from mowl.logic import check_ontology
        supported = self.expressivity
        if supported is None:
            return None
        self.expressivity_report = check_ontology(self.dataset.ontology, supported,
                                                  warn=False)
        if warn and not self.expressivity_report.ok:
            import warnings
            warnings.warn(f"{type(self).__name__}: {self.expressivity_report.summary()}",
                          UserWarning, stacklevel=3)
        return self.expressivity_report

    @property
    def as_geometry(self):
        """What the model contains, behind one interface
        (:class:`mowl.representation.Geometry`): the regions of classes, individuals and
        compound classes, and the relations between them.

        :raises NotImplementedError: If the method does not expose a geometry yet.
        """
        from mowl.representation import geometry_of
        return geometry_of(self)

    def train(self, *args, **kwargs):
        '''Abstract method for training the model. This method must be implemented in children classes
        '''
        raise NotImplementedError("Method train is not implemented.")

    def evaluate(self, *args, **kwargs):
        if self._evaluator is None:
            raise AttributeError(msg.EVALUATOR_NOT_SET)

        self._metrics = self._evaluator.evaluate(self.evaluation_model,
                                                 *args, **kwargs)
        
        
    
    def eval_fn(self, *args, **kwargs):
        raise NotImplementedError("Method eval_fn is not implemented.")

    @versionadded(version="0.2.0", reason="Axiom scoring method added to the base class.")
    def score(self, axiom):
        """
        Returns the score of the given axiom.

        :param axiom: The axiom to score.
        :type axiom: :class:`org.semanticweb.owlapi.model.OWLAxiom`
        """
        
        raise NotImplementedError("Method score is not implemented.")

    
    @property
    def model_filepath(self):
        """Path for saving the model.

        :rtype: str
        """
        if self._model_filepath is None:
            model_filepath = tempfile.NamedTemporaryFile()
            self._model_filepath = model_filepath.name
        return self._model_filepath

    @property
    def class_index_dict(self):
        """Dictionary with class names as keys and class indexes as values.

        :rtype: dict
        """
        class_to_id = {v: k for k, v in enumerate(self.dataset.classes.as_str)}
        return class_to_id

    @property
    def individual_index_dict(self):
        """Dictionary with individual names as keys and indexes as values.

        :rtype: dict
        """
        individual_to_id = {v: k for k, v in enumerate(self.dataset.individuals.as_str)}
        return individual_to_id
                            
    @property
    def object_property_index_dict(self):
        """Dictionary with object property names as keys and object property indexes as values.

        :rtype: dict
        """
        object_property_to_id = {v: k for k, v in enumerate(self.dataset.object_properties.as_str)}
        return object_property_to_id

    @versionadded(version="0.2.0")
    @property
    def class_embeddings(self):
        """
        Returns a dictionary with class names as keys and class embeddings as values.
        
        :rtype: dict
        """
        raise NotImplementedError()

    @versionadded(version="0.2.0")
    @property
    def object_property_embeddings(self):
        """
        Returns a dictionary with object property names as keys and object property embeddings as values.

        :rtype: dict
        """
        raise NotImplementedError()

    @versionadded(version="0.2.0")
    @property
    def individual_embeddings(self):
        """
        Returns a dictionary with individual names as keys and individual embeddings as values.

        :rtype: dict
        """
        raise NotImplementedError()

    @versionadded(version="1.0.0")
    @property
    def evaluation_model(self):
        """Returns the evaluation model. In models relying on Word2Vec embeddings, this method calls an auxiliary evaluation model for scoring. Methods using KGEs or Geometric Embeddings would return the model itself."""
        raise NotImplementedError("Method evaluation_model must be implemented in a subclass.")

    @versionadded(version="1.0.0")
    @property
    def metrics(self):
        if self._metrics is None:
            raise AttributeError("Model has not been evaluated yet.")
        else:
            return self._metrics

    
    @versionadded(version="0.2.0")
    def add_axioms(self, *axioms):
        """
        This method adds axioms to the dataset contained in the model and reorders the embedding information for each entity accordingly. New entites are initalized with random embedding.
        
        :param axioms: Axioms to be added to the dataset.
        :type axioms: org.semanticweb.owlapi.model.OWLAxiom
        """
        raise NotImplementedError()

    @versionadded(version="0.2.0")
    def from_pretrained(self, file_name):
        """
        This method loads a pretrained model from a file.

        :param file_name: Path to the pretrained model file.
        :type file_name: str
        """
        raise NotImplementedError()
    


    @versionadded(version="1.0.0")
    def set_evaluator(self, evaluator, *args, **kwargs):
        """
        This method sets the evaluator for the model.

        :param evaluator: Evaluator object.
        :type evaluator: mowl.evaluation.base.Evaluator
        """
        
        if isinstance(evaluator, RankingEvaluator):
            self._evaluator = evaluator
        else:
            self._evaluator = evaluator(self.dataset, *args, **kwargs)

        
            
