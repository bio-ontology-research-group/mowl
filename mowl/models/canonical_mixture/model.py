"""CanonicalMixture model class: a constructed canonical model and a trained individual model."""
from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Optional

import mowl.geometric.completion as completion_mod
import mowl.geometric.membership as membership_mod
from mowl.base_models.elmodel import EmbeddingELModel
from mowl.geometric.membership import MEMBERSHIP_CODE_TAG, MembershipStructure
from mowl.nn.el.canonical_mixture import CanonicalMixtureModule
from mowl.representation.construction import READINGS, SOURCES
from mowl.representation.library.mixture_training import IndividualBatches

logger = logging.getLogger(__name__)

__all__ = ["CanonicalMixture"]


class CanonicalMixture(EmbeddingELModel):
    """Canonical-element mixture model for ABox completion under an :math:`\\mathcal{EL}` TBox.

    The method is a combination of parts
    (:mod:`mowl.representation.library.mixture_methods`): the canonical model of the
    training ontology is *constructed* (the source ``"canonical completion"`` of
    :mod:`mowl.representation.construction`) and read sparsely as sets of elements (the
    reading ``"membership"``, a :class:`~mowl.geometric.membership.MembershipStructure`,
    cached next to ``dataset.ontology_path`` when available, keyed by the ontology SHA-256,
    the index maps and the source hashes of ``mowl/geometric/completion.py`` and
    ``mowl/geometric/membership.py``); only the individuals are learned, as distributions
    over the elements (:class:`~mowl.nn.el.canonical_mixture.CanonicalMixtureModule`). The
    training regime (``self.training``, default
    :class:`~mowl.representation.library.mixture_training.IndividualBatches`) goes over
    mini-batches of individuals.

    Guarantees:

    - Raises ``ValueError("CanonicalMixture needs individuals")`` when the
      dataset contains no individuals.
    - Every training-entailed class assertion (including assertions entailed
      through roles) scores ``0.0`` exactly (probability ``1``) and every
      contradicted class assertion scores ``+inf`` (probability ``0``) at
      initialisation and after any number of training epochs.
    - Individuals that occur only in validation or testing splits (no training
      assertions) receive admissible support equal to all of :math:`S`.
    - Hierarchy violation rate is identically ``0`` for all parameters.
    - :meth:`train` iterates over mini-batches of ``batch_size`` individuals per
      epoch, invokes ``epoch_callback(epoch, self)`` when provided, evaluates
      validation loss every ``validate_every`` epochs saving the best
      ``state_dict`` checkpoint to :attr:`model_filepath`, and reloads the best
      checkpoint at the end of :meth:`train`.

    Non-guarantees:

    - Only defined for datasets with individuals; raises :class:`ValueError` on
      TBox-only datasets.
    - Named classes have no dense vector embeddings (a class is its column in
      :attr:`MembershipStructure.M`).

    :param training: The training regime (default :class:`IndividualBatches`; or
        :class:`~mowl.representation.library.mixture_training.AssertionBatches`).
    :param source, reading: The construction (names in
        :data:`~mowl.representation.construction.SOURCES` and
        :data:`~mowl.representation.construction.READINGS`, or instances).
    """

    #: :meth:`train` is the loop of the training regime ``self.training``; it consults no hook
    #: of a :class:`~mowl.training.TrainingRegime`.
    training_hooks = frozenset()

    def __init__(
        self,
        dataset,
        rank: int = 50,
        lr: float = 0.01,
        batch_size: int = 1024,
        entropy_reg: float = 0.0,
        init_self_mass: float = 0.9,
        model_filepath: Optional[str] = None,
        device: str = "cpu",
        load_normalized: bool = False,
        use_cache: bool = True,
        training=None,
        source="canonical completion",
        reading="membership",
    ) -> None:
        if len(dataset.individuals) == 0:
            raise ValueError("CanonicalMixture needs individuals")

        super().__init__(
            dataset,
            embed_dim=rank,
            batch_size=batch_size,
            extended=True,
            model_filepath=model_filepath,
            load_normalized=load_normalized,
            device=device,
            learning_rate=lr,
        )

        if len(self.individual_index_dict) == 0:
            raise ValueError("CanonicalMixture needs individuals")

        self.rank = int(rank)
        self.lr = float(lr)
        self.entropy_reg = float(entropy_reg)
        self.init_self_mass = float(init_self_mass)
        self.use_cache = bool(use_cache)
        self._eval_gci_name = "class_assertion"
        self.source = SOURCES[source]() if isinstance(source, str) else source
        self.reading = READINGS[reading]() if isinstance(reading, str) else reading
        #: The training regime of the parts.
        self.training = training if training is not None else IndividualBatches()

        self.structure: MembershipStructure = self._build_or_load_structure()
        self.init_module()

    @property
    def eval_gci_name(self) -> str:
        """The GCI/assertion type used for validation and evaluation."""
        return self._eval_gci_name

    @eval_gci_name.setter
    def eval_gci_name(self, value: str) -> None:
        valid_names = [
            "gci0",
            "gci1",
            "gci2",
            "gci3",
            "gci0_bot",
            "gci1_bot",
            "gci3_bot",
            "class_assertion",
            "object_property_assertion",
        ]
        if value not in valid_names:
            raise ValueError(
                f"eval_gci_name must be one of {valid_names}, got {value!r}"
            )
        self._eval_gci_name = value

    def _cache_path(self) -> Optional[Path]:
        """Return cache file path next to ``dataset.ontology_path`` if available."""
        ontology_path = getattr(self.dataset, "ontology_path", None)
        if not ontology_path or not os.path.exists(ontology_path):
            return None
        hasher = hashlib.sha256()
        with open(ontology_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        # Hash the source of mowl/geometric/completion.py and mowl/geometric/membership.py
        for mod in (completion_mod, membership_mod):
            mod_file = getattr(mod, "__file__", None)
            if mod_file and os.path.exists(mod_file):
                with open(mod_file, "rb") as mf:
                    hasher.update(mf.read())
        # Mix in class/individual/role index ordering so changes invalidate cache
        for k, v in sorted(self.named_class_index_dict.items()):
            hasher.update(f"C:{k}={v}\n".encode("utf-8"))
        for k, v in sorted(self.individual_index_dict.items()):
            hasher.update(f"I:{k}={v}\n".encode("utf-8"))
        for k, v in sorted(self.object_property_index_dict.items()):
            hasher.update(f"R:{k}={v}\n".encode("utf-8"))
        digest = hasher.hexdigest()[:12]
        base = ontology_path[:-4] if ontology_path.endswith(".owl") else ontology_path
        return Path(f"{base}_mowl_{MEMBERSHIP_CODE_TAG}_{digest}.pt")

    @staticmethod
    def _names(index):
        return [n for n, _ in sorted(index.items(), key=lambda kv: kv[1])]

    def _construct(self) -> MembershipStructure:
        """The constructed model: the source builds the canonical model of the training
        ontology, the reading restricts it to the named classes and the individuals."""
        built = self.source.build(self.dataset.ontology)
        self.reading.check(built)
        index = self.named_class_index_dict
        if sorted(index.values()) != list(range(len(index))):
            raise ValueError("The class ids of the dataset are not 0 .. n - 1.")
        return self.reading.structure(built, self._names(index),
                                      self._names(self.object_property_index_dict),
                                      self._names(self.individual_index_dict))

    def _build_or_load_structure(self) -> MembershipStructure:
        """Build the constructed model from ``dataset.ontology`` or load it from the cache."""
        cache_file = self._cache_path() if self.use_cache else None
        if cache_file is not None and cache_file.exists():
            try:
                return MembershipStructure.load(cache_file)
            except Exception as exc:
                logger.warning(
                    "Failed to load cached MembershipStructure from %s (%s); rebuilding.",
                    cache_file,
                    exc,
                )

        structure = self._construct()
        if cache_file is not None:
            try:
                structure.save(cache_file)
            except OSError as exc:
                logger.warning(
                    "Could not write MembershipStructure cache to %s: %s",
                    cache_file,
                    exc,
                )
        return structure

    def init_module(self) -> None:
        """Initialise :class:`~mowl.nn.el.canonical_mixture.CanonicalMixtureModule`."""
        ca_ds = self.training_datasets.get("class_assertion")
        training_ca = ca_ds[:] if (ca_ds is not None and len(ca_ds) > 0) else None
        self.module = CanonicalMixtureModule(
            self.structure,
            rank=self.rank,
            entropy_reg=self.entropy_reg,
            init_self_mass=self.init_self_mass,
            training_class_assertions=training_ca,
        ).to(self.device)

    def _compute_validation_loss(self) -> float:
        """Compute validation loss on ``validation_datasets``."""
        return self.training.validation_loss(self)

    def train(
        self,
        epochs: int,
        validate_every: int = 1,
        epoch_callback=None,
    ) -> None:
        """Train the model with the training regime ``self.training`` (default: mini-batches
        of individuals).

        Parameters
        ----------
        epochs : int
            Number of training epochs.
        validate_every : int, default 1
            Validate and checkpoint every ``validate_every`` epochs.
        epoch_callback : callable, optional
            Optional callback ``epoch_callback(epoch, self)`` called after each
            epoch.
        """
        if len(self.individual_index_dict) == 0:
            raise ValueError("CanonicalMixture needs individuals")
        self.training.run(self, epochs, validate_every, epoch_callback)
