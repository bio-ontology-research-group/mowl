"""CanonicalMixture model class (R4)."""
from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Optional

import torch as th
from tqdm import trange

from mowl.base_models.elmodel import EmbeddingELModel
from mowl.datasets.el import ELDataset
from mowl.geometric.completion import CanonicalModel
from mowl.geometric.membership import MEMBERSHIP_CODE_TAG, MembershipStructure
from mowl.nn.el.canonical_mixture import CanonicalMixtureModule

logger = logging.getLogger(__name__)


class CanonicalMixture(EmbeddingELModel):
    """Canonical-element mixture model (R4) for ABox completion under an :math:`\\mathcal{EL}` TBox.

    Builds the :class:`~mowl.geometric.completion.CanonicalModel` and
    :class:`~mowl.geometric.membership.MembershipStructure` of the training
    ontology (caching the :class:`MembershipStructure` next to
    ``dataset.ontology_path`` when available) and trains a
    :class:`~mowl.nn.el.canonical_mixture.CanonicalMixtureModule` with mini-batches
    of individuals.

    Guarantees:
    - Raises ``ValueError("CanonicalMixture needs individuals")`` when the
      dataset contains no individuals.
    - Every training-entailed class assertion scores ``0.0`` (probability ``1``)
      and every contradicted class assertion scores ``+inf`` (probability ``0``)
      at initialisation and after any number of training epochs.
    - Hierarchy violation rate is identically ``0`` for all parameters.
    - :meth:`train` iterates over mini-batches of ``batch_size`` individuals per
      epoch, invokes ``epoch_callback(epoch, self)`` when provided, and
      evaluates validation loss every ``validate_every`` epochs saving the best
      ``state_dict`` checkpoint to :attr:`model_filepath`.

    Non-guarantees:
    - Only defined for datasets with individuals; raises :class:`ValueError` on
      TBox-only datasets.
    - Named classes have no dense vector embeddings (a class is its column in
      :attr:`MembershipStructure.M`).
    """

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

    def _load_datasets(self) -> None:
        """Load GCI and ABox assertion datasets, passing ``individual_index_dict``."""
        if self._datasets_loaded:
            return

        ontology_path = getattr(self.dataset, "ontology_path", None)
        validation_path = getattr(self.dataset, "validation_path", None)
        testing_path = getattr(self.dataset, "testing_path", None)

        training_el_dataset = ELDataset(
            self.dataset.ontology,
            self.class_index_dict,
            self.object_property_index_dict,
            individual_index_dict=self.individual_index_dict,
            extended=self._extended,
            load_normalized=self.load_normalized,
            device=self.device,
            ontology_path=ontology_path,
            load_role_axioms=self.load_role_axioms,
            normalized_gcis=self._training_normalized_gcis,
        )
        self._training_datasets = training_el_dataset.get_gci_datasets()

        self._validation_datasets = None
        if self.dataset.validation is not None:
            validation_el_dataset = ELDataset(
                self.dataset.validation,
                self.class_index_dict,
                self.object_property_index_dict,
                individual_index_dict=self.individual_index_dict,
                extended=self._extended,
                device=self.device,
                ontology_path=validation_path,
                load_role_axioms=self.load_role_axioms,
            )
            self._validation_datasets = validation_el_dataset.get_gci_datasets()

        self._testing_datasets = None
        if self.dataset.testing is not None:
            testing_el_dataset = ELDataset(
                self.dataset.testing,
                self.class_index_dict,
                self.object_property_index_dict,
                individual_index_dict=self.individual_index_dict,
                extended=self._extended,
                device=self.device,
                ontology_path=testing_path,
                load_role_axioms=self.load_role_axioms,
            )
            self._testing_datasets = testing_el_dataset.get_gci_datasets()

        self._datasets_loaded = True

    def _cache_path(self) -> Optional[Path]:
        """Return cache file path next to ``dataset.ontology_path`` if available."""
        ontology_path = getattr(self.dataset, "ontology_path", None)
        if not ontology_path or not os.path.exists(ontology_path):
            return None
        hasher = hashlib.sha256()
        with open(ontology_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        # Also mix in class/individual/role index ordering so changes invalidate cache
        for k, v in sorted(self.named_class_index_dict.items()):
            hasher.update(f"C:{k}={v}\n".encode("utf-8"))
        for k, v in sorted(self.individual_index_dict.items()):
            hasher.update(f"I:{k}={v}\n".encode("utf-8"))
        for k, v in sorted(self.object_property_index_dict.items()):
            hasher.update(f"R:{k}={v}\n".encode("utf-8"))
        digest = hasher.hexdigest()[:12]
        base = ontology_path[:-4] if ontology_path.endswith(".owl") else ontology_path
        return Path(f"{base}_mowl_{MEMBERSHIP_CODE_TAG}_{digest}.pt")

    def _build_or_load_structure(self) -> MembershipStructure:
        """Build :class:`MembershipStructure` from ``dataset.ontology`` or load from cache."""
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

        cm = CanonicalModel.from_ontology(self.dataset.ontology)
        structure = MembershipStructure.from_canonical_model(
            cm,
            class_index_dict=self.named_class_index_dict,
            individual_index_dict=self.individual_index_dict,
            relation_index_dict=self.object_property_index_dict,
        )
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
        with th.no_grad():
            self.module.eval()
            val_datasets = self.validation_datasets
            if not val_datasets:
                return 0.0

            gci_name = self._eval_gci_name
            if gci_name in val_datasets and len(val_datasets[gci_name]) > 0:
                v_data = val_datasets[gci_name][:]
                return float(th.mean(self.module(v_data, gci_name)).detach().item())

            for fallback in ("class_assertion", "object_property_assertion"):
                if fallback in val_datasets and len(val_datasets[fallback]) > 0:
                    v_data = val_datasets[fallback][:]
                    return float(th.mean(self.module(v_data, fallback)).detach().item())
            return 0.0

    def train(
        self,
        epochs: int,
        validate_every: int = 1,
        epoch_callback=None,
    ) -> None:
        """Train the model using mini-batches of individuals.

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
        n_ind = len(self.individual_index_dict)
        if n_ind == 0:
            raise ValueError("CanonicalMixture needs individuals")

        ca_ds = self.training_datasets.get("class_assertion")
        ca_data = (
            ca_ds[:].to(self.device)
            if (ca_ds is not None and len(ca_ds) > 0)
            else th.empty((0, 2), dtype=th.long, device=self.device)
        )
        opa_ds = self.training_datasets.get("object_property_assertion")
        opa_data = (
            opa_ds[:].to(self.device)
            if (opa_ds is not None and len(opa_ds) > 0)
            else th.empty((0, 3), dtype=th.long, device=self.device)
        )

        optimizer = self.get_optimizer()
        best_loss = float("inf")

        for epoch in trange(epochs):
            self.module.train()
            perm = th.randperm(n_ind, device=self.device)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n_ind, self.batch_size):
                batch_inds = perm[start : start + self.batch_size]
                loss = th.tensor(0.0, device=self.device)
                has_supervision = False

                if ca_data.shape[0] > 0:
                    mask_ca = th.isin(ca_data[:, 0], batch_inds)
                    if bool(mask_ca.any().item()):
                        pos_ca = ca_data[mask_ca]
                        neg_ca = self.module.sample_class_negatives(pos_ca)
                        loss_ca = (
                            self.module.class_assertion_loss(pos_ca, neg=False).mean()
                            + self.module.class_assertion_loss(neg_ca, neg=True).mean()
                        )
                        loss = loss + loss_ca
                        has_supervision = True

                if opa_data.shape[0] > 0:
                    mask_opa = th.isin(opa_data[:, 0], batch_inds)
                    if bool(mask_opa.any().item()):
                        pos_opa = opa_data[mask_opa]
                        neg_opa = self.module.sample_role_negatives(pos_opa)
                        loss_opa = (
                            self.module.object_property_assertion_loss(
                                pos_opa, neg=False
                            ).mean()
                            + self.module.object_property_assertion_loss(
                                neg_opa, neg=True
                            ).mean()
                        )
                        loss = loss + loss_opa
                        has_supervision = True

                reg = self.module.regularization_loss(batch_inds)
                if has_supervision or self.entropy_reg > 0.0:
                    loss = loss + reg
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    epoch_loss += float(loss.detach().item())
                    n_batches += 1

            train_loss = epoch_loss / max(n_batches, 1)

            if epoch_callback is not None:
                epoch_callback(epoch, self)

            if validate_every and (epoch + 1) % validate_every == 0:
                if self.dataset.validation is not None:
                    valid_loss = self._compute_validation_loss()
                    if valid_loss < best_loss:
                        best_loss = valid_loss
                        th.save(self.module.state_dict(), self.model_filepath)
                else:
                    if train_loss < best_loss:
                        best_loss = train_loss
                        th.save(self.module.state_dict(), self.model_filepath)
