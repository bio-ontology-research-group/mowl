import logging

import torch as th

from mowl.baselines.base_models.elmodel import EmbeddingELModel
from mowl.baselines.nn.el.mcmultibox.module import VolumeMultiBoxELModule

logger = logging.getLogger(__name__)


class VolumeMultiBoxEL(EmbeddingELModel):
    r"""Volume-based MultiBoxEL, after [aharoni2024]_ (Chapter 7).

    Concepts are unions of ``num_boxes`` boxes in :math:`\mathbb{R}^d`, roles are unions of boxes
    in :math:`\mathbb{R}^{2d}`, and every normal form is trained with the loss
    :math:`1 - \mathrm{Vol}(A \cap B)/\mathrm{Vol}(A)` whose volumes are estimated on a fixed
    quasi-Monte-Carlo point set (see :class:`~mowl.nn.el.mcmultibox.module.VolumeMultiBoxELModule`
    for the exact losses, including the simplified role chain loss).

    This is **not** :class:`mowl.models.MultiBoxEL`, which is a different method (Federico,
    Leemhuis and Kutz, 2026: unions of boxes with Box\ :sup:`2`\ EL-style distance losses).

    **Training.** As in the thesis, the training set of each normal form is split into
    ``num_batches`` batches per epoch (thesis: 5) and one optimiser step is taken per batch.
    The thesis gives no optimiser or learning rate; we use Adam with ``learning_rate``.
    Per-batch memory is bounded by ``max_chunk_elements``: the axioms of a batch are processed
    in chunks and the points in checkpointed chunks of ``point_chunk_size``, so neither the
    batch size nor the point count changes the peak memory (about 2 GiB with the defaults,
    measured on the CPU; see
    :class:`~mowl.nn.el.mcmultibox.module.VolumeMultiBoxELModule`).

    **What is trained.** ``gci0`` to ``gci3`` and, with ``load_role_axioms=True``, role inclusions
    and role chains. The thesis defines nothing for the bottom normal forms, assertions or
    negatives: bottom datasets and assertion datasets are skipped with a warning, so ontologies
    with disjointness axioms are trained without them (unsatisfiable concepts are not pushed to
    the empty set).

    **Evaluation (thesis Section 7.1.2).** The distance of a pair is the fraction of points
    sampled from the boxes of the left concept that fall outside the right concept
    (:meth:`~mowl.nn.el.mcmultibox.module.VolumeMultiBoxELModule.distance`); lower is better.
    The thesis writes the *count* of contained points; the normalisation by the number of points
    and the sign flip are ours. Set :attr:`eval_gci_name` before evaluating.

    **Thesis hyper-parameters:** ``embed_dim=5``, ``num_boxes=5``, ``num_points=20000``,
    5 epochs, ``num_batches=5``, on GPU, about 8 hours per run. The defaults below use these
    except for ``bound`` and ``learning_rate`` which the thesis does not state.

    :param dataset: mOWL dataset.
    :param embed_dim: Dimension :math:`d` of the concept space.
    :param num_boxes: Boxes per multibox.
    :param bound: Half-width :math:`b_o` of the space :math:`S` (not in the thesis).
    :param num_points: Number of quasi-Monte-Carlo points.
    :param temperature: Sigmoid temperature (thesis: 1).
    :param aggregate: ``"min"`` (default) or ``"mean"`` (literal thesis formula); see
        :mod:`mowl.nn.el.mcmultibox.volume`.
    :param learning_rate: Adam learning rate (not in the thesis).
    :param num_batches: Batches per epoch.
    :param seed: Seed for the point set, the initialisation and batch shuffling.
    :param point_chunk_size: Points processed at a time.
    :param max_chunk_elements: Largest intermediate tensor of a loss (axioms x points x
        boxes x coordinates); ``None`` processes a whole batch at once (memory then grows with
        the number of axioms).
    :param eval_points_per_box: Points per box in the evaluation distance.
    :param load_role_axioms: Train on role inclusions and role chains.
    """

    #: :meth:`train` is a loop of its own with a fixed objective; it takes no training regime.
    training_hooks = frozenset()

    #: Normal forms with a loss in the thesis.
    TRAINABLE_GCIS = ("gci0", "gci1", "gci2", "gci3", "role_inclusion", "role_chain")

    def __init__(self, dataset, embed_dim=5, num_boxes=5, bound=1.0, num_points=20000,
                 temperature=1.0, aggregate="min", learning_rate=0.01, num_batches=5,
                 seed=0, point_chunk_size=2048, eval_points_per_box=32,
                 model_filepath=None, device="cpu", load_role_axioms=False,
                 max_chunk_elements=2 ** 25):
        super().__init__(dataset, embed_dim, 4096, extended=True, model_filepath=model_filepath,
                         device=device, learning_rate=learning_rate,
                         load_role_axioms=load_role_axioms)
        if num_batches < 1:
            raise ValueError("num_batches must be at least 1.")
        self.num_boxes = num_boxes
        self.bound = bound
        self.num_points = num_points
        self.temperature = temperature
        self.aggregate = aggregate
        self.num_batches = num_batches
        self.seed = seed
        self.point_chunk_size = point_chunk_size
        self.eval_points_per_box = eval_points_per_box
        self.max_chunk_elements = max_chunk_elements
        #: Mean training loss of each epoch (sum over normal forms), filled by :meth:`train`.
        self.training_losses = []
        self.init_module()

    def init_module(self):
        self.module = VolumeMultiBoxELModule(
            len(self.class_index_dict),
            len(self.object_property_index_dict),
            nb_inds=len(self.individual_index_dict) or None,
            embed_dim=self.embed_dim,
            num_boxes=self.num_boxes,
            bound=self.bound,
            num_points=self.num_points,
            temperature=self.temperature,
            aggregate=self.aggregate,
            seed=self.seed,
            point_chunk_size=self.point_chunk_size,
            eval_points_per_box=self.eval_points_per_box,
            max_chunk_elements=self.max_chunk_elements,
        ).to(self.device)

    def _trainable_datasets(self):
        usable = {}
        skipped = []
        for name, ds in self.training_datasets.items():
            if len(ds) == 0:
                continue
            if name in self.TRAINABLE_GCIS:
                usable[name] = ds
            else:
                skipped.append(name)
        if skipped:
            logger.warning("VolumeMultiBoxEL: the thesis defines no loss for %s; these "
                           "axioms are not used for training.", skipped)
        return usable

    @staticmethod
    def _canonical_order(data):
        """Sort rows lexicographically: the axiom order produced by the normaliser is not
        stable between runs, and sorting makes training reproducible for a fixed seed."""
        for col in reversed(range(data.shape[1])):
            data = data[th.sort(data[:, col], stable=True).indices]
        return data

    def train(self, epochs, validate_every=1, epoch_callback=None):
        """Train with ``num_batches`` optimiser steps per epoch.

        :param epochs: Number of epochs (thesis: 5).
        :param validate_every: Validate every this many epochs on ``eval_gci_name`` of the
            validation set if one exists; the best state is kept in ``model_filepath``.
        :param epoch_callback: Optional ``callable(epoch, model)`` called after each epoch.
        """
        datasets = self._trainable_datasets()
        if not datasets:
            raise ValueError("No axioms with a loss defined by the thesis were found.")
        optimizer = self.get_optimizer()
        gen = th.Generator().manual_seed(self.seed)
        best_valid = float("inf")
        self.training_losses = []

        for epoch in range(epochs):
            self.module.train()
            full = {name: self._canonical_order(ds[:]) for name, ds in datasets.items()}
            splits = {name: th.randperm(len(ds), generator=gen).chunk(self.num_batches)
                      for name, ds in datasets.items()}
            epoch_loss = 0.0
            for b in range(self.num_batches):
                loss = th.tensor(0.0, device=self.device)
                for name in datasets:
                    if b >= len(splits[name]):
                        continue
                    data = full[name][splits[name][b].to(full[name].device)]
                    loss = loss + th.mean(self.module(data, name))
                if not loss.requires_grad:
                    # Fewer axioms than batches: this batch is empty for every axiom type.
                    continue
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.detach().item() / self.num_batches
            self.training_losses.append(epoch_loss)
            msg = f"Epoch {epoch + 1}: train loss {epoch_loss:.4f}"

            valid_loss = None
            if (epoch + 1) % validate_every == 0 and self.dataset.validation is not None \
                    and self._eval_gci_name is not None:
                with th.no_grad():
                    self.module.eval()
                    data = self.validation_datasets[self._eval_gci_name][:]
                    if len(data) > 0:
                        valid_loss = th.mean(self.module(data, self._eval_gci_name)).item()
                        msg += f", valid loss {valid_loss:.4f}"
            if valid_loss is None:
                th.save(self.module.state_dict(), self.model_filepath)
            elif valid_loss < best_valid:
                best_valid = valid_loss
                th.save(self.module.state_dict(), self.model_filepath)
            logger.info(msg)
            if epoch_callback is not None:
                epoch_callback(epoch, self)

    def eval_method(self, data):
        """Evaluation distance of thesis Section 7.1.2 for the GCI type ``eval_gci_name``."""
        if self._eval_gci_name is None:
            raise ValueError("eval_gci_name must be set before evaluation.")
        return self.module.distance(data, self._eval_gci_name)

    @property
    def evaluation_model(self):
        """Callable ``(data, gci_name) -> distances`` for the mOWL evaluators (lower is better)."""
        if self._evaluation_model is not None:
            return self._evaluation_model
        module = self.module

        def model(data, gci_name, neg=False):
            return module.distance(data, gci_name)

        return model

    @evaluation_model.setter
    def evaluation_model(self, value):
        self._evaluation_model = value
