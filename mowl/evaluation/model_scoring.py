"""One way of obtaining axiom scores from any model, for the rank-based regimes.

The regimes of :mod:`mowl.evaluation` rank candidate axioms with a *scorer*
``f(model, gci_name, rows) -> costs`` (lower is more plausible), where ``rows`` are
candidate axioms in mOWL's normal-form column layout
(:data:`mowl.evaluation.axiom_ranking.COLUMNS`), indexed with the model's
``class_index_dict``, ``object_property_index_dict`` and ``individual_index_dict``. A model
says how it provides such scores with its class attribute ``axiom_scoring``:

``"module"``
    ``model.module(rows, gci_name)`` is the cost: the per-axiom losses of the
    :math:`\\mathcal{EL}^{++}` models, the order violation of CatE.
``"evaluation_model"``
    ``model.evaluation_model(rows, gci_name)`` is the cost. This is the adapter mOWL's
    evaluators have always called; models whose geometry is not a normal-form module
    (ALC cones, fuzzy sets, quantum subspaces, binary code boxes, knowledge graph
    embeddings of a projection) provide it. The adapter raises
    :class:`UnsupportedAxiomForm` for a form the model cannot score.
``"method"``
    ``model.axiom_costs(gci_name, rows)`` is the cost, for models whose
    ``evaluation_model`` has another calling convention (the ALC models, whose modules
    score an axiom pattern; knowledge graph embeddings of a projection, whose scorer
    takes triples of graph nodes).
``"vectors"``
    The model gives one vector per class and nothing else (random walks and corpus
    models with Word2Vec). A score needs a *scoring head* trained on the vectors:
    :class:`PairClassifierHead`.

:func:`resolve_model_scorer` turns this into a scorer; ``scoring="auto"`` of every regime
uses it, so a regime does not need to know the model family.
"""

import logging

import numpy as np
import torch as th

logger = logging.getLogger(__name__)

__all__ = ["UnsupportedAxiomForm", "UnsupportedScoring", "axiom_scoring_of",
           "module_scores", "evaluation_model_scores", "method_scores", "PairClassifierHead",
           "pair_classifier_scores", "resolve_model_scorer", "scorable_forms"]


class UnsupportedAxiomForm(ValueError):
    """The model has no score for axioms of this normal form."""


class UnsupportedScoring(TypeError):
    """The model does not say how to obtain axiom scores from it."""


def axiom_scoring_of(model):
    """The way ``model`` provides axiom scores: ``"module"``, ``"evaluation_model"``,
    ``"method"`` or ``"vectors"`` (see the module documentation).

    :raises UnsupportedScoring: If the model does not declare one.
    """
    kind = getattr(model, "axiom_scoring", None)
    if kind not in ("module", "evaluation_model", "method", "vectors"):
        raise UnsupportedScoring(
            f"{type(model).__name__} does not declare how it scores axioms. Set the class "
            f"attribute 'axiom_scoring' to 'module', 'evaluation_model', 'method' or 'vectors' "
            f"(see mowl.evaluation.model_scoring), or pass a scorer "
            f"f(model, gci_name, rows) as the 'scoring' argument of the regime.")
    return kind


def _per_row(out, n):
    return out.reshape(n, -1).mean(dim=1)


def module_scores(model, gci_name, rows):
    """``model.module(rows, gci_name)``, one cost per row."""
    return _per_row(model.module(rows, gci_name), len(rows))


def evaluation_model_scores(model, gci_name, rows):
    """``model.evaluation_model(rows, gci_name)``, one cost per row."""
    out = model.evaluation_model(rows, gci_name)
    if not th.is_tensor(out):
        out = th.as_tensor(np.asarray(out), device=rows.device)
    return _per_row(out, len(rows))


def method_scores(model, gci_name, rows):
    """``model.axiom_costs(gci_name, rows)``, one cost per row."""
    out = model.axiom_costs(gci_name, rows)
    if not th.is_tensor(out):
        out = th.as_tensor(np.asarray(out), device=rows.device)
    return _per_row(out, len(rows))


# ---------------------------------------------------------------------------------------
# Scoring head for models that only give vectors
# ---------------------------------------------------------------------------------------

class PairClassifierHead:
    """A classifier on pairs of class vectors, trained on the training axioms.

    For the axioms :math:`C \\sqsubseteq D` (``gci0``) and, per role :math:`r`,
    :math:`C \\sqsubseteq \\exists r.D` (``gci2``) of the training ontology (for a role
    without training axioms: over all roles), a small
    network on the features :math:`[x_C, x_D, x_C \\odot x_D, |x_C - x_D|]` is trained to
    separate the stated pairs from pairs with a uniformly drawn second class. The cost of
    a candidate axiom is minus its logit. One network is trained per axiom form and
    role, the first time that form is scored.

    This is how vector-only methods are applied to an asymmetric prediction task in the
    literature (a classifier on concatenated vectors; OWL2Vec* trains a random forest or
    a multi-layer perceptron this way for subsumption prediction). Cosine similarity, the
    other common choice, is symmetric and cannot tell :math:`C \\sqsubseteq D` from
    :math:`D \\sqsubseteq C`. The head sees the training ontology only.

    :param hidden: Width of the hidden layer.
    :param epochs: Passes over the training pairs.
    :param negatives: Corrupted pairs per stated pair and pass.
    :param learning_rate: Adam step size.
    :param batch_size: Pairs per step.
    :param seed: Seed of initialisation, negatives and shuffling.
    """

    key = "pair-classifier-head"

    def __init__(self, hidden=256, epochs=50, negatives=4, learning_rate=1e-3,
                 batch_size=1024, seed=0):
        self.hidden = int(hidden)
        self.epochs = int(epochs)
        self.negatives = int(negatives)
        self.learning_rate = float(learning_rate)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self._vectors = None
        self._rows = None
        self._nets = {}
        self.history = {}

    # -- data ---------------------------------------------------------------------------
    def vectors(self, model):
        """Class vectors as one ``(num_classes, dim)`` tensor in ``class_index_dict``
        order (zeros for a class the model has no vector for)."""
        if self._vectors is None:
            embeddings = model.class_embeddings
            index = model.class_index_dict
            dim = len(next(iter(embeddings.values())))
            matrix = np.zeros((len(index), dim), dtype=np.float32)
            missing = 0
            for iri, i in index.items():
                vector = embeddings.get(iri)
                if vector is None:
                    missing += 1
                else:
                    matrix[i] = np.asarray(vector, dtype=np.float32)
            if missing:
                logger.warning("%d of %d classes have no vector; they get the zero vector.",
                               missing, len(index))
            norm = np.linalg.norm(matrix, axis=1, keepdims=True)
            matrix = matrix / np.maximum(norm, 1e-12)
            # Training and scoring run on the model's device (ranking scores every
            # class for every query).
            self._vectors = th.from_numpy(matrix).to(getattr(model, "device", "cpu"))
        return self._vectors

    def _training_rows(self, model):
        if self._rows is None:
            from mowl.evaluation.axiom_ranking import el_rows
            self._rows, _ = el_rows(model, model.dataset.ontology)
        return self._rows

    @staticmethod
    def _features(x, y):
        return th.cat([x, y, x * y, (x - y).abs()], dim=1)

    @staticmethod
    def _in_order(pairs):
        """The pairs sorted (by first, then second class): the axioms come from the
        normaliser in an order that changes from call to call, and the shuffling of
        :meth:`fit` must not depend on it, or the same seed trains another network."""
        if len(pairs) < 2:
            return pairs
        key = pairs[:, 0] * (int(pairs[:, 1].max()) + 1) + pairs[:, 1]
        return pairs[th.argsort(key, stable=True)]

    def _pairs(self, model, gci_name, relation):
        return self._in_order(self._unordered_pairs(model, gci_name, relation))

    def _unordered_pairs(self, model, gci_name, relation):
        rows = self._training_rows(model).get(gci_name)
        if rows is None or len(rows) == 0:
            return th.zeros((0, 2), dtype=th.long)
        if gci_name == "gci2":
            if relation is not None:
                rows = rows[rows[:, 1] == relation]
            return rows[:, [0, 2]]
        if gci_name == "gci0":
            return rows[:, [0, 1]]
        raise UnsupportedAxiomForm(
            f"The pair classifier head scores 'gci0' and 'gci2' axioms, not {gci_name!r}.")

    # -- training -----------------------------------------------------------------------
    def fit(self, model, gci_name, relation=None):
        """Train the network for one axiom form (and role); returns it."""
        key = (gci_name, relation)
        if key in self._nets:
            return self._nets[key]
        vectors = self.vectors(model)
        pairs = self._pairs(model, gci_name, relation)
        if len(pairs) == 0 and relation is not None:
            # A role without a training axiom: the network trained on the pairs of all
            # roles.
            self._nets[key] = self.fit(model, gci_name, None)
            return self._nets[key]
        if len(pairs) == 0:
            raise UnsupportedAxiomForm(
                f"The training ontology has no {gci_name!r} axiom"
                + (f" with role index {relation}" if relation is not None else "")
                + " to train the pair classifier head on.")
        generator = th.Generator().manual_seed(self.seed)
        state = th.random.get_rng_state()
        th.manual_seed(self.seed)
        net = th.nn.Sequential(th.nn.Linear(4 * vectors.shape[1], self.hidden), th.nn.ReLU(),
                               th.nn.Linear(self.hidden, 1))
        th.random.set_rng_state(state)
        device = vectors.device
        net = net.to(device)
        optimizer = th.optim.Adam(net.parameters(), lr=self.learning_rate)
        num_classes = vectors.shape[0]
        losses = []
        with th.enable_grad():
            for _ in range(self.epochs):
                order = th.randperm(len(pairs), generator=generator)
                total = 0.0
                for start in range(0, len(pairs), self.batch_size):
                    batch = pairs[order[start:start + self.batch_size]]
                    sub = batch[:, 0].repeat(1 + self.negatives)
                    sup = th.cat([batch[:, 1], th.randint(
                        num_classes, (len(batch) * self.negatives,), generator=generator)])
                    label = th.zeros(len(sub), device=device)
                    label[:len(batch)] = 1.0
                    sub, sup = sub.to(device), sup.to(device)
                    logit = net(self._features(vectors[sub], vectors[sup])).squeeze(1)
                    loss = th.nn.functional.binary_cross_entropy_with_logits(logit, label)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total += float(loss) * len(batch)
                losses.append(total / len(pairs))
        net.eval()
        self._nets[key] = net
        self.history[key] = losses
        return net

    # -- scoring ------------------------------------------------------------------------
    def __call__(self, model, gci_name, rows):
        if gci_name == "gci0":
            columns, relations = (0, 1), [None]
        elif gci_name == "gci2":
            columns, relations = (0, 2), sorted(set(rows[:, 1].tolist()))
        else:
            raise UnsupportedAxiomForm(
                f"The pair classifier head scores 'gci0' and 'gci2' axioms, not "
                f"{gci_name!r}.")
        vectors = self.vectors(model)
        cpu = rows.cpu()
        out = th.empty(len(cpu))
        with th.no_grad():
            for relation in relations:
                mask = th.ones(len(cpu), dtype=th.bool) if relation is None \
                    else cpu[:, 1] == relation
                net = self.fit(model, gci_name, relation)
                part = cpu[mask]
                scores = []
                for start in range(0, len(part), 2 ** 18):
                    chunk = part[start:start + 2 ** 18].to(vectors.device)
                    scores.append(net(self._features(vectors[chunk[:, columns[0]]],
                                                     vectors[chunk[:, columns[1]]])).cpu())
                out[mask] = -th.cat(scores).squeeze(1)
        return out.to(rows.device)


def pair_classifier_scores(model, gci_name, rows):
    """Costs from the model's :class:`PairClassifierHead` (created with default arguments
    and stored as ``model.scoring_head`` on first use)."""
    head = getattr(model, "scoring_head", None)
    if head is None:
        head = PairClassifierHead()
        model.scoring_head = head
    return head(model, gci_name, rows)


pair_classifier_scores.__name__ = "pair_classifier"


def resolve_model_scorer(model):
    """``(scorer, name)`` for a model, from its ``axiom_scoring`` attribute."""
    kind = axiom_scoring_of(model)
    if kind == "module":
        return module_scores, "loss"
    if kind == "evaluation_model":
        return evaluation_model_scores, "evaluation_model"
    if kind == "method":
        return method_scores, "model"
    return pair_classifier_scores, "pair_classifier"


def scorable_forms(model, forms=("gci0", "gci1", "gci2", "gci3", "class_assertion",
                                 "object_property_assertion")):
    """The normal forms of ``forms`` the model declares a score for (its
    ``scored_forms`` attribute; all of ``forms`` if it has none)."""
    declared = getattr(model, "scored_forms", None)
    return tuple(forms) if declared is None else tuple(f for f in forms if f in declared)
