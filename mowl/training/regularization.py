"""Regularisation as a component of its own.

Embedding methods restrict where their parameters may lie, and publications describe this in
one line ("the centres are normalised", "we add a norm penalty"). Three different things go
under the name, and they behave differently:

*Penalties* add a term to the loss that grows as the parameters leave the allowed set, for
example ELEmbeddings' :math:`|\\lVert c \\rVert - 1|` for every class centre. The constraint
holds only approximately and competes with the axiom losses.

*Projections* move the parameters back into the allowed set after every optimisation step,
for example onto the unit sphere (TransE) or into the unit cube. The constraint holds
exactly, and the optimiser never sees it.

*Parametrisations* make a violation impossible, for example a box size stored as its
logarithm, or a point of the Poincaré ball mapped into the ball in the forward pass. They
are part of the representation, stay in the model, and are not handled here. The same holds
for a projection the geometry of a model needs to be well defined (the unit cube of the box
lattice model): the model applies it itself and no regime can switch it off.

A :class:`Regularizer` is a penalty or a projection on a chosen set of embedding tables. The
training regime :class:`Regularization` applies any number of them to a model, in addition
to or instead of the once-per-step regularisation term the model's module defines
(``regularization_loss``):

.. code-block:: python

   # ELEm without its unit-norm penalty
   model.training_regime = Regularization(keep_default=False)
   # ELEm with the centres projected onto the unit sphere instead
   model.training_regime = Regularization(SphereProjection(), keep_default=False)
   # ELBE with ELEm's penalty on the box centres
   model.training_regime = Regularization(NormPenalty(parameters="class_embed"))

Terms that a module adds inside the loss of every axiom (EmEL++, or ELEm with
:class:`~mowl.training.PerAxiomNormPenalty`) are part of that loss and are not affected by
``keep_default``.
"""

import torch as th

from mowl.training.regime import TrainingRegime


def embedding_tables(model):
    """Names of the embedding tables (:class:`torch.nn.Embedding`) of the module of
    ``model``, the names :class:`Regularizer` accepts as ``parameters``."""
    module = getattr(model, "module", model)
    return tuple(name for name, sub in module.named_modules()
                 if isinstance(sub, th.nn.Embedding))


class Regularizer:
    """A constraint on embedding tables, as a penalty or as a projection.

    :param parameters: What is constrained: the name of an attribute of the model's module
        (an embedding table or a parameter, dotted names allowed), a sequence of such names,
        or a function from the model to an iterable of tensors. Every tensor is read as one
        vector per row. Default: ``"class_embed"``.
    :param columns: ``(start, stop)`` to constrain only these columns of every vector, for
        modules that store several things in one table (a box as ``min | size``).
        Default: all columns.
    :param weight: Factor of the penalty (penalties only).
    """

    #: ``"penalty"`` or ``"projection"``.
    kind = ""

    def __init__(self, parameters="class_embed", columns=None, weight=1.0):
        self.parameters = parameters
        if columns is not None:
            columns = tuple(columns)
            if len(columns) != 2:
                raise ValueError("columns must be (start, stop).")
        self.columns = columns
        self.weight = weight

    def tensors(self, model):
        """The parameter tensors this regulariser constrains in ``model`` (a model or a
        module), each two-dimensional."""
        module = getattr(model, "module", model)
        if callable(self.parameters):
            found = list(self.parameters(model))
        else:
            names = (self.parameters,) if isinstance(self.parameters, str) \
                else tuple(self.parameters)
            found = []
            for name in names:
                target = module
                try:
                    for part in name.split("."):
                        target = getattr(target, part)
                except AttributeError:
                    raise TypeError(
                        f"{type(module).__name__} has no parameter {name!r} to regularise; "
                        f"its embedding tables are {list(embedding_tables(module))}.") from None
                found.append(target.weight if isinstance(target, th.nn.Module) else target)
        for t in found:
            if not isinstance(t, th.Tensor) or t.dim() != 2:
                raise TypeError("A regulariser constrains two-dimensional tensors "
                                "(one vector per row).")
        return found

    def _view(self, tensor):
        if self.columns is None:
            return tensor
        return tensor[:, self.columns[0]:self.columns[1]]

    def penalty(self, model):
        """The loss term of this regulariser for ``model`` (a scalar tensor), or ``None``
        for a projection."""
        return None

    @th.no_grad()
    def project(self, model):
        """Move the parameters of ``model`` back into the allowed set, in place. Does
        nothing for a penalty."""

    def __repr__(self):
        return f"{type(self).__name__}(parameters={self.parameters!r})"


class _Penalty(Regularizer):
    kind = "penalty"

    def _term(self, x):
        raise NotImplementedError

    def penalty(self, model):
        total = None
        for t in self.tensors(model):
            term = self._term(self._view(t))
            total = term if total is None else total + term
        return None if total is None else self.weight * total


class _Projection(Regularizer):
    kind = "projection"

    def _projected(self, x):
        raise NotImplementedError

    @th.no_grad()
    def project(self, model):
        for t in self.tensors(model):
            view = self._view(t)
            view.copy_(self._projected(view))


class NormPenalty(_Penalty):
    """:math:`\\operatorname{mean}_x |\\lVert x \\rVert_p - \\text{target}|`: every vector
    is pulled towards the sphere of radius ``target``. With the defaults this is the
    regularisation of ELEmbeddings (Kulmanov et al. 2019) on the class centres.

    :param target: Radius of the sphere.
    :param p: Order of the norm.
    """

    def __init__(self, parameters="class_embed", target=1.0, p=2, columns=None, weight=1.0):
        super().__init__(parameters, columns, weight)
        self.target = target
        self.p = p

    def _term(self, x):
        return th.abs(th.linalg.norm(x, ord=self.p, dim=1) - self.target).mean()


class MaxNormPenalty(_Penalty):
    """:math:`\\operatorname{mean}_x \\max(0, \\lVert x \\rVert_p - \\text{max\\_norm})`:
    vectors inside the ball of radius ``max_norm`` are free, vectors outside are pulled
    back. The soft form of :class:`BallProjection`."""

    def __init__(self, parameters="class_embed", max_norm=1.0, p=2, columns=None, weight=1.0):
        super().__init__(parameters, columns, weight)
        self.max_norm = max_norm
        self.p = p

    def _term(self, x):
        return th.relu(th.linalg.norm(x, ord=self.p, dim=1) - self.max_norm).mean()


class PowerPenalty(_Penalty):
    """:math:`\\operatorname{mean}_x \\sum_i |x_i|^p`: shrinks the vectors towards the
    origin. ``p=2`` is the usual L2 penalty, ``p=1`` favours sparse vectors, ``p=3`` is the
    N3 regulariser of Lacroix et al. (2018)."""

    def __init__(self, parameters="class_embed", p=2, columns=None, weight=1.0):
        super().__init__(parameters, columns, weight)
        self.p = p

    def _term(self, x):
        return (th.abs(x) ** self.p).sum(dim=1).mean()


class SphereProjection(_Projection):
    """After every step, rescale every vector to norm ``radius``: the constraint
    :class:`NormPenalty` approximates, enforced exactly (the entity normalisation of
    TransE, Bordes et al. 2013)."""

    def __init__(self, parameters="class_embed", radius=1.0, p=2, columns=None, eps=1e-12):
        super().__init__(parameters, columns)
        self.radius = radius
        self.p = p
        self.eps = eps

    def _projected(self, x):
        norm = th.linalg.norm(x, ord=self.p, dim=1, keepdim=True).clamp_min(self.eps)
        return x * (self.radius / norm)


class BallProjection(_Projection):
    """After every step, rescale the vectors whose norm exceeds ``max_norm`` to norm
    ``max_norm``; vectors inside the ball are left alone."""

    def __init__(self, parameters="class_embed", max_norm=1.0, p=2, columns=None, eps=1e-12):
        super().__init__(parameters, columns)
        self.max_norm = max_norm
        self.p = p
        self.eps = eps

    def _projected(self, x):
        norm = th.linalg.norm(x, ord=self.p, dim=1, keepdim=True).clamp_min(self.eps)
        return x * th.clamp(self.max_norm / norm, max=1.0)


class IntervalProjection(_Projection):
    """After every step, clamp every coordinate to ``[low, high]`` (either may be
    ``None``): the unit cube with ``(0, 1)``, non-negative sizes with ``(0, None)``."""

    def __init__(self, parameters="class_embed", low=None, high=None, columns=None):
        super().__init__(parameters, columns)
        if low is None and high is None:
            raise ValueError("Give low, high or both.")
        self.low = low
        self.high = high

    def _projected(self, x):
        return x.clamp(min=self.low, max=self.high)


class ModuleProjection(Regularizer):
    """After every step, the projection that the module of the model defines (``module.project()``):
    for a geometry whose parts keep their parameters in a region, for example the balls around
    the centroids of FaithEL. A module without ``project`` is refused."""

    kind = "projection"

    def __init__(self):
        super().__init__("class_embed")

    def tensors(self, model):
        module = getattr(model, "module", model)
        if not callable(getattr(module, "project", None)):
            raise TypeError(f"{type(module).__name__} has no project() to apply.")
        return []

    @th.no_grad()
    def project(self, model):
        model.module.project()


class Regularization(TrainingRegime):
    """Apply regularisers to a model: the penalties are added to the loss of every
    optimisation step, the projections are applied after every step, in the order given.

    :param regularizers: :class:`Regularizer` objects.
    :param keep_default: Keep the once-per-step regularisation term of the model's module
        (default). ``False`` drops it, so that the regularisers given here replace it;
        ``Regularization(keep_default=False)`` trains without it.
    """

    key = "regularization"

    def __init__(self, *regularizers, keep_default=True):
        super().__init__()
        for r in regularizers:
            if not isinstance(r, Regularizer):
                raise TypeError("Regularization takes Regularizer objects.")
        self.regularizers = tuple(regularizers)
        self.keep_default = keep_default

    def overridden_hooks(self):
        hooks = set()
        if not self.keep_default or any(r.kind == "penalty" for r in self.regularizers):
            hooks.add("regularization")
        if any(r.kind == "projection" for r in self.regularizers):
            hooks.add("after_step")
        return frozenset(hooks)

    def attach(self, model):
        super().attach(model)
        for r in self.regularizers:
            r.tensors(model)  # fail now, not at the first step

    def regularization(self, default):
        total = default if self.keep_default else 0
        for r in self.regularizers:
            term = r.penalty(self.model)
            if term is not None:
                total = total + term
        return total

    def after_step(self):
        for r in self.regularizers:
            r.project(self.model)
