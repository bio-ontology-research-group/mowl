"""Direct implementations, kept for comparison.

Every method of mOWL is defined as a combination of parts (class model, relation model,
measure, objective, regularisation, training regime, evaluation regime; see
:mod:`mowl.representation`). The classes users work with (:mod:`mowl.models`,
:mod:`mowl.nn`) are those combinations.

This package holds the earlier, direct implementation of each method: one hand-written
loss module and model class per paper. They are the reference that the combinations are
tested against (same losses, same training run, same scores; see ``tests/baselines``) and
benchmarked against. Nothing outside this package imports it.

``mowl.baselines.models.<method>`` and ``mowl.baselines.nn.<...>`` mirror the layout of
:mod:`mowl.models` and :mod:`mowl.nn`, with the same class names.
"""
