"""Direct implementations of the constructed methods of :mod:`mowl.baselines.geometric.core`, kept for
comparison (see :mod:`mowl.baselines`).

Each of them builds the regions of its paper in one piece of code. In :mod:`mowl.baselines.geometric.core`
the same classes are combinations of a source and a reading
(:mod:`mowl.representation.construction`); ``tests/baselines/pairs_construct.py`` checks
that both give the same regions, the same truth values and the same scores. The neutral
mathematics (completion, the model checker, the region classes of
:mod:`mowl.baselines.geometric.core.regions`, finite interpretations, the syntax of ALC and LTL_ALC) is
imported from :mod:`mowl.baselines.geometric.core`.
"""
