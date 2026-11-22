"""Regime card: Shuffled splits: an epoch of several steps."""

from mowl.catalog.card import Reference, RegimeCard, RegimeKind

AHARONI2024 = Reference(
    "aharoni2024thesis",
    "Aharoni, R. (2024). Volume-based MultiBoxEL, Chapter 7 of the thesis.",
    "https://github.com/KRR-Oxford")

REGIME = RegimeCard(
    key='shuffled-splits',
    name='Shuffled splits: an epoch of several steps',
    kind=RegimeKind.TRAINING,
    summary=(
        'At the start of every epoch every normal form is shuffled and cut into'
        'num_batches parts; the epoch takes one step on the b-th part of every normal'
        'form. Every axiom is used once per epoch, in a fresh random batch.'),
    components=('batching',),
    applies_to='EmbeddingELModel and models built on it (one generic epoch is one step)',
    references=(AHARONI2024,),
    standalone=False,
    introduced_with=('mcmultibox',),
    used_by=(),
    implementation='mowl.training.splits.ShuffledSplits',
    notes=(
        'Volume MultiBoxEL (Aharoni 2024, Chapter 7: five batches per epoch). Differs'
        'from per-normal-form-minibatch, which draws every batch independently. Uses a'
        'generator of its own seeded when training starts, and sorts the axioms so a run'
        'is reproducible.'),
)
