"""Regime cards: the training of the text-encoder hierarchy methods (HiT, OnT) as parts."""

from mowl.catalog.card import RegimeCard, RegimeKind
from mowl.catalog.regimes._regime_refs import HE2024HIT, YANG2025ONT

REGIMES = (
    RegimeCard(
        key="linear-warmup-decay",
        name="Linear warm-up followed by a linear decay of the learning rate",
        kind=RegimeKind.TRAINING,
        summary=("The learning rate rises linearly from 0 to its base value in a number of "
                 "steps and falls linearly to 0 at the last step, stepped once per "
                 "optimisation step."),
        components=("schedule",),
        applies_to="Models trained by the loop of mowl.training.loop or by the generic EL loop",
        references=(HE2024HIT, YANG2025ONT),
        standalone=False,
        introduced_with=("hit", "ont"),
        used_by=(),
        implementation="mowl.training.text.LinearWarmupDecay",
        notes=("The default scheduler of the trainer of the HiT and OnT codes "
               "(get_linear_schedule_with_warmup of Hugging Face transformers; 500 warm-up "
               "steps). linear-lr-decay has no warm-up and works per epoch."),
    ),
    RegimeCard(
        key="sampled-logical-losses",
        name="Logical losses on samples of the size of the batch",
        kind=RegimeKind.TRAINING,
        summary=("Every step adds the loss of existential restrictions and the loss of "
                 "conjunctions, each on a random sample of the size of the batch of the "
                 "triplets that reuses the negatives of the batch, with a weight each."),
        components=("loss combination", "positives"),
        applies_to="Text-encoder models whose geometry has existential_loss and hierarchy_loss "
                   "(OnT)",
        references=(YANG2025ONT,),
        standalone=False,
        introduced_with=("ont",),
        used_by=(),
        implementation="mowl.training.text.SampledLogicalLosses",
        notes=("Eq. 6 and 7 of OnT as the code adds them: samples from all existential "
               "restrictions and conjunctions of the ontology, negatives a random "
               "permutation of those of the batch."),
    ),
)
