"""Regime cards: the parts of GroundedEL's training design, each a shared regime, and
their composition."""

from mowl.catalog.card import Reference, RegimeCard, RegimeKind

# No publication: the design is original to mOWL and specified in the model's docstring.
MOWL_GROUNDED = Reference(
    key="mowl-grounded",
    citation="mOWL (original, unpublished): GroundedEL, docstring of "
             "mowl.models.grounded.model.GroundedEL and docs/source/methods/grounded.rst.",
)

_NOTE = ("Original to mOWL (GroundedEL). Before, this was a fixed step of GroundedEL's "
         "own training loop; the model now runs through the regime hooks and uses this "
         "regime by default.")

REGIMES = (
    RegimeCard(
        key="temperature-annealing",
        name="Temperature annealing of soft memberships",
        kind=RegimeKind.TRAINING,
        summary=("The temperature of the model's soft (sigmoid) memberships goes from a "
                 "high value at the first epoch to a low one at the last, geometrically "
                 "or linearly: smooth gradients first, agreement with the crisp regions "
                 "at the end."),
        components=("schedule",),
        applies_to="Models whose module has a temperature (set_tau(value) or a tau "
                   "attribute): GroundedEL, MultiBoxEL",
        references=(MOWL_GROUNDED,),
        standalone=False,
        introduced_with=("grounded",),
        used_by=("multiboxel",),
        implementation="mowl.training.schedules.TemperatureAnnealing",
        notes=_NOTE + " MultiBoxEL has the same schedule built in (tau_anneal=True).",
    ),
    RegimeCard(
        key="linear-lr-decay",
        name="Linearly decayed learning rate",
        kind=RegimeKind.TRAINING,
        summary="The learning rate decays linearly from its initial value to a floor at "
                "the last epoch.",
        components=("optimiser schedule",),
        applies_to="Any model trained through the regime hooks",
        references=(MOWL_GROUNDED,),
        standalone=False,
        introduced_with=("grounded",),
        implementation="mowl.training.schedules.LinearDecay",
        notes=(_NOTE + " GroundedEL's default decay (when lr_final is given) is the "
               "cosine one: the regime 'cosine-lr-decay' without a constant phase."),
    ),
    RegimeCard(
        key="idle-component-reseeding",
        name="Re-seeding of idle components",
        kind=RegimeKind.TRAINING,
        summary=("Every n epochs, after the optimisation step, components of "
                 "multi-component regions that cover nothing are moved to where they "
                 "are needed, and the optimiser's moment estimates are cleared. The "
                 "model decides which components are idle and where they go."),
        components=("structure adaptation",),
        applies_to="Models with reseed_components(epoch): GroundedEL, MultiBoxEL",
        references=(MOWL_GROUNDED,),
        standalone=False,
        introduced_with=("grounded",),
        used_by=("multiboxel",),
        implementation="mowl.training.structure.IdleComponentReseeding",
        limitations=("A discrete, non-differentiable step: the loss can rise right "
                     "after it.",),
        notes=_NOTE,
    ),
    RegimeCard(
        key="lagrangian-dual-weights",
        name="Loss terms as constraints with dual weights",
        kind=RegimeKind.TRAINING,
        summary=("Each loss term has a weight that grows, after every epoch, by a step "
                 "size times the amount by which the term is still violated (dual "
                 "ascent), up to a bound; a term that is slow to reach zero is not "
                 "hidden by larger ones."),
        components=("loss combination",),
        applies_to="Models that report their loss per term and expose the weights as "
                   "a dual_weights mapping: GroundedEL (terms fact, axioms, region)",
        references=(MOWL_GROUNDED,),
        standalone=False,
        introduced_with=("grounded",),
        implementation="mowl.training.constraints.LagrangianDualWeights",
        limitations=("Weights never decrease; with a positive lower bound of a term "
                     "(an unsatisfiable constraint) its weight goes to the bound.",),
        notes=_NOTE,
    ),
    RegimeCard(
        key="grounded-el-training",
        name="Training design of GroundedEL",
        kind=RegimeKind.TRAINING,
        summary=("The composition GroundedEL trains with when no regime is assigned: "
                 "temperature annealing; optionally a cosine or linear learning-rate "
                 "decay, idle-component re-seeding and Lagrangian dual weights. Full "
                 "batch over sampled rows, Adam, one step per epoch."),
        components=("schedule", "optimiser schedule", "structure adaptation",
                    "loss combination"),
        applies_to="mowl.models.GroundedEL; its parts apply more widely (see their "
                   "cards)",
        references=(MOWL_GROUNDED,),
        standalone=False,
        introduced_with=("grounded",),
        implementation="mowl.training.grounded.GroundedELTraining",
        limitations=(
            "GroundedEL's loss (element memberships, sampled region points, minimality) "
            "is part of the method and is not replaced by a regime's loss hook.",
            "Of the negative-sampling hooks, only those for object property assertions "
            "are used (when neg_weight > 0).",
        ),
        notes="Corrupted facts can be drawn and filtered by the shared negative "
              "regimes, e.g. ComposedRegime(KnownPositiveFilter(), "
              "GroundedELTraining.for_model(model)).",
    ),
)
