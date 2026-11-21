"""Regime card: Corruption of each side separately."""

from mowl.catalog.card import RegimeCard, RegimeKind
from mowl.catalog.regimes._regime_refs import JACKERMEIER2024, YANG2025

REGIME = RegimeCard(
    key='per-column-corruption',
    assumption="closed world over stated axioms",
    name='Corruption of each side separately',
    kind=RegimeKind.TRAINING,
    summary=('Every positive gets one block of negatives per side: one with the left-hand side '
             'corrupted and one with the right-hand side corrupted (2K negatives per positive for'
             ' K per side).'),
    components=('negative sampling',),
    applies_to='EmbeddingELModel',
    references=(JACKERMEIER2024, YANG2025),
    standalone=False,
    introduced_with=('box2el',),
    used_by=('transbox',),
    implementation='mowl.training.negatives.PerColumnCorruption',
    limitations=(
        ('Uniform corruption can produce axioms that are entailed by the ontology (false '
         "negatives): Box2EL (Jackermeier et al. 2024, Section 3.2) notes that the procedure 'may"
         " occasionally generate false negative samples'; DELE (Mashkova et al., Section 5) shows"
         ' that for C ⊓ D ⊑ E with F ⊑ D in the ontology a random corruption is entailed in up to'
         ' 75% of the cases.'),
    ),
    notes=('Box2EL reference code (train.py, sample_negatives: a tail- and a head-corrupted copy '
           "of every NF3 axiom, num_neg times), although the paper (Section 3.2) says 'replacing "
           "either C or D'. TransBox, Section 4.3 ('replacing A with A' and B with B''). Built "
           "into mowl.models.TransBox for gci2; mowl's BoxSquaredEL corrupts only D (its num_negs"
           ' parameter is not used by the default sampler).'),
)
