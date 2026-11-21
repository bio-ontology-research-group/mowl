"""Regime card: Uniform corruption of either side."""

from mowl.catalog.card import RegimeCard, RegimeKind
from mowl.catalog.regimes._regime_refs import (
    KULMANOV2019, MONDAL2021, XIONG2022, JACKERMEIER2024, SINGH2026, ZHANG2022)

REGIME = RegimeCard(
    key='either-side-corruption',
    assumption="closed world over stated axioms",
    name='Uniform corruption of either side',
    kind=RegimeKind.TRAINING,
    summary=("Every negative corrupts one side of the axiom, chosen uniformly per negative: C' ⊑ "
             "∃R.D or C ⊑ ∃R.D', C' ⊑ D or C ⊑ D', r(a', b) or r(a, b')."),
    components=('negative sampling',),
    applies_to='EmbeddingELModel',
    references=(KULMANOV2019, MONDAL2021, XIONG2022, JACKERMEIER2024, SINGH2026, ZHANG2022),
    standalone=False,
    introduced_with=('elem',),
    used_by=('emelpp', 'boxel', 'box2el', 'elkem', 'bcbox'),
    implementation='mowl.training.negatives.EitherSideCorruption',
    limitations=(
        ('Uniform corruption can produce axioms that are entailed by the ontology (false '
         "negatives): Box2EL (Jackermeier et al. 2024, Section 3.2) notes that the procedure 'may"
         " occasionally generate false negative samples'; DELE (Mashkova et al., Section 5) shows"
         ' that for C ⊓ D ⊑ E with F ⊑ D in the ontology a random corruption is entailed in up to'
         ' 75% of the cases.'),
        ('Applied to C ⊑ ∃R.D only (as in ELEmbeddings, ELBE, Box2EL), the other normal forms get'
         ' no negatives and can converge to trivial solutions (CatE, Zhapa-Camacho and Hoehndorf,'
         ' Section 4.4; DELE, Mashkova et al., Section 4).'),
    ),
    notes=("ELEmbeddings, Section 3.4 ('replacing either C or D', NF3 only); EmEL++, Section 3.3 "
           "(same); BoxEL, Section 4.5 (role assertions: 'replacing one of the head or tail "
           "entity') and Section 5.2 (C ⋢ D' or C' ⋢ D, weighted by φ = 0.05); Box2EL, Section "
           "3.2 as stated in the paper (the reference code uses 'per-column-corruption'); ELK-Em,"
           " Section 4.2 (for C ⊑ ∃r.D, 'C or D with equal probability'); binary-code boxes, "
           "Section 4.4 ('randomly corrupting either the source or target node', 32 per "
           'positive). Columns of the other normal forms are mOWL defaults (SIDE_COLUMNS).'),
)
