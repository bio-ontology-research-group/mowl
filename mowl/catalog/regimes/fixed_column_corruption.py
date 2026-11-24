"""Regime card: Uniform corruption of a fixed column."""

from mowl.catalog.card import RegimeCard, RegimeKind
from mowl.catalog.regimes._regime_refs import ZHAPA2023, SINGH2026, MASHKOVA2024DELE

REGIME = RegimeCard(
    key='fixed-column-corruption',
    name='Uniform corruption of a fixed column',
    kind=RegimeKind.TRAINING,
    summary=('Every negative replaces the entity in one fixed column of a training axiom (the '
             'right-hand side by default) by an entity drawn uniformly from the named classes, '
             'individuals or roles; K negatives per positive.'),
    components=('negative sampling',),
    applies_to='EmbeddingELModel and any model with per-axiom-type datasets',
    references=(ZHAPA2023, SINGH2026, MASHKOVA2024DELE),
    standalone=False,
    introduced_with=(),
    used_by=('cate', 'elkem'),
    implementation='mowl.training.negatives.FixedColumnCorruption',
    limitations=(
        ('Uniform corruption can produce axioms that are entailed by the ontology (false '
         "negatives): Box2EL (Jackermeier et al. 2024, Section 3.2) notes that the procedure 'may"
         " occasionally generate false negative samples'; DELE (Mashkova et al., Section 5) shows"
         ' that for C ⊓ D ⊑ E with F ⊑ D in the ontology a random corruption is entailed in up to'
         ' 75% of the cases.'),
        ('Applied to C ⊑ ∃R.D only (as in ELEmbeddings, ELBE, Box2EL), the other normal forms get'
         ' no negatives and can converge to trivial solutions (CatE, Zhapa-Camacho and Hoehndorf,'
         ' Section 4.4; DELE, Mashkova et al., Section 4).'),
        ('Only one side is ever corrupted, so the embedding is never told that a wrong left-hand '
         'side is false (ELK-Em corrupts D for C ⊑ D but C or D for C ⊑ ∃r.D, Singh et al. 2026, '
         'Section 4.2).'),
    ),
    notes=("The default sampler of mOWL's EmbeddingELModel (one negative per positive, right-hand"
           ' side, named classes only); attaching the regime with default arguments reproduces '
           'it. CatE, Section 3.2, Eq. 7 (replace B of A ⊑ B, uniform; 1, 2 or 4 negatives); '
           "ELK-Em, Section 4.2 (k_neg corruptions of D, 'without filtering corruptions that "
           "happen to be entailed'); DELE, Section 4 (one column per normal form; DELE's own "
           "regime is 'dele-all-normal-form-negatives'). mOWL's PPI example models restrict the "
           'pool of gci2 to proteins: FixedColumnCorruption(pools=...).'),
)
