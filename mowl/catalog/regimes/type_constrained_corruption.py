"""Regime card: Type-constrained corruption."""

from mowl.catalog.card import RegimeCard, RegimeKind, Guarantee, GuaranteeKind
from mowl.catalog.regimes._regime_refs import MASHKOVA2024NESY, MASHKOVA2024DELE, MASHKOVA2026

REGIME = RegimeCard(
    key='type-constrained-corruption',
    assumption="closed world over stated axioms",
    name='Type-constrained corruption',
    kind=RegimeKind.TRAINING,
    summary=('An entity is replaced only by an entity of the same type (for example a protein by '
             'a protein, a GO class by a GO class), so that negatives are well-typed but false.'),
    components=('negative sampling',),
    applies_to='EmbeddingELModel on ontologies with a partition of entities into types',
    references=(MASHKOVA2024NESY, MASHKOVA2024DELE, MASHKOVA2026),
    standalone=False,
    introduced_with=(),
    used_by=(),
    implementation='mowl.training.negatives.TypeConstrainedCorruption',
    properties=(
        Guarantee(
            kind=GuaranteeKind.EXPRESSIVITY,
            statement=('Every replacement entity has the type of the entity it replaces (entities'
                       " without a type are replaced from the column's default pool)."),
            source='by construction (TypeConstrainedCorruption)',
            condition='by construction',
            verified_in_mowl='tests/training/test_regime_negatives.py',
        ),
    ),
    limitations=(
        ('Uniform corruption can produce axioms that are entailed by the ontology (false '
         "negatives): Box2EL (Jackermeier et al. 2024, Section 3.2) notes that the procedure 'may"
         " occasionally generate false negative samples'; DELE (Mashkova et al., Section 5) shows"
         ' that for C ⊓ D ⊑ E with F ⊑ D in the ontology a random corruption is entailed in up to'
         ' 75% of the cases.'),
    ),
    notes=('Not stated in the NeSy 2024 or DELE papers but done by their reference implementation'
           ' (geometric_embeddings, data_utils/dataloader.py: GO classes and protein classes are '
           'corrupted from separate pools). Mashkova and Hoehndorf (2026, Section 6.3) build '
           'their stratified sampler on type-constrained negative sampling (Krompaß et al. 2015, '
           'not read).'),
)
