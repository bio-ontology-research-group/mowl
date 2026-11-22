"""The training loop asks a module for a negative loss only where the loss regime needs one."""
from mowl.training import ComposedRegime, FixedColumnCorruption, PearsonCorrelationLoss
from mowl.training.regime import TrainingRegime


def test_score_loss_scores_its_negatives_like_positives_for_its_forms_only():
    pearson = PearsonCorrelationLoss(normal_forms=("gci0",))
    assert pearson.scores_negatives_like_positives("gci0") is True
    assert pearson.scores_negatives_like_positives("gci2") is None
    by_module = PearsonCorrelationLoss(neg_scores="module")
    assert by_module.scores_negatives_like_positives("gci0") is False


def test_composed_regime_asks_the_regime_that_decides_the_loss():
    corrupt = FixedColumnCorruption(columns={"gci0": 1}, require_negative_loss=False)
    composed = ComposedRegime(corrupt, PearsonCorrelationLoss())
    assert composed.scores_negatives_like_positives("gci0") is True
    assert ComposedRegime(TrainingRegime()).scores_negatives_like_positives("gci0") is False
