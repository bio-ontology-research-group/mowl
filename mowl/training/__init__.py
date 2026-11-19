"""Training regimes (:mod:`mowl.training.regime`): the DELE regimes and the shared
deductive-closure utility (:mod:`mowl.training.dele`, :mod:`mowl.training.closure`), and
the regimes of the literature (negatives, losses, positives, schedules, model selection,
batching)."""

from mowl.training.regime import TrainingRegime, ComposedRegime
from mowl.training.closure import DeductiveClosure, NORMAL_FORMS
from mowl.training.dele import (AllNormalFormNegatives, ClosureFilteredNegatives,
                                ClosurePositives, dele_regime)
from mowl.training.negatives import (CorruptionNegatives, FixedColumnCorruption,
                                     EitherSideCorruption, PerColumnCorruption,
                                     TypeConstrainedCorruption, KnownPositiveFilter,
                                     SiblingHardNegatives, ContradictionNegatives)
from mowl.training.losses import (SumLoss, MSETargetLoss, PairwiseMarginRankingLoss,
                                  LogSigmoidLoss, BPRLoss, LogSumExpContrastiveLoss,
                                  PearsonCorrelationLoss, ProbabilisticBCELoss,
                                  NormalFormFrequencyWeighting)
from mowl.training.positives import TransitiveReductionPositives, TransitiveClosurePositives
from mowl.training.schedules import StepDecay, ConstantThenCosine, CyclicLearningRate, \
    ReduceOnPlateau
from mowl.training.selection import ValidationSelection, ValidationRankSelection
from mowl.training.batching import PerNormalFormMiniBatches

__all__ = [
    "TrainingRegime", "ComposedRegime",
    "DeductiveClosure", "NORMAL_FORMS",
    "AllNormalFormNegatives", "ClosureFilteredNegatives", "ClosurePositives", "dele_regime",
    "CorruptionNegatives", "FixedColumnCorruption", "EitherSideCorruption",
    "PerColumnCorruption", "TypeConstrainedCorruption", "KnownPositiveFilter",
    "SiblingHardNegatives", "ContradictionNegatives",
    "SumLoss", "MSETargetLoss", "PairwiseMarginRankingLoss", "LogSigmoidLoss", "BPRLoss",
    "LogSumExpContrastiveLoss", "PearsonCorrelationLoss", "ProbabilisticBCELoss",
    "NormalFormFrequencyWeighting",
    "TransitiveReductionPositives", "TransitiveClosurePositives",
    "StepDecay", "ConstantThenCosine", "CyclicLearningRate", "ReduceOnPlateau",
    "ValidationSelection", "ValidationRankSelection",
    "PerNormalFormMiniBatches",
]
