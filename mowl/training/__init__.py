"""Training regimes (:mod:`mowl.training.regime`): the DELE regimes and the shared
deductive-closure utility (:mod:`mowl.training.dele`, :mod:`mowl.training.closure`), and
the regimes of the literature (negatives, losses, positives, schedules, model selection,
batching)."""

from mowl.training.regime import TrainingRegime, ComposedRegime
from mowl.training.closure import DeductiveClosure, NORMAL_FORMS
from mowl.training.dele import (AllNormalFormNegatives, ClosureFilteredNegatives,
                                ClosurePositives, DELECodeLosses, DELECodeTraining,
                                dele_regime)
from mowl.training.negatives import (CorruptionNegatives, FixedColumnCorruption,
                                     EitherSideCorruption, PerColumnCorruption,
                                     TypeConstrainedCorruption, KnownPositiveFilter,
                                     SiblingHardNegatives, ContradictionNegatives,
                                     FixedNegatives)
from mowl.training.initialisation import JointNormalisedInitialisation
from mowl.training.loss_forms import (UnboundedNegativeLoss, PerAxiomNormPenalty,
                                      Box2ELBaselineLosses, EmELppCodeObjective)
from mowl.training.losses import (SumLoss, MSETargetLoss, PairwiseMarginRankingLoss,
                                  LogSigmoidLoss, BPRLoss, LogSumExpContrastiveLoss,
                                  SoftmaxEnergyContrastiveLoss, PearsonCorrelationLoss,
                                  ProbabilisticBCELoss, NormalFormFrequencyWeighting)
from mowl.training.positives import (TransitiveReductionPositives,
                                     TransitiveClosurePositives, ValidationAxiomsAsPositives)
from mowl.training.schedules import StepDecay, ConstantThenCosine, CyclicLearningRate, \
    ReduceOnPlateau
from mowl.training.selection import (ValidationSelection, ValidationRankSelection,
                                     ValidationF1Selection, validation_f1)
from mowl.training.batching import PerNormalFormMiniBatches
from mowl.training.margin import SlackMargin, StrictMargin, margin_semantics
from mowl.training.reference import (Box2ELCodeTraining, TransBoxCodeTraining,
                                     ELKEmCodeTraining, EmELppCodeTraining,
                                     RegDCodeTraining)
from mowl.training.cate import CateCodeTraining

__all__ = [
    "TrainingRegime", "ComposedRegime",
    "SlackMargin", "StrictMargin", "margin_semantics",
    "Box2ELCodeTraining", "TransBoxCodeTraining", "ELKEmCodeTraining",
    "EmELppCodeTraining", "EmELppCodeObjective", "ValidationAxiomsAsPositives",
    "RegDCodeTraining",
    "CateCodeTraining",
    "DeductiveClosure", "NORMAL_FORMS",
    "DELECodeLosses", "DELECodeTraining",
    "AllNormalFormNegatives", "ClosureFilteredNegatives", "ClosurePositives", "dele_regime",
    "CorruptionNegatives", "FixedColumnCorruption", "EitherSideCorruption",
    "PerColumnCorruption", "TypeConstrainedCorruption", "KnownPositiveFilter",
    "SiblingHardNegatives", "ContradictionNegatives", "FixedNegatives",
    "JointNormalisedInitialisation", "UnboundedNegativeLoss", "PerAxiomNormPenalty",
    "Box2ELBaselineLosses",
    "SumLoss", "MSETargetLoss", "PairwiseMarginRankingLoss", "LogSigmoidLoss", "BPRLoss",
    "LogSumExpContrastiveLoss", "SoftmaxEnergyContrastiveLoss",
    "PearsonCorrelationLoss", "ProbabilisticBCELoss",
    "NormalFormFrequencyWeighting",
    "TransitiveReductionPositives", "TransitiveClosurePositives",
    "StepDecay", "ConstantThenCosine", "CyclicLearningRate", "ReduceOnPlateau",
    "ValidationSelection", "ValidationRankSelection", "ValidationF1Selection",
    "validation_f1",
    "PerNormalFormMiniBatches",
]
