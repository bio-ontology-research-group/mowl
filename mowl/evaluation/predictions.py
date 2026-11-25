"""Precision, recall and AUC of scored axioms (mOWL 2.1.0).

.. deprecated:: 2.2.0
   Use the evaluation regimes of :mod:`mowl.evaluation` (and :mod:`mowl.evaluation.metrics`).
"""
import warnings

import numpy as np
from mowl.evaluation.metrics import auc, precision_and_recall_at_k, precision, recall


def evaluate_predictions(true_axioms, predictions, ks, pos_label=1):
    """Method that evaluates precision, recall and AUC for predictions of axioms.

    :param true_axioms: Axioms in the true positives set
    :type true_axioms: set
    :param predictions: Dictionary of predictions of the form prediction -> score
    :type: dict

    .. deprecated:: 2.2.0
       Use the evaluation regimes of :mod:`mowl.evaluation`.
    """
    warnings.warn("mowl.evaluation.predictions.evaluate_predictions is deprecated: use the "
                  "evaluation regimes of mowl.evaluation.", DeprecationWarning, stacklevel=2)

    y_scores = []
    y_true = []
    true_axioms = set(true_axioms)

    if pos_label != 1 and pos_label != 0:
        raise ValueError("Pos label must be either 0 or 1")

    invert = False
    if pos_label == 0:
        invert = True

    for name, k in predictions.items():
        if name in true_axioms:
            y_true.append(1)
        else:
            y_true.append(0)

        if invert:
            k = 1 - k
        y_scores.append(k)

    pos_label = 1

    y_scores = np.array(y_scores)
    y_true = np.array(y_true)

    metrics = dict()

    auc_value, threshold = auc(y_true, y_scores, pos_label=pos_label)
    metrics["auc"] = auc_value

    y_scores = [1 if x > threshold else 0 for x in y_scores]

    prec = precision(y_true, y_scores, pos_label=pos_label)
    rec = recall(y_true, y_scores, pos_label=pos_label)
    metrics["precision"] = prec
    metrics["recall"] = rec

    for k in ks:
        prec, rec = precision_and_recall_at_k(y_true, y_scores, k, len(true_axioms),
                                              pos_label=pos_label)
        metrics[f"prec@{k}"] = prec
        metrics[f"rec@{k}"] = rec

    new_metrics = {}
    for k, v in sorted(metrics.items()):
        new_metrics[k] = v

    return new_metrics
