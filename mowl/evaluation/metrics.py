from sklearn.utils import column_or_1d
from sklearn.utils.multiclass import type_of_target
from sklearn import metrics as skm
import numpy as np


def precision_and_recall_at_k(y_true, y_score, k, nb_positives, pos_label=1):

    y_true_type = type_of_target(y_true)
    if not (y_true_type == "binary"):
        raise ValueError("y_true must be a binary column.")

    # Makes this compatible with various array types
    y_true_arr = column_or_1d(y_true)
    y_score_arr = column_or_1d(y_score)

    y_true_arr = y_true_arr == pos_label

    desc_sort_order = np.argsort(y_score_arr)[::-1]
    y_true_sorted = y_true_arr[desc_sort_order]

    predicted_true_positives = y_true_sorted[:k].sum()

    return predicted_true_positives / k, predicted_true_positives / nb_positives


def auc(y_true, y_score, pos_label=1):
    fpr, tpr, thresholds = skm.roc_curve(y_true, y_score, pos_label=pos_label)
    gmeans = np.sqrt(tpr * (1 - fpr))
    # locate the index of the largest g-mean
    ix = np.argmax(gmeans)
    return skm.auc(fpr, tpr), thresholds[ix]


def precision(y_true, y_score, pos_label=1):
    return skm.precision_score(y_true, y_score, pos_label=pos_label)


def recall(y_true, y_score, pos_label=1):
    return skm.recall_score(y_true, y_score, pos_label=pos_label)
