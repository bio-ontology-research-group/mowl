allowed_diff = 1e-6

def auc_from_mr(mr, num_entities):
    auc = 1 - (mr-1) / (num_entities -1)
    return auc
