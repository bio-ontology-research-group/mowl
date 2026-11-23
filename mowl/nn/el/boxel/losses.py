"""The loss functions of the released BoxEL module (mOWL 2.1.0), deprecated.

The formulas are parts of :mod:`mowl.representation`; every function here computes the loss
with the module of the method on the tables it is given
(:func:`mowl.nn.el.released_losses.released_loss`), with the released signature, numbers and
shape.

.. deprecated:: 2.2.0
"""
from mowl.nn.el.released_losses import released_loss as _loss
from mowl.nn.el.released_losses import released_regularization as _regularization

_M = "boxel"


def regularization_loss(min_embed, delta_embed, log_scale=True):
    if not log_scale:
        raise NotImplementedError("Only the regulariser of the module (log_scale=True) is "
                                  "kept; see mowl.representation.library.boxrel_regularizers.")
    return _regularization(_M, dict(min_embed=min_embed, delta_embed=delta_embed))


def gci0_loss(data, min_embed, delta_embed, temperature, neg=False):
    return _loss(_M, "gci0", data, neg, dict(min_embed=min_embed, delta_embed=delta_embed),
                 temperature=temperature)


def gci0_bot_loss(*args, **kwargs):
    # as released: the loss of C ⊑ ⊥ is that of C ⊑ D with D the bottom class
    return gci0_loss(*args, **kwargs)


def gci1_loss(data, min_embed, delta_embed, temperature, neg=False):
    return _loss(_M, "gci1", data, neg, dict(min_embed=min_embed, delta_embed=delta_embed),
                 temperature=temperature)


def gci1_bot_loss(data, min_embed, delta_embed, temperature, neg=False):
    return _loss(_M, "gci1_bot", data, neg, dict(min_embed=min_embed, delta_embed=delta_embed),
                 temperature=temperature)


def gci2_loss(data, min_embed, delta_embed, relation_embed, scaling_embed, temperature,
              neg=False):
    return _loss(_M, "gci2", data, neg, dict(
        min_embed=min_embed, delta_embed=delta_embed, relation_embed=relation_embed,
        scaling_embed=scaling_embed), temperature=temperature)


def gci3_loss(data, min_embed, delta_embed, relation_embed, scaling_embed, temperature,
              neg=False):
    return _loss(_M, "gci3", data, neg, dict(
        min_embed=min_embed, delta_embed=delta_embed, relation_embed=relation_embed,
        scaling_embed=scaling_embed), temperature=temperature)


def gci3_bot_loss(*args, **kwargs):
    # as released: the loss of ∃r.C ⊑ ⊥ is that of ∃r.C ⊑ D with D the bottom class
    return gci3_loss(*args, **kwargs)


def class_assertion_loss(data, ind_embed, ind_delta_embed, min_embed, delta_embed, temperature,
                         neg=False):
    return _loss(_M, "class_assertion", data, neg, dict(
        ind_embed=ind_embed, ind_delta_embed=ind_delta_embed, min_embed=min_embed,
        delta_embed=delta_embed), temperature=temperature)


def object_property_assertion_loss(data, ind_embed, ind_delta_embed, relation_embed,
                                   scaling_embed, temperature, neg=False):
    return _loss(_M, "object_property_assertion", data, neg, dict(
        ind_embed=ind_embed, ind_delta_embed=ind_delta_embed, relation_embed=relation_embed,
        scaling_embed=scaling_embed), temperature=temperature)


def role_inclusion_loss(data, relation_embed, scaling_embed):
    return _loss(_M, "role_inclusion", data, False, dict(relation_embed=relation_embed,
                                                         scaling_embed=scaling_embed))


def role_chain_loss(data, relation_embed, scaling_embed):
    return _loss(_M, "role_chain", data, False, dict(relation_embed=relation_embed,
                                                     scaling_embed=scaling_embed))
