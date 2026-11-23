"""The loss functions of the released ELEmbeddings module (mOWL 2.1.0), deprecated.

The formulas are parts of :mod:`mowl.representation`; every function here computes the loss
with the module of the method on the tables it is given
(:func:`mowl.nn.el.released_losses.released_loss`), with the released signature, numbers and
shape.

.. deprecated:: 2.2.0
"""
from mowl.nn.el.released_losses import released_loss as _loss
from mowl.nn.el.released_losses import released_regularization as _regularization

_M = "elem"


def class_assertion_loss(data, ind_embed, ind_rad, class_embed, class_rad, margin, neg=False):
    return _loss(_M, "class_assertion", data, neg, dict(
        ind_embed=ind_embed, ind_rad=ind_rad, class_embed=class_embed, class_rad=class_rad),
        margin=margin)


def object_property_assertion_loss(data, ind_embed, ind_rad, rel_embed, margin, neg=False):
    return _loss(_M, "object_property_assertion", data, neg, dict(
        ind_embed=ind_embed, ind_rad=ind_rad, rel_embed=rel_embed), margin=margin)


def gci0_loss(data, class_embed, class_rad, margin, neg=False):
    return _loss(_M, "gci0", data, neg, dict(class_embed=class_embed, class_rad=class_rad),
                 margin=margin)


def gci0_bot_loss(data, class_rad, neg=False):
    return _loss(_M, "gci0_bot", data, neg, dict(class_rad=class_rad))


def gci1_loss(data, class_embed, class_rad, margin, neg=False):
    return _loss(_M, "gci1", data, neg, dict(class_embed=class_embed, class_rad=class_rad),
                 margin=margin)


def gci1_bot_loss(data, class_embed, class_rad, margin, neg=False):
    return _loss(_M, "gci1_bot", data, neg, dict(class_embed=class_embed, class_rad=class_rad),
                 margin=margin)


def gci2_loss(data, class_embed, class_rad, rel_embed, margin, neg=False):
    return _loss(_M, "gci2", data, neg, dict(class_embed=class_embed, class_rad=class_rad,
                                             rel_embed=rel_embed), margin=margin)


def gci3_loss(data, class_embed, class_rad, rel_embed, margin, neg=False):
    return _loss(_M, "gci3", data, neg, dict(class_embed=class_embed, class_rad=class_rad,
                                             rel_embed=rel_embed), margin=margin)


def gci3_bot_loss(data, class_rad, neg=False):
    return _loss(_M, "gci3_bot", data, neg, dict(class_rad=class_rad))


def regularization_loss(class_embed, ind_embed=None, reg_norm=1):
    return _regularization(_M, dict(class_embed=class_embed, ind_embed=ind_embed),
                           reg_norm=reg_norm)
