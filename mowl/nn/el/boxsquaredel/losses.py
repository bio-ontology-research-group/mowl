"""The loss functions of the released Box²EL module (mOWL 2.1.0), deprecated.

The formulas are parts of :mod:`mowl.representation`; every function here computes the loss
with the module of the method on the tables it is given
(:func:`mowl.nn.el.released_losses.released_loss`), with the released signature, numbers and
shape.

.. deprecated:: 2.2.0
"""
from mowl.nn.el.released_losses import released_loss as _loss
from mowl.nn.el.released_losses import released_regularization as _regularization

_M = "boxsquaredel"


def class_assertion_loss(data, ind_center, ind_offset, class_center, class_offset, gamma,
                         neg=False):
    return _loss(_M, "class_assertion", data, neg, dict(
        ind_center=ind_center, ind_offset=ind_offset, class_center=class_center,
        class_offset=class_offset), gamma=gamma)


def object_property_assertion_loss(data, ind_center, ind_offset, head_center, head_offset,
                                   tail_center, tail_offset, bump_individuals, gamma, delta,
                                   neg=False):
    return _loss(_M, "object_property_assertion", data, neg, dict(
        ind_center=ind_center, ind_offset=ind_offset, head_center=head_center,
        head_offset=head_offset, tail_center=tail_center, tail_offset=tail_offset,
        bump_individuals=bump_individuals), gamma=gamma, delta=delta)


def gci0_loss(data, class_center, class_offset, gamma, neg=False):
    return _loss(_M, "gci0", data, neg, dict(class_center=class_center,
                                             class_offset=class_offset), gamma=gamma)


def gci0_bot_loss(data, class_offset):
    return _loss(_M, "gci0_bot", data, False, dict(class_offset=class_offset))


def gci1_loss(data, class_center, class_offset, gamma, neg=False):
    return _loss(_M, "gci1", data, neg, dict(class_center=class_center,
                                             class_offset=class_offset), gamma=gamma)


def gci1_bot_loss(data, class_center, class_offset, gamma, neg=False):
    return _loss(_M, "gci1_bot", data, neg, dict(class_center=class_center,
                                                 class_offset=class_offset), gamma=gamma)


def gci2_loss(data, class_center, class_offset, head_center, head_offset, tail_center,
              tail_offset, bump, gamma, delta, neg=False):
    return _loss(_M, "gci2", data, neg, dict(
        class_center=class_center, class_offset=class_offset, head_center=head_center,
        head_offset=head_offset, tail_center=tail_center, tail_offset=tail_offset, bump=bump),
        gamma=gamma, delta=delta)


def gci3_loss(data, class_center, class_offset, head_center, head_offset, tail_center,
              tail_offset, bump, gamma, neg=False):
    return _loss(_M, "gci3", data, neg, dict(
        class_center=class_center, class_offset=class_offset, head_center=head_center,
        head_offset=head_offset, tail_center=tail_center, tail_offset=tail_offset, bump=bump),
        gamma=gamma)


def gci3_bot_loss(data, head_offset):
    return _loss(_M, "gci3_bot", data, False, dict(head_offset=head_offset))


def reg_loss(bump, reg_factor):
    return _regularization(_M, dict(bump=bump), reg_factor=reg_factor)
