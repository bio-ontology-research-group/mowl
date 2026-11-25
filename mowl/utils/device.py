"""Where a model computes."""

import torch as th


def default_device():
    """``cuda`` if a GPU is available, else ``cpu`` (a :class:`torch.device`)."""
    return th.device("cuda" if th.cuda.is_available() else "cpu")
