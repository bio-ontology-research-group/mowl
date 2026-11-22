"""The sparse membership reading of a canonical model; moved to
:mod:`mowl.representation.library.membership_structure` (the reading ``"membership"`` of
:mod:`mowl.representation.construction`). The names stay importable from here."""
from mowl.representation.library.membership_structure import (  # noqa: F401
    MEMBERSHIP_CODE_TAG,
    MembershipStructure,
    _sparse_bool_coo,
)

__all__ = ["MEMBERSHIP_CODE_TAG", "MembershipStructure"]
