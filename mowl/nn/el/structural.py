"""Old import path of :mod:`mowl.representation.library.structural_tables`.

The structural tables (``DefinitionTable``, ``ExistenceTargets``, ``WitnessTargets``,
``Regions``, ``inside_measure``, ``build_structural_tables``) are general parts and live
in :mod:`mowl.representation.library.structural_tables`; this module re-exports them.
"""
from mowl.representation.library.structural_tables import *  # noqa: F401,F403
from mowl.representation.library.structural_tables import (  # noqa: F401
    Regions, inside_measure, DefinitionTable, slot_components, WitnessTargets,
    ExistenceTargets, build_structural_tables, _extract_normal_form_rows)
