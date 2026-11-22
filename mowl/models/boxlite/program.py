"""The convex programme of BoxLitE (:mod:`mowl.representation.library.solved_boxlite`): the
variables are the tables of the parts, the constraints the grounding of the TBox, the
objective the signed-distance losses and the regularisers."""

from mowl.representation.library.solved_boxlite import (  # noqa: F401
    VARIANTS, BoxLitEProgram, BoxLitEProgramResult as ProgramResult,
    negative_concept_assertions)
from mowl.representation.library.solved_solver import default_solver, DEFAULT_CONIC_SOLVERS as DEFAULT_SOLVERS  # noqa: F401,E501

__all__ = ["VARIANTS", "BoxLitEProgram", "ProgramResult", "negative_concept_assertions",
           "default_solver", "DEFAULT_SOLVERS"]
