"""Logic-level descriptions shared by methods, reasoners, regimes and benchmarks.

* :mod:`mowl.logic.expressivity`: description logic expressivity in standard notation, the
  expressivity of an ontology, and the check of an ontology against what a method supports.
* :mod:`mowl.logic.tasks`: the standard inference tasks, the completion tasks of the
  knowledge graph and knowledge base completion literature, and how they relate.

Importing this package does not need the JVM.
"""

from mowl.logic.expressivity import (Construct, Expressivity, ExpressivityReport, NAMED,
                                     ABOX, axiom_constructs, ontology_expressivity,
                                     check_ontology)
from mowl.logic.tasks import (Box, LogicalStatus, FaithfulnessCriterion, InferenceTask,
                              CompletionTask, Target, WorldAssumption,
                              TASKS, COMPLETION_TASKS, KGC, KBC,
                              get_task, get_completion_task)

__all__ = [
    "Construct", "Expressivity", "ExpressivityReport", "NAMED", "ABOX",
    "axiom_constructs", "ontology_expressivity", "check_ontology",
    "Box", "LogicalStatus", "FaithfulnessCriterion", "InferenceTask", "CompletionTask",
    "Target", "WorldAssumption", "TASKS", "COMPLETION_TASKS", "KGC", "KBC", "get_task", "get_completion_task",
]
