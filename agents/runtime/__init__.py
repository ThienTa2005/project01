"""Runtime multiagent system for diabetes decision support.

The training/evaluation agents in :mod:`agents` remain unchanged.  This
package implements the patient-facing workflow described in the project
architecture: validation, deep-learning inference, explicit rules,
knowledge retrieval, explanation, monitoring, and an optional RL simulation.
"""

from .coordinator import DecisionSupportCoordinator

__all__ = ["DecisionSupportCoordinator"]
