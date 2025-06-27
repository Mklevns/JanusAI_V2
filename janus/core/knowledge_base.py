# janus/core/knowledge_base.py

"""
# janus/core/knowledge_base.py
This module defines the enhanced `KnowledgeBase` class, which serves as a
central repository for all environment specifications and solver performance data.
It is designed to support the meta-learning capabilities of the `ArchitectAgent`
by providing a structured and comprehensive view of the system's history and performance.
"""


from __future__ import annotations
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import time
import numpy as np

# Configure logging for KnowledgeBase operations
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class PerformanceRecord:
    """A record of a single performance evaluation for a solver on a spec."""
    solver_id: str
    score: float
    timestamp: float = field(default_factory=time.time)
    # In a real implementation, this might be a path to a saved model checkpoint
    solver_policy_ref: Optional[str] = None 

@dataclass
class KnowledgeEntry:
    """A structured entry in the KnowledgeBase for a single environment spec.

    This acts as a "lab notebook" entry for each environment, tracking its
    origin, parameters, and a full history of all solver attempts.
    """
    spec_id: str
    parent_id: Optional[str]
    parameters: Dict[str, Any]
    status: str = "ACTIVE"  # e.g., ACTIVE, PRUNED, MASTERED
    created_timestamp: float = field(default_factory=time.time)
    novelty_score: Optional[float] = None
    performance_history: List[PerformanceRecord] = field(default_factory=list)
    best_performance: Optional[PerformanceRecord] = None

class KnowledgeBase:
    """
    Central store for all data related to environment specs and solver performance.

    This enhanced KnowledgeBase acts as the system's long-term memory, capturing
    the rich, contextual data needed for meta-learning by the ArchitectAgent.
    It tracks not just the best score, but the full performance history, environment
    lineage, and solver policies.
    """

    def __init__(self) -> None:
        """Initializes an empty KnowledgeBase."""
        self._entries: Dict[str, KnowledgeEntry] = {}
        logging.info("Enhanced KnowledgeBase initialized.")

    def register_spec(self, spec: 'BaseEnvironmentSpec', parent_id: Optional[str] = None) -> None:
        """
        Registers a new environment specification in the KnowledgeBase.

        Args:
            spec (BaseEnvironmentSpec): The environment specification object.
            parent_id (Optional[str]): The ID of the spec it was mutated from.
        """
        if spec.spec_id in self._entries:
            logging.warning(f"Spec '{spec.spec_id}' already registered. Ignoring.")
            return
        
        entry = KnowledgeEntry(
            spec_id=spec.spec_id,
            parent_id=parent_id,
            parameters=spec.get_parameter_values()
        )
        self._entries[spec.spec_id] = entry
        logging.info(f"Registered new spec '{spec.spec_id}' with parent '{parent_id}'.")

    def record_performance(self, solver_id: str, spec_id: str, score: float, solver_policy_ref: Optional[str] = None) -> None:
        """
        Records a solver's performance on an environment spec and updates history.

        Args:
            solver_id (str): Identifier for the solver (e.g., "PPO_01").
            spec_id (str): Identifier for the environment specification.
            score (float): The score or reward achieved by the solver.
            solver_policy_ref (Optional[str]): A reference (e.g., file path) to the
                solver's policy state_dict that achieved this score.
        """
        if spec_id not in self._entries:
            logging.error(f"Cannot record score for unregistered spec '{spec_id}'.")
            return

        entry = self._entries[spec_id]
        record = PerformanceRecord(solver_id, score, solver_policy_ref=solver_policy_ref)
        entry.performance_history.append(record)

        if entry.best_performance is None or score > entry.best_performance.score:
            entry.best_performance = record
            logging.info(f"Updated best score for spec '{spec_id}' to {score} (solver: {solver_id}).")

    def get_best_scores(self) -> Dict[str, float]:
        """
        Retrieves the best scores achieved for each environment spec.

        Returns:
            Dict[str, float]: Mapping from spec_id to the highest score recorded.
        """
        logging.info("Fetching best scores for all specs.")
        return {
            spec_id: entry.best_performance.score
            for spec_id, entry in self._entries.items()
            if entry.best_performance is not None
        }

    def get_entry(self, spec_id: str) -> Optional[KnowledgeEntry]:
        """Retrieves the full knowledge entry for a given spec ID."""
        return self._entries.get(spec_id)

    def update_spec_status(self, spec_id: str, status: str) -> None:
        """
        Updates the status of a spec (e.g., to PRUNED or MASTERED).

        Args:
            spec_id (str): The ID of the spec to update.
            status (str): The new status.
        """
        if spec_id in self._entries:
            self._entries[spec_id].status = status
            logging.info(f"Updated status for spec '{spec_id}' to '{status}'.")
        else:
            logging.warning(f"Attempted to update status for non-existent spec '{spec_id}'.")


if __name__ == '__main__':
    # This import would be `from janus.core.base_environment_spec import BaseEnvironmentSpec`
    # For demonstration, we'll redefine a minimal version here.
    class MockBaseEnvironmentSpec:
        def __init__(self, spec_id, params):
            self.spec_id = spec_id
            self._params = params
        def get_parameter_values(self):
            return self._params
            
    print("--- Demonstrating Enhanced KnowledgeBase Functionality ---")
    
    # 1. Initialize KB and register the root environment
    kb = KnowledgeBase()
    root_spec = MockBaseEnvironmentSpec("root_env", {"gravity": 9.8})
    kb.register_spec(spec=root_spec)
    
    # 2. Simulate some solver activity on the root environment
    kb.record_performance("PPO_01", "root_env", 250.0, "policies/ppo_01_v1.pt")
    kb.record_performance("PPO_01", "root_env", 280.0, "policies/ppo_01_v2.pt")
    kb.record_performance("ES_01", "root_env", 295.0, "policies/es_01_v1.pt")

    # 3. Register a new spec that was mutated from the root
    child_spec = MockBaseEnvironmentSpec("child_env_A", {"gravity": 10.3})
    kb.register_spec(spec=child_spec, parent_id="root_env")

    # 4. Record performance for the child spec
    kb.record_performance("PPO_02", "child_env_A", 120.0, "policies/ppo_02_v1.pt")
    kb.record_performance("PPO_02", "child_env_A", 155.0, "policies/ppo_02_v2.pt")
    
    # 5. ArchitectAgent checks performance and decides to mark root as MASTERED
    kb.update_spec_status("root_env", "MASTERED")

    # 6. Inspect the rich data stored for the child environment
    print("\n--- Inspecting Knowledge Entry for 'child_env_A' ---")
    child_entry = kb.get_entry("child_env_A")
    if child_entry:
        print(f"  Spec ID: {child_entry.spec_id}")
        print(f"  Parent ID: {child_entry.parent_id}")
        print(f"  Status: {child_entry.status}")
        print(f"  Parameters: {child_entry.parameters}")
        print(f"  Best Performance Record: {child_entry.best_performance}")
        print(f"  Full Performance History ({len(child_entry.performance_history)} records):")
        for record in child_entry.performance_history:
            print(f"    - {record}")

    # 7. The ArchitectAgent can still get a simple high-score list for pruning decisions
    print("\n--- Architect getting best scores for minimal criterion check ---")
    best_scores = kb.get_best_scores()
    print(best_scores)

    assert best_scores.get("child_env_A") == 155.0
    print("\nThis enhanced KB now captures the rich, historical, and relational data")
    print("required for the ArchitectAgent to perform advanced meta-learning.")
