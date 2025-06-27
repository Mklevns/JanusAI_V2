# janus/agents/architect_agent.py

"""
This module defines the `ArchitectAgent`, which orchestrates the open-ended discovery process
by managing environments and solvers. 
"""

import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, Tuple, Any

# Assume these are correctly located in the JanusAI framework
from janus.agents.environment_host import EnvironmentHost
from janus.core.base_environment_spec import BaseEnvironmentSpec
from janus.core.knowledge_base import KnowledgeBase, KnowledgeEntry, PerformanceRecord

# Use structured logging for better filtering and analysis
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - Epoch %(epoch)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Suggestion 4 & 7: Decouple Transfer Logic into a Strategy Protocol ---
class TransferSelector(Protocol):
    """A protocol for defining different transfer selection strategies."""
    def select(
        self,
        target_spec_id: str,
        all_entries: Dict[str, KnowledgeEntry],
        log_extra: Dict
    ) -> List[PerformanceRecord]:
        """Selects candidate policies from other specs to transfer to the target."""
        ...

class BestForeignPolicySelector(TransferSelector):
    """A simple strategy that selects the single best policy from any other spec."""
    def select(
        self,
        target_spec_id: str,
        all_entries: Dict[str, KnowledgeEntry],
        log_extra: Dict
    ) -> List[PerformanceRecord]:
        potential_transfers = [
            entry.best_performance for id, entry in all_entries.items()
            if id != target_spec_id and entry.best_performance and entry.status == "ACTIVE"
        ]
        if not potential_transfers:
            return []
        
        best_foreign_policy = max(potential_transfers, key=lambda p: p.score)
        logger.debug(f"Selected best foreign policy from spec '{best_foreign_policy.solver_id}' "
                     f"for target '{target_spec_id}'", extra=log_extra)
        return [best_foreign_policy]

# --- Suggestion 2: Decouple Solver Execution into a Protocol ---
class SolverRunner(Protocol):
    """A protocol for any object that can run a solver in an environment."""
    def run(self, spec_id: str, solver_id: str, policy_ref: Optional[str] = None) -> float:
        ...

# --- Suggestion 3: Use a typed schema for configuration ---
@dataclass
class ArchitectConfig:
    """Typed configuration for the ArchitectAgent."""
    reproduction_threshold: float
    repro_num_children: int
    max_population: int
    transfer_frequency: int
    minimal_criterion: Dict[str, float]
    solver_failure_score: float = -1.0e9 # A specific score to log on failure

class ArchitectAgent:
    """Orchestrates the open-ended discovery process by managing environments and solvers."""

    def __init__(
        self,
        host: EnvironmentHost,
        kb: KnowledgeBase,
        solver_runner: SolverRunner,
        transfer_selector: TransferSelector,
        config: ArchitectConfig
    ):
        self.host = host
        self.kb = kb
        self.solver_runner = solver_runner
        self.transfer_selector = transfer_selector
        self.config = config
        logger.info("ArchitectAgent initialized with injected dependencies and typed config.")

    # --- Suggestion 1: Split Epoch into public methods ---
    def run_epoch(self, epoch: int) -> None:
        """Executes one full cycle of the POET-inspired discovery loop."""
        log_extra = {'epoch': epoch}
        logger.info("Starting Architect Epoch.", extra=log_extra)

        self.reproduce(log_extra)
        self.evaluate_active_specs(log_extra)
        self.prune_population(log_extra)

        if epoch > 0 and epoch % self.config.transfer_frequency == 0:
            self.attempt_transfers(log_extra)
            self.evaluate_active_specs(log_extra, "Post-Transfer Evaluation")

        logger.info("Finished Architect Epoch.", extra=log_extra)

    def reproduce(self, log_extra: Dict) -> None:
        """Selects parent specs and commands the host to create new children."""
        parent_ids = self._select_parents_for_reproduction()
        if not parent_ids:
            logger.warning("No eligible parents found for reproduction.", extra=log_extra)
            return

        for parent_id in parent_ids:
            new_ids = self.host.speciate(parent_id, num_children=self.config.repro_num_children)
            # Register new specs in the KB
            parent_spec = self.host.get_spec(parent_id)
            for child_id in new_ids:
                child_spec = self.host.get_spec(child_id)
                self.kb.register_spec(child_spec, parent_id=parent_id)

    def evaluate_active_specs(self, log_extra: Dict, context: str = "Main Evaluation") -> None:
        """Dispatches solver jobs in parallel for all active, unevaluated specs."""
        logger.info(f"Starting {context} Phase.", extra=log_extra)
        active_spec_ids = [sid for sid, s in self.host.get_all_specs().items() if self.kb.get_entry(sid).status == "ACTIVE"]

        # Suggestion 2: Parallelize solver dispatch
        with ThreadPoolExecutor() as executor:
            future_to_spec = {
                executor.submit(self.solver_runner.run, spec_id=sid, solver_id="PPO_main"): sid
                for sid in active_spec_ids
            }
            for future in as_completed(future_to_spec):
                spec_id = future_to_spec[future]
                try:
                    score = future.result()
                    self.kb.record_performance("PPO_main", spec_id, score, f"policies/PPO_main_{spec_id}.pt")
                except Exception as e:
                    # Suggestion 5: Guard and Recover from Solver Failures
                    logger.error(f"Solver failed for spec '{spec_id}': {e}", extra=log_extra)
                    self.kb.record_performance("PPO_main", spec_id, self.config.solver_failure_score, None)

    def prune_population(self, log_extra: Dict) -> None:
        """Prunes population based on minimal criterion and capacity."""
        self._prune_by_minimal_criterion(log_extra)
        self._prune_by_capacity(log_extra)

    def attempt_transfers(self, log_extra: Dict) -> None:
        """Identifies and tests potential policy transfers between specs."""
        logger.info("Attempting policy transfers.", extra=log_extra)
        active_specs = self.host.get_all_specs()
        all_entries = {sid: self.kb.get_entry(sid) for sid in active_specs.keys()}
        
        with ThreadPoolExecutor() as executor:
            for target_id in active_specs.keys():
                # Use the injected strategy to select candidates
                candidates = self.transfer_selector.select(target_id, all_entries, log_extra)
                for candidate in candidates:
                    executor.submit(
                        self.solver_runner.run,
                        spec_id=target_id,
                        solver_id="transfer_solver",
                        policy_ref=candidate.solver_policy_ref
                    )
            # We don't need the results here, they will be picked up in the next evaluate phase

    def _select_parents_for_reproduction(self) -> List[str]:
        # Implementation from previous version...
        return [spec_id for spec_id, score in self.kb.get_best_scores().items() if score >= self.config.reproduction_threshold and self.kb.get_entry(spec_id).status == "ACTIVE"]

    def _prune_by_minimal_criterion(self, log_extra: Dict) -> None:
        # Implementation from previous version...
        _, to_prune = self.host.apply_minimal_criterion(self.kb.get_best_scores(), self.config.minimal_criterion['min_score'], self.config.minimal_criterion['max_score'])
        self.host.prune(to_prune)
        for spec_id in to_prune: self.kb.update_spec_status(spec_id, "PRUNED_MC")

    def _prune_by_capacity(self, log_extra: Dict) -> None:
        # Suggestion 8: Ensure Deterministic Pruning Order
        all_specs = self.host.get_all_specs().values()
        active_entries = [self.kb.get_entry(s.spec_id) for s in all_specs if self.kb.get_entry(s.spec_id).status == "ACTIVE"]
        
        if len(active_entries) > self.config.max_population:
            num_to_prune = len(active_entries) - self.config.max_population
            active_entries.sort(key=lambda e: (e.created_timestamp, e.spec_id)) # Add spec_id as tie-breaker
            to_prune_age = [e.spec_id for e in active_entries[:num_to_prune]]
            
            self.host.prune(to_prune_age)
            for spec_id in to_prune_age: self.kb.update_spec_status(spec_id, "PRUNED_AGE")
            logger.info(f"Pruned {num_to_prune} oldest specs due to capacity.", extra=log_extra)

if __name__ == '__main__':
    """Demonstrates the refactored ArchitectAgent with parallel execution and failure recovery."""
    print("--- Demonstrating Hardened ArchitectAgent ---")

    # A more advanced mock solver that can fail
    class MockParallelSolverRunner(SolverRunner):
        def run(self, spec_id: str, solver_id: str, policy_ref: str | None = None) -> float:
            time.sleep(0.1) # Simulate work
            if "fail" in spec_id:
                raise ValueError("Solver simulation crashed!")
            complexity = sum(ord(c) for c in spec_id if c.isdigit())
            base_score = max(0, 250 - complexity)
            transfer_bonus = 80 if policy_ref else 0
            return base_score + transfer_bonus + (hash(spec_id) % 20)

    # 1. Setup ecosystem
    config = ArchitectConfig(
        reproduction_threshold=200.0, repro_num_children=3,
        max_population=8, transfer_frequency=1,
        minimal_criterion={'min_score': 100.0, 'max_score': 290.0}
    )
    root_spec = BaseEnvironmentSpec("root_spec", {"p1": {"initial_value": 0, "mutation_step": 1}})
    kb = KnowledgeBase(); kb.register_spec(root_spec)
    host = EnvironmentHost(initial_spec=root_spec)
    solver_runner = MockParallelSolverRunner()
    transfer_selector = BestForeignPolicySelector()
    architect = ArchitectAgent(host, kb, solver_runner, transfer_selector, config)

    # Kickstart the process
    kb.record_performance("init", "root_spec", 210.0)

    # 2. Run the main loop
    for i in range(1, 3):
        architect.run_epoch(epoch=i)
        
        # Manually create a failing spec for demonstration
        if i == 1:
            host.speciate("root_spec", 1)
            # The new ID will be pseudo-random, let's find it.
            new_spec_id = [sid for sid in host.get_all_specs() if sid not in ["root_spec"]][0]
            # Rename it to something that will cause the mock solver to fail
            failing_spec = host.get_all_specs().pop(new_spec_id)
            failing_spec.spec_id = "spec_that_will_fail"
            host.get_all_specs()[failing_spec.spec_id] = failing_spec
            kb.register_spec(failing_spec, parent_id="root_spec")
    
    # 3. Print final state
    print("\n--- Final State of KnowledgeBase ---")
    for spec_id, entry in sorted(kb._entries.items()):
        best_score = entry.best_performance.score if entry.best_performance else 'N/A'
        print(f"  - Spec: {spec_id:<25} | Status: {entry.status:<12} | Best Score: {best_score}")

