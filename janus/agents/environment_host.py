# janus/agents/environment_host.py

import logging
import uuid
from typing import Dict, List, Tuple

from janus.core.base_environment_spec import BaseEnvironmentSpec

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EnvironmentHost:
    """Manages a population of environment specifications.

    The EnvironmentHost acts as a service or "problem factory" that is controlled
    by a higher-level ArchitectAgent. It is responsible for holding a population
    of BaseEnvironmentSpec objects, creating new specs through mutation (speciation),
    and pruning the population based on performance criteria.
    """

    def __init__(self, initial_spec: BaseEnvironmentSpec):
        """
        Initializes the EnvironmentHost with a starting specification.

        Args:
            initial_spec (BaseEnvironmentSpec): The root environment spec to
                start the population.
        """
        self._specs: Dict[str, BaseEnvironmentSpec] = {initial_spec.spec_id: initial_spec}
        logging.info(f"EnvironmentHost initialized with root spec: '{initial_spec.spec_id}'")

    def get_spec(self, spec_id: str) -> BaseEnvironmentSpec | None:
        """Retrieves a single environment specification by its ID."""
        return self._specs.get(spec_id)

    def get_all_specs(self) -> Dict[str, BaseEnvironmentSpec]:
        """Returns the entire dictionary of current specifications."""
        return self._specs

    def speciate(self, parent_spec_id: str, num_children: int = 1) -> List[str]:
        """
        Creates new child specs by mutating a parent spec.

        This is a primary action called by the ArchitectAgent.

        Args:
            parent_spec_id (str): The ID of the parent spec to mutate.
            num_children (int): The number of new child specs to generate.

        Returns:
            List[str]: A list of the new child spec IDs.
        
        Raises:
            KeyError: If the parent_spec_id is not found.
        """
        if parent_spec_id not in self._specs:
            raise KeyError(f"Parent spec ID '{parent_spec_id}' not found in host.")
        
        parent_spec = self._specs[parent_spec_id]
        new_spec_ids = []
        for _ in range(num_children):
            new_id = str(uuid.uuid4())[:8]
            child_spec = parent_spec.mutate(new_spec_id=new_id)
            self._specs[child_spec.spec_id] = child_spec
            new_spec_ids.append(child_spec.spec_id)
        
        logging.info(f"Speciated {num_children} children from '{parent_spec_id}'. New IDs: {new_spec_ids}")
        return new_spec_ids

    def prune(self, spec_ids_to_prune: List[str]) -> None:
        """Removes a list of specifications from the population."""
        for spec_id in spec_ids_to_prune:
            if spec_id in self._specs:
                del self._specs[spec_id]
                logging.info(f"Pruned spec '{spec_id}'.")
            else:
                logging.warning(f"Attempted to prune non-existent spec '{spec_id}'.")

    def apply_minimal_criterion(
        self,
        solver_performances: Dict[str, float],
        min_threshold: float,
        max_threshold: float
    ) -> Tuple[List[str], List[str]]:
        """
        Filters specs based on the POET minimal criterion.

        An environment is kept if its challenge is "not too easy, not too hard"
        for the current population of solvers.

        Args:
            solver_performances (Dict[str, float]): A mapping of spec_id to the
                best score achieved by any solver on that spec.
            min_threshold (float): The minimum score for a spec to be considered
                non-trivial and thus kept.
            max_threshold (float): The maximum score for a spec to be considered
                challenging enough and thus kept. Specs with scores above this are
                considered "mastered" or too easy.

        Returns:
            Tuple[List[str], List[str]]: A tuple containing two lists:
                - The IDs of specs to promote (keep).
                - The IDs of specs to prune.
        """
        to_promote = []
        to_prune = []

        for spec_id in self._specs:
            score = solver_performances.get(spec_id)
            if score is None:
                # If a spec has not been evaluated, we don't prune it yet
                continue
            
            if min_threshold <= score < max_threshold:
                to_promote.append(spec_id)
            else:
                to_prune.append(spec_id)
        
        logging.info(f"Minimal Criterion Evaluation: Promoting {len(to_promote)}, Pruning {len(to_prune)}")
        return to_promote, to_prune

if __name__ == '__main__':
    """
    Demonstrates the EnvironmentHost's role as a service driven by an external
    process (simulating the ArchitectAgent).
    """
    print("--- Demonstrating Architect-Driven EnvironmentHost ---")

    # 1. Define the initial parameter space for a physics experiment (from a YAML file)
    physics_param_space = {
        "gravity": {"initial_value": 9.8, "mutation_step": 0.5, "max_value": 20.0},
        "friction": {"initial_value": 0.1, "mutation_step": 0.05, "max_value": 1.0},
    }
    root_spec = BaseEnvironmentSpec(spec_id="root_physics_env", params=physics_param_space)

    # 2. Instantiate the EnvironmentHost
    host = EnvironmentHost(initial_spec=root_spec)
    print(f"\nHost initialized with {len(host.get_all_specs())} spec(s).")
    print(host.get_all_specs())

    # --- SIMULATE ARCHITECT AGENT LOOP ---

    # 3. Architect decides to expand the environment space from the root node
    print("\n[Architect] Action: Speciate from 'root_physics_env'.")
    new_ids = host.speciate(parent_spec_id="root_physics_env", num_children=5)
    print(f"\nHost now contains {len(host.get_all_specs())} specs.")
    
    # After speciation, solvers would be run on these new environments and
    # their performance would be logged to the KnowledgeBase.

    # 4. Architect reads performance data from the KnowledgeBase
    # Let's create some dummy performance data for demonstration.
    dummy_solver_performances = {
        "root_physics_env": 350.0, # Mastered, should be pruned
        new_ids[0]: 150.0,         # Good, should be kept
        new_ids[1]: 25.0,          # Too hard, should be pruned
        new_ids[2]: 180.0,         # Good, should be kept
        new_ids[3]: 400.0,         # Mastered, should be pruned
        # new_ids[4] is not in the dict, simulating it hasn't been evaluated yet
    }
    print("\n[Architect] Reading solver performance from KnowledgeBase...")
    print(dummy_solver_performances)

    # 5. Architect commands the Host to evaluate and prune the population
    print("\n[Architect] Action: Apply minimal criterion and prune population.")
    min_score_threshold = 50.0
    max_score_threshold = 300.0
    
    promoted_ids, pruned_ids = host.apply_minimal_criterion(
        solver_performances=dummy_solver_performances,
        min_threshold=min_score_threshold,
        max_threshold=max_score_threshold,
    )
    
    print(f"  -> Specs to Promote: {promoted_ids}")
    print(f"  -> Specs to Prune: {pruned_ids}")

    host.prune(pruned_ids)
    
    print(f"\nHost now contains {len(host.get_all_specs())} spec(s) after pruning:")
    print(host.get_all_specs())
    
    print("\nThis demonstrates the Host acting as a service, modifying its population")
    print("based on external performance data and commands, fulfilling the Option C design.")