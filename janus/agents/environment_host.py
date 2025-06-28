# janus/arch/environment_host.py

"""
EnvironmentHost: A service for managing environment specifications.

This module defines the EnvironmentHost class, which acts as a service or "problem factory"
controlled by an ArchitectAgent. It manages a population of BaseEnvironmentSpec objects,
creating new specifications through mutation (speciation) and pruning the population based
on performance criteria.
"""

import logging
import uuid
from typing import Dict, List, Optional, Tuple

from janus.core.base_environment_spec import BaseEnvironmentSpec

# Get a logger for this module
logger = logging.getLogger(__name__)


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
        self._specs: Dict[str, BaseEnvironmentSpec] = {
            initial_spec.spec_id: initial_spec
        }
        logger.info(
            "EnvironmentHost initialized with root spec: '%s'", initial_spec.spec_id
        )

    def get_spec(self, spec_id: str) -> Optional[BaseEnvironmentSpec]:
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
        parent_spec = self._specs.get(parent_spec_id)
        if not parent_spec:
            raise KeyError(f"Parent spec ID '{parent_spec_id}' not found in host.")

        new_spec_ids = []
        for _ in range(num_children):
            new_id = str(uuid.uuid4())[:8]
            child_spec = parent_spec.mutate(new_spec_id=new_id)
            self._specs[child_spec.spec_id] = child_spec
            new_spec_ids.append(child_spec.spec_id)

        logger.info(
            "Speciated %d children from '%s'. New IDs: %s",
            num_children,
            parent_spec_id,
            new_spec_ids,
        )
        return new_spec_ids

    def prune(self, spec_ids_to_prune: List[str]) -> None:
        """Removes a list of specifications from the population."""
        for spec_id in spec_ids_to_prune:
            if spec_id in self._specs:
                del self._specs[spec_id]
                logger.info("Pruned spec '%s'.", spec_id)
            else:
                logger.warning("Attempted to prune non-existent spec '%s'.", spec_id)

    def apply_minimal_criterion(
        self,
        solver_performances: Dict[str, float],
        min_threshold: float,
        max_threshold: float,
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
            A tuple containing two lists: the IDs of specs to promote (keep) and
            the IDs of specs to prune.
        """
        to_promote = []
        to_prune = []

        # Iterate over a copy of the spec IDs in case the dictionary changes
        for spec_id in list(self._specs):
            score = solver_performances.get(spec_id)
            if score is None:
                # If a spec has not been evaluated, we don't prune it yet
                to_promote.append(spec_id)
                continue

            if min_threshold <= score < max_threshold:
                to_promote.append(spec_id)
            else:
                to_prune.append(spec_id)

        logger.info(
            "Minimal Criterion Evaluation: Promoting %d, Pruning %d",
            len(to_promote),
            len(to_prune),
        )
        return to_promote, to_prune


def demo_environment_host():
    """Demonstrates the EnvironmentHost's role as a service driven by an external process."""
    logger.info("--- Demonstrating Architect-Driven EnvironmentHost ---")

    # 1. Define the initial parameter space
    physics_param_space = {
        "gravity": {"initial_value": 9.8, "mutation_step": 0.5, "max_value": 20.0},
        "friction": {"initial_value": 0.1, "mutation_step": 0.05, "max_value": 1.0},
    }
    root_spec = BaseEnvironmentSpec(
        spec_id="root_physics_env", params=physics_param_space
    )

    # 2. Instantiate the EnvironmentHost
    host = EnvironmentHost(initial_spec=root_spec)
    logger.info("Host initialized with %d spec(s).", len(host.get_all_specs()))
    logger.debug("Initial specs: %s", host.get_all_specs())

    # --- SIMULATE ARCHITECT AGENT LOOP ---

    # 3. Architect speciates from the root node
    logger.info("[Architect] Action: Speciate from 'root_physics_env'.")
    new_ids = host.speciate(parent_spec_id="root_physics_env", num_children=5)
    logger.info("Host now contains %d specs.", len(host.get_all_specs()))

    # 4. Architect reads performance data from the KnowledgeBase
    dummy_solver_performances = {
        "root_physics_env": 350.0,
        new_ids[0]: 150.0,
        new_ids[1]: 25.0,
        new_ids[2]: 180.0,
        new_ids[3]: 400.0,
    }
    logger.info("[Architect] Reading solver performance from KnowledgeBase...")
    logger.debug("Dummy solver performances: %s", dummy_solver_performances)

    # 5. Architect commands the Host to evaluate and prune
    logger.info("[Architect] Action: Apply minimal criterion and prune population.")
    promoted_ids, pruned_ids = host.apply_minimal_criterion(
        solver_performances=dummy_solver_performances,
        min_threshold=50.0,
        max_threshold=300.0,
    )

    logger.info("Specs to Promote: %s", promoted_ids)
    logger.info("Specs to Prune: %s", pruned_ids)

    host.prune(pruned_ids)

    logger.info(
        "Host now contains %d spec(s) after pruning.", len(host.get_all_specs())
    )
    logger.debug("Final specs: %s", host.get_all_specs())

    logger.info("Demonstration complete.")


if __name__ == "__main__":
    # --- Logging Setup for Standalone Execution ---
    # This configuration is only active when the script is run directly.
    # It prevents conflicts with other parts of a larger application.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    demo_environment_host()
