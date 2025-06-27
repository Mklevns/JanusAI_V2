import logging
from typing import Dict, List, Optional

from janus.agents.environment_host import EnvironmentHost
from janus.core.knowledge_base import KnowledgeBase

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ArchitectAgent:
    """
    The meta-learning agent that orchestrates the open-ended discovery ecosystem.

    It reads experiment outcomes from the KnowledgeBase, makes strategic decisions
    about which environment specs to expand or prune, and instructs the EnvironmentHost
    to carry out speciation and cleanup.
    """

    def __init__(
        self,
        host: EnvironmentHost,
        knowledge_base: KnowledgeBase,
        min_score: float,
        max_score: float,
        children_per_cycle: int = 1,
    ):
        """
        Args:
            host (EnvironmentHost): The environment factory service.
            knowledge_base (KnowledgeBase): Central store of solver performance.
            min_score (float): Lower bound for minimal criterion.
            max_score (float): Upper bound for minimal criterion.
            children_per_cycle (int): Number of new specs to generate per cycle.
        """
        self.host = host
        self.kb = knowledge_base
        self.min_score = min_score
        self.max_score = max_score
        self.children_per_cycle = children_per_cycle

    def run_cycle(self) -> None:
        """
        Executes one iteration of the meta-learning loop:
          1. Fetch latest solver performances from the KnowledgeBase.
          2. Apply minimal criterion to determine specs to promote/prune.
          3. Prune unpromising specs via the host.
          4. Speciate (mutate) selected specs to expand the challenge space.
        """
        # 1. Gather performance data
        performances = self.kb.get_solver_best_scores()  # type: Dict[str, float]
        logging.info(f"Architect fetched performances: {performances}")

        # 2. Determine promotable and prunable specs
        to_promote, to_prune = self.host.apply_minimal_criterion(
            solver_performances=performances,
            min_threshold=self.min_score,
            max_threshold=self.max_score,
        )
        logging.info(f"Specs to promote: {to_promote}")
        logging.info(f"Specs to prune: {to_prune}")

        # 3. Prune specs
        if to_prune:
            self.host.prune(to_prune)

        # 4. Speciate on promoted specs
        for parent in to_promote:
            new_ids = self.host.speciate(parent_spec_id=parent, num_children=self.children_per_cycle)
            logging.info(f"Created children {new_ids} from parent {parent}")

        logging.info("Cycle complete. Ecosystem updated.")

if __name__ == '__main__':
    # Example demonstration of the ArchitectAgent controlling the EnvironmentHost

    from janus.core.base_environment_spec import BaseEnvironmentSpec
    from janus.core.knowledge_base import KnowledgeBase

    # 1. Set up initial spec and host
    init_params = {
        "gravity": {"initial_value": 9.8, "mutation_step": 0.5, "min_value": 0.1, "max_value": 20.0},
        "friction": {"initial_value": 0.1, "mutation_step": 0.05, "min_value": 0.0, "max_value": 1.0},
    }
    root = BaseEnvironmentSpec(spec_id="root_env", params=init_params)
    host = EnvironmentHost(initial_spec=root)

    # 2. Create a dummy KnowledgeBase and record fake performances
    kb = KnowledgeBase()
    kb.record_score(solver_id="solverA", spec_id="root_env", score=120.0)

    # 3. Instantiate ArchitectAgent
    arch = ArchitectAgent(
        host=host,
        knowledge_base=kb,
        min_score=50.0,
        max_score=300.0,
        children_per_cycle=2,
    )

    # 4. Run one meta-learning cycle
    arch.run_cycle()

    # 5. Inspect updated specs in the host
    print(f"Current specs: {list(host.get_all_specs().keys())}")
