# scripts/run_simple_e2e_test.py
"""
Runs a simple end-to-end test of the JanusAI V2 framework.
Instantiates a simple environment and a random agent, then runs a discovery loop.
"""
import logging

from janus.envs.symbolic_regression import SymbolicRegressionEnv
from janus.agents.random_agent import RandomAgent

def main():
    """Main execution function."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    logger.info("--- Starting JanusAI V2 Simple End-to-End Test ---")

    # 1. Instantiate the Environment
    env = SymbolicRegressionEnv()
    logger.info(f"Environment created. Target function: {env.function_name}")
    grammar = env.grammar
    logger.info(f"Grammar has {len(grammar.get_variables())} variables and {len(grammar.get_binary_ops())} binary ops.")

    # 2. Instantiate the Agent
    agent = RandomAgent(env)
    logger.info("RandomAgent created.")

    # 3. Run the discovery loop
    num_episodes = 20
    best_reward = -float('inf')
    best_expression = ""

    logger.info(f"\n--- Running {num_episodes} discovery episodes ---")
    for i in range(num_episodes):
        obs, _ = env.reset()
        action = agent.get_action(observation=obs)
        obs, reward, terminated, truncated, info = env.step(action)
        
        logger.info(
            f"Episode {i+1:02d} | "
            f"Expression: {info.get('expression', 'N/A'):<45} | "
            f"Reward: {info.get('reward', 0.0):.2f}"
        )

        if reward > best_reward:
            best_reward = reward
            best_expression = info.get('expression', 'N/A')
    
    logger.info("\n--- Test Complete ---")
    logger.info(f"Best expression found: {best_expression}")
    logger.info(f"Best reward achieved: {best_reward:.4f}")

if __name__ == "__main__":
    main()