# File: janus/envs/gridworld.py
"""A simple GridWorld environment."""

import numpy as np
from janus.envs.base import BaseEnv


class GridWorldEnv(BaseEnv):
    """
    A simple GridWorld environment.
    The agent starts at a random position and needs to navigate to a goal.
    """

    def __init__(self, size=10, max_steps=100, seed=None):
        super().__init__()
        self.size = size
        self.max_steps = max_steps
        self.observation_space = np.zeros(2)
        self.action_space = np.arange(4)  # 0: up, 1: down, 2: left, 3: right
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        """Resets the environment to a new random state."""
        self.agent_pos = self.rng.integers(0, self.size, size=2)
        self.goal_pos = self.rng.integers(0, self.size, size=2)
        # Ensure agent and goal are not at the same position
        while np.array_equal(self.agent_pos, self.goal_pos):
            self.goal_pos = self.rng.integers(0, self.size, size=2)
        self.current_step = 0
        return self.agent_pos

    def step(self, action):
        """
        Run one timestep of the environment's dynamics.
        The action is a value from 0 to 3, representing up, down, left, or right.
        """
        # 0: up, 1: down, 2: left, 3: right
        moves = np.array([[0, 1], [0, -1], [-1, 0], [1, 0]])
        self.agent_pos = np.clip(self.agent_pos + moves[action], 0, self.size - 1)

        self.current_step += 1
        done = (np.array_equal(self.agent_pos, self.goal_pos) or
                self.current_step >= self.max_steps)
        reward = 1.0 if np.array_equal(self.agent_pos, self.goal_pos) else -0.1
        info = {}
        return self.agent_pos, reward, done, info

    def render(self, mode='human'):
        """Renders the environment to the console."""
        grid = np.full((self.size, self.size), '_')
        grid[self.agent_pos[1], self.agent_pos[0]] = 'A'
        grid[self.goal_pos[1], self.goal_pos[0]] = 'G'
        print(np.flipud(grid))