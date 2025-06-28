# File: janus/envs/base.py
# Base class for all environments

import abc


class BaseEnv(abc.ABC):
    """
    Abstract base class for all environments in the JanusAI library.
    It defines the standard interface that all environments must implement.
    """

    def __init__(self):
        self.action_space = None
        self.observation_space = None

    @abc.abstractmethod
    def step(self, action):
        """
        Run one timestep of the environment's dynamics.

        Args:
            action: An action provided by the agent.

        Returns:
            A tuple (observation, reward, done, info).
            - observation: Agent's observation of the current environment.
            - reward: Amount of reward returned after previous action.
            - done: Whether the episode has ended.
            - info: A dictionary with auxiliary diagnostic information.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self):
        """
        Resets the environment to an initial state and returns an initial
        observation.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def render(self, mode="human"):
        """
        Renders the environment.
        """
        raise NotImplementedError

    def close(self):
        """
        Performs any necessary cleanup.
        """
