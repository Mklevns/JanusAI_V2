# Abstract base class for agents
from abc import ABC, abstractmethod

class BaseAgent(ABC):
    @abstractmethod
    def act(self, observation):
        pass

    @abstractmethod
    def learn(self, *args, **kwargs):
        pass
