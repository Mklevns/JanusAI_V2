# Abstract base class for environments
from abc import ABC, abstractmethod

class BaseEnv(ABC):
    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, action):
        pass
