# Abstract base class for rewards
from abc import ABC, abstractmethod

class BaseReward(ABC):
    @abstractmethod
    def compute(self, *args, **kwargs):
        pass
