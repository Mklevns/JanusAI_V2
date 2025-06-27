# Abstract base class for grammars
from abc import ABC, abstractmethod

class BaseGrammar(ABC):
    @abstractmethod
    def generate(self):
        pass
