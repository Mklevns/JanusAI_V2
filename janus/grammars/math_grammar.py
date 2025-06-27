# janus/grammars/math_grammar.py
"""
A simple, non-progressive grammar for basic mathematical expressions.
"""
from typing import Set, List, Dict, Any, Tuple
import random

from janus.core.base_grammar import BaseGrammar
from janus.core.symbolic import Variable, Expression

class BasicMathGrammar(BaseGrammar):
    """A concrete implementation of a basic mathematical grammar."""

    def __init__(self, variables: List[Variable]):
        self._variables = variables
        self._constants = {"1.0": 1.0, "0.5": 0.5} # Added more constants for variety
        self._unary_ops = {"-"}
        self._binary_ops = {"+", "-", "*"}
        self._all_ops = self._unary_ops.union(self._binary_ops)

    def get_variables(self) -> List[Variable]:
        return self._variables

    def get_constants(self) -> Dict[str, Any]:
        return self._constants

    def get_unary_ops(self) -> Set[str]:
        return self._unary_ops

    def get_binary_ops(self) -> Set[str]:
        return self._binary_ops

    def sample_rule(self, **kwargs) -> str:
        """Samples a random operator from the available set."""
        return random.choice(list(self._all_ops))

    def get_arity(self, op_name: str) -> int:
        """Gets the arity of a given operator."""
        if op_name in self.get_unary_ops():
            return 1
        if op_name in self.get_binary_ops():
            return 2
        return 0