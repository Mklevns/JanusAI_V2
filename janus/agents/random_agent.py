# janus/agents/random_agent.py
"""
A simple agent that generates random actions based on a grammar.
"""
import random
from typing import Dict, Any

from janus.core.base_agent import BaseAgent
from janus.core.base_env import BaseEnv
from janus.core.symbolic import Expression

class RandomAgent(BaseAgent):
    """
    A baseline agent that takes actions by generating random expressions
    from the environment's grammar.
    """
    def __init__(self, env: BaseEnv, config: Dict[str, Any] = None):
        if not hasattr(env, 'grammar'):
            raise ValueError("Environment for RandomAgent must have a 'grammar' attribute.")
        self.grammar = env.grammar
        self.config = config or {"max_depth": 3}

    def get_action(self, observation: any, **kwargs) -> Expression:
        """Ignores the observation and returns a random expression."""
        max_depth = self.config.get("max_depth", 3)
        return self._generate_random_expr(depth=0, max_depth=max_depth)

    def _generate_random_expr(self, depth: int, max_depth: int) -> Expression:
        """
        Recursively generates a random expression.
        Inspired by the generator logic in the original operators.py.
        """
        # Inspired by operators.py: at max depth, must choose a terminal
        if depth >= max_depth or random.random() < 0.4:
            if random.random() < 0.7 and self.grammar.get_variables():
                var = random.choice(self.grammar.get_variables())
                return Expression(operator='var', operands=[var])
            else:
                const_val = float(random.choice(list(self.grammar.get_constants().values())))
                return Expression(operator='const', operands=[const_val])
        
        op_type = random.choice(['unary', 'binary'])

        if op_type == 'unary' and self.grammar.get_unary_ops():
            op = random.choice(list(self.grammar.get_unary_ops()))
            operand = self._generate_random_expr(depth + 1, max_depth)
            return Expression(operator=op, operands=[operand])
        
        if self.grammar.get_binary_ops(): # Default to binary if available
            op = random.choice(list(self.grammar.get_binary_ops()))
            left = self._generate_random_expr(depth + 1, max_depth)
            right = self._generate_random_expr(depth + 1, max_depth)
            return Expression(operator=op, operands=[left, right])
        
        # Ultimate fallback to a simple variable
        var = random.choice(self.grammar.get_variables())
        return Expression(operator='var', operands=[var])