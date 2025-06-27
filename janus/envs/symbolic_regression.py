# janus/envs/symbolic_regression.py
"""
A simple environment for symbolic regression on a known mathematical function.
"""
import numpy as np
import sympy

from janus.core.base_env import BaseEnv
from janus.core.symbolic import Variable, Expression
from janus.grammars.math_grammar import BasicMathGrammar

class SymbolicRegressionEnv(BaseEnv):
    """
    An environment where the agent's goal is to discover a hidden
    symbolic mathematical function.

    - **State**: The current symbolic expression being built.
    - **Action**: A complete symbolic expression to evaluate.
    - **Reward**: The negative Mean Squared Error (MSE).
    """
    def __init__(self, function_name: str = "f(x) = x**2 + 0.5*x - 1.0"):
        self.function_name = function_name
        self.x_var = Variable(name="x", index=0)

        self.X_data = np.linspace(-10, 10, 100).reshape(-1, 1)
        self.y_true = self._target_function(self.X_data)

        self.grammar = BasicMathGrammar(variables=[self.x_var])
        self.reset()

    def _target_function(self, x: np.ndarray) -> np.ndarray:
        return x**2 + 0.5 * x - 1.0

    def reset(self, **kwargs):
        self.current_expression = None
        # Return a simple, placeholder observation
        return np.array([0]), {}

    def step(self, action: Expression):
        if not isinstance(action, Expression):
            raise TypeError("Action must be a valid Expression object.")

        self.current_expression = action
        y_pred = self._evaluate_expression(action)

        mse = np.mean((self.y_true - y_pred)**2)
        # Add a penalty for nonsensical results (inf/nan)
        if not np.isfinite(mse):
            mse = 1e9
        reward = -mse

        done = True
        info = {"expression": str(action), "mse": mse, "reward": reward}

        return np.array([0]), reward, done, False, info # Adheres to gymnasium 5-tuple return

    def _evaluate_expression(self, expr: Expression) -> np.ndarray:
        """Safely evaluates a symbolic expression against the environment's data."""
        try:
            x_sym = sympy.Symbol(self.x_var.name)
            sympy_expr = self._to_sympy(expr, {self.x_var.name: x_sym})
            
            # Inspired by symbolic_math.py: create a fast numerical function
            func = sympy.lambdify(x_sym, sympy_expr, 'numpy')
            
            # Ensure the function is robust against different numpy versions
            x_flat = self.X_data.flatten()
            result = func(x_flat)
            
            # Handle cases where the expression is a constant
            if np.isscalar(result):
                result = np.full_like(x_flat, result)

            return result.reshape(-1, 1)
        except Exception:
            return np.full_like(self.y_true, 1e9)

    def _to_sympy(self, expr: Expression, symbols: dict) -> sympy.Expr:
        op = expr.operator
        operands = expr.operands
        
        if op == 'var':
            return symbols[operands[0].name]
        if op == 'const':
            return sympy.sympify(operands[0])

        sympy_operands = [self._to_sympy(o, symbols) for o in operands]

        if op == '+': return sympy.Add(*sympy_operands)
        if op == '-': return sympy.Subtract(*sympy_operands) if len(sympy_operands) > 1 else -sympy_operands[0]
        if op == '*': return sympy.Mul(*sympy_operands)
        
        raise ValueError(f"Unknown operator: {op}")