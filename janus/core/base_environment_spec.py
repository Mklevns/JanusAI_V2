# janus/core/base_environment_spec.py

from __future__ import annotations
import typing
from typing import Any, Dict, List, Union
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EvolvableParameter:
    """Represents a single evolvable parameter of an environment."""

    def __init__(self, name: str, initial_value: Union[float, List[float]],
                 mutation_step: float, min_value: float = -np.inf, max_value: float = np.inf):
        """
        Initializes an evolvable parameter.

        Args:
            name (str): The name of the parameter.
            initial_value (Union[float, List[float]]): The starting value. If a list of two
                floats, it represents an interval [low, high].
            mutation_step (float): The magnitude of change during mutation.
            min_value (float): The minimum allowable value for the parameter.
            max_value (float): The maximum allowable value for the parameter.
        """
        self.name = name
        self.value = np.array(initial_value, dtype=np.float32)
        self.mutation_step = mutation_step
        self.min_value = min_value
        self.max_value = max_value

    def mutate(self) -> None:
        """Applies a mutation to the parameter's value."""
        mutation = np.random.choice([-self.mutation_step, self.mutation_step])
        self.value = np.clip(self.value + mutation, self.min_value, self.max_value)
        if self.value.shape == (2,) and self.value[0] > self.value[1]:
            # Ensure interval remains valid (low <= high)
            self.value[0], self.value[1] = self.value[1], self.value[0]

    def __repr__(self) -> str:
        return f"EvolvableParameter(name='{self.name}', value={self.value})"

class BaseEnvironmentSpec:
    """A base class for defining and evolving the specification of an environment.

    This class holds the evolvable parameters that define a specific challenge.
    It provides the core `mutate` functionality inspired by the POET algorithm,
    allowing new, slightly different environments to be generated from existing ones.
    """

    def __init__(self, spec_id: str, params: Dict[str, Dict[str, Any]]):
        """
        Initializes the environment specification from a dictionary.

        Args:
            spec_id (str): A unique identifier for this environment specification.
            params (Dict[str, Dict[str, Any]]): A dictionary defining the evolvable
                parameters, loaded from a YAML configuration file.
        """
        self.spec_id = spec_id
        self.evolvable_params: Dict[str, EvolvableParameter] = {}
        for name, config in params.items():
            self.evolvable_params[name] = EvolvableParameter(name=name, **config)

    def mutate(self, new_spec_id: str) -> BaseEnvironmentSpec:
        """
        Creates a new, mutated version of this environment specification.

        This method is a core component of the EnvironmentHost agent's action space.

        Args:
            new_spec_id (str): The ID for the new child environment spec.

        Returns:
            BaseEnvironmentSpec: A new instance with mutated parameters.
        """
        # Create a deep copy of the parameter definitions
        new_params_dict = {name: p.__dict__ for name, p in self.evolvable_params.items()}
        
        # Create a new spec instance to be mutated
        child_spec = BaseEnvironmentSpec(spec_id=new_spec_id, params={})
        child_spec.evolvable_params = {name: EvolvableParameter(**config) for name, config in new_params_dict.items()}

        # Mutate at least one parameter
        param_to_mutate = np.random.choice(list(child_spec.evolvable_params.keys()))
        child_spec.evolvable_params[param_to_mutate].mutate()

        logging.info(f"Mutated '{self.spec_id}' to create '{new_spec_id}'. Changed '{param_to_mutate}'.")
        return child_spec

    def get_parameter_values(self) -> Dict[str, Union[float, np.ndarray]]:
        """Returns a dictionary of current parameter names and their values."""
        return {name: p.value for name, p in self.evolvable_params.items()}

    def __repr__(self) -> str:
        param_str = ", ".join([f"{name}={p.value}" for name, p in self.evolvable_params.items()])
        return f"BaseEnvironmentSpec(id='{self.spec_id}', params=[{param_str}])"


if __name__ == '__main__':
    """
    Demonstrative example for a Physics Experiment use case.
    We define an environment spec for a 2D projectile motion simulation
    and show how the EnvironmentHost can evolve it.
    """
    print("--- Demonstrating Environment Specification and Evolution ---")

    # 1. Define the parameter space for a physics experiment via a dictionary
    #    This would typically be loaded from a YAML config file.
    physics_param_space = {
        "gravity": {
            "initial_value": 9.8,
            "mutation_step": 0.5,
            "min_value": 0.1,
            "max_value": 20.0,
        },
        "friction": {
            "initial_value": 0.1,
            "mutation_step": 0.05,
            "min_value": 0.0,
            "max_value": 1.0,
        },
        "target_distance_range": {
            "initial_value": [10.0, 15.0],
            "mutation_step": 1.0,
            "min_value": 5.0,
            "max_value": 100.0,
        }
    }

    # 2. Create the initial ("parent") environment specification
    parent_spec = BaseEnvironmentSpec(spec_id="physics_env_0", params=physics_param_space)
    print(f"\nInitial Parent Environment:\n{parent_spec}")
    print(f"Parameter Values: {parent_spec.get_parameter_values()}")

    # 3. The EnvironmentHost agent takes the 'mutate' action
    child_spec = parent_spec.mutate(new_spec_id="physics_env_1")
    
    # 4. A new, slightly more challenging environment is created
    print(f"\nGenerated Child Environment:\n{child_spec}")
    print(f"Parameter Values: {child_spec.get_parameter_values()}")

    # 5. Show another mutation to demonstrate randomness
    child_spec_2 = parent_spec.mutate(new_spec_id="physics_env_2")
    print(f"\nGenerated Second Child Environment:\n{child_spec_2}")
    print(f"Parameter Values: {child_spec_2.get_parameter_values()}")