# tests/test_rewards.py
"""Unit tests for the modular reward system."""

import pytest
import numpy as np
from typing import Dict, Any
from janus.rewards import (
    BaseRewardComponent,
    RewardHandler,
    TaskSuccessReward,
    ComplexityPenalty,
    NoveltyBonus,
    ProgressReward
)


class TestTaskSuccessReward:
    """Test cases for TaskSuccessReward component."""
    
    @pytest.fixture
    def component(self) -> TaskSuccessReward:
        """Create a test instance."""
        config = {'success_reward': 100.0, 'enabled': True}
        return TaskSuccessReward(config)
    
    def test_success_reward(self, component: TaskSuccessReward) -> None:
        """Test reward when task is completed."""
        obs = np.array([1, 2, 3])
        action = np.array([0.5])
        next_obs = np.array([2, 3, 4])
        info = {'task_completed': True}
        
        reward = component.compute(obs, action, next_obs, info)
        assert reward == 100.0
    
    def test_no_success_reward(self, component: TaskSuccessReward) -> None:
        """Test no reward when task not completed."""
        obs = np.array([1, 2, 3])
        action = np.array([0.5])
        next_obs = np.array([2, 3, 4])
        info = {'task_completed': False}
        
        reward = component.compute(obs, action, next_obs, info)
        assert reward == 0.0
    
    def test_no_info_dict(self, component: TaskSuccessReward) -> None:
        """Test behavior when info is None."""
        obs = np.array([1, 2, 3])
        action = np.array([0.5])
        next_obs = np.array([2, 3, 4])
        
        reward = component.compute(obs, action, next_obs, None)
        assert reward == 0.0


class TestComplexityPenalty:
    """Test cases for ComplexityPenalty component."""
    
    @pytest.fixture
    def component(self) -> ComplexityPenalty:
        """Create a test instance."""
        config = {'penalty_weight': 0.5, 'enabled': True}
        return ComplexityPenalty(config)
    
    def test_complexity_penalty(self, component: ComplexityPenalty) -> None:
        """Test penalty calculation."""
        obs = np.array([1, 2, 3])
        action = np.array([0.5])
        next_obs = np.array([2, 3, 4])
        info = {'solution_complexity': 10}
        
        reward = component.compute(obs, action, next_obs, info)
        assert reward == -5.0  # -0.5 * 10
    
    def test_zero_complexity(self, component: ComplexityPenalty) -> None:
        """Test no penalty for zero complexity."""
        obs = np.array([1, 2, 3])
        action = np.array([0.5])
        next_obs = np.array([2, 3, 4])
        info = {'solution_complexity': 0}
        
        reward = component.compute(obs, action, next_obs, info)
        assert reward == 0.0


class TestNoveltyBonus:
    """Test cases for NoveltyBonus component."""
    
    @pytest.fixture
    def component(self) -> NoveltyBonus:
        """Create a test instance."""
        config = {
            'bonus_scale': 2.0,
            'memory_size': 5,
            'enabled': True
        }
        return NoveltyBonus(config)
    
    def test_first_state_novelty(self, component: NoveltyBonus) -> None:
        """Test maximum novelty for first state."""
        obs = np.array([1, 2, 3])
        action = np.array([0.5])
        next_obs = np.array([2, 3, 4])
        
        reward = component.compute(obs, action, next_obs)
        assert reward == 2.0  # bonus_scale * 1.0
    
    def test_repeated_state_novelty(self, component: NoveltyBonus) -> None:
        """Test reduced novelty for repeated states."""
        obs = np.array([1, 2, 3])
        action = np.array([0.5])
        next_obs = np.array([2, 3, 4])
        
        # Visit same state twice
        reward1 = component.compute(obs, action, next_obs)
        reward2 = component.compute(obs, action, next_obs)
        
        assert reward1 > reward2
        assert reward2 < 2.0
    
    def test_memory_limit(self, component: NoveltyBonus) -> None:
        """Test memory size limit."""
        for i in range(10):
            obs = np.array([i, i+1, i+2])
            action = np.array([0.5])
            next_obs = np.array([i+1, i+2, i+3])
            component.compute(obs, action, next_obs)
        
        assert len(component.state_memory) == 5
    
    def test_reset(self, component: NoveltyBonus) -> None:
        """Test reset functionality."""
        obs = np.array([1, 2, 3])
        action = np.array([0.5])
        next_obs = np.array([2, 3, 4])
        
        component.compute(obs, action, next_obs)
        assert len(component.state_memory) == 1
        
        component.reset()
        assert len(component.state_memory) == 0


class TestProgressReward:
    """Test cases for ProgressReward component."""
    
    @pytest.fixture
    def component(self) -> ProgressReward:
        """Create a test instance."""
        config = {'scale_factor': 5.0, 'enabled': True}
        return ProgressReward(config)
    
    def test_progress_reward(self, component: ProgressReward) -> None:
        """Test progress-based reward."""
        obs = np.array([1, 2, 3])
        action = np.array([0.5])
        next_obs = np.array([2, 3, 4])
        info = {'progress': 0.7}
        
        reward = component.compute(obs, action, next_obs, info)
        assert reward == 3.5  # 5.0 * 0.7


class TestRewardHandler:
    """Test cases for RewardHandler."""
    
    @pytest.fixture
    def config(self) -> Dict[str, Any]:
        """Create test configuration."""
        return {
            'components': [
                {
                    'type': 'task_success',
                    'weight': 10.0,
                    'success_reward': 100.0,
                    'enabled': True
                },
                {
                    'type': 'complexity_penalty',
                    'weight': 0.1,
                    'penalty_weight': 1.0,
                    'enabled': True
                },
                {
                    'type': 'novelty_bonus',
                    'weight': 0.5,
                    'bonus_scale': 2.0,
                    'memory_size': 1000,
                    'enabled': True
                }
            ]
        }
    
    @pytest.fixture
    def handler(self, config: Dict[str, Any]) -> RewardHandler:
        """Create test handler."""
        return RewardHandler(config)
    
    def test_initialization(self, handler: RewardHandler) -> None:
        """Test handler initialization."""
        assert len(handler.components) == 3
        assert len(handler.weights) == 3
        assert handler.weights == [10.0, 0.1, 0.5]
    
    def test_compute_reward(self, handler: RewardHandler) -> None:
        """Test composite reward computation."""
        obs = np.array([1, 2, 3])
        action = np.array([0.5])
        next_obs = np.array([2, 3, 4])
        info = {
            'task_completed': True,
            'solution_complexity': 5
        }
        
        reward = handler.compute_reward(obs, action, next_obs, info)
        # Expected: 10.0 * 100 + 0.1 * (-5) + 0.5 * 2.0
        # = 1000 - 0.5 + 1.0 = 1000.5
        assert reward == pytest.approx(1000.5)
    
    def test_component_breakdown(self, handler: RewardHandler) -> None:
        """Test getting component breakdown."""
        obs = np.array([1, 2, 3])
        action = np.array([0.5])
        next_obs = np.array([2, 3, 4])
        info = {
            'task_completed': True,
            'solution_complexity': 5
        }
        
        breakdown = handler.get_component_breakdown(
            obs, action, next_obs, info
        )
        
        assert 'TaskSuccessReward' in breakdown
        assert 'ComplexityPenalty' in breakdown
        assert 'NoveltyBonus' in breakdown
        assert 'total' in breakdown
        assert breakdown['TaskSuccessReward'] == 100.0
        assert breakdown['ComplexityPenalty'] == -5.0
    
    def test_disabled_component(self) -> None:
        """Test that disabled components are not included."""
        config = {
            'components': [
                {
                    'type': 'task_success',
                    'weight': 1.0,
                    'success_reward': 100.0,
                    'enabled': False
                }
            ]
        }
        handler = RewardHandler(config)
        assert len(handler.components) == 0
    
    def test_unknown_component_type(self) -> None:
        """Test error handling for unknown component."""
        config = {
            'components': [
                {
                    'type': 'unknown_type',
                    'weight': 1.0
                }
            ]
        }
        with pytest.raises(ValueError, match="Unknown component type"):
            RewardHandler(config)
    
    def test_reset_all_components(self, handler: RewardHandler) -> None:
        """Test resetting all components."""
        # First, generate some state in novelty component
        obs = np.array([1, 2, 3])
        action = np.array([0.5])
        next_obs = np.array([2, 3, 4])
        handler.compute_reward(obs, action, next_obs)
        
        # Reset and verify
        handler.reset()
        novelty_comp = None
        for comp in handler.components:
            if isinstance(comp, NoveltyBonus):
                novelty_comp = comp
                break
        
        assert novelty_comp is not None
        assert len(novelty_comp.state_memory) == 0


# Integration test
class TestIntegration:
    """Integration tests for the reward system."""
    
    def test_full_episode_simulation(self) -> None:
        """Simulate a full episode with reward tracking."""
        config = {
            'components': [
                {
                    'type': 'task_success',
                    'weight': 10.0,
                    'success_reward': 100.0
                },
                {
                    'type': 'progress',
                    'weight': 1.0,
                    'scale_factor': 5.0
                }
            ]
        }
        
        handler = RewardHandler(config)
        total_reward = 0.0
        
        # Simulate episode steps
        for step in range(10):
            obs = np.random.randn(4)
            action = np.random.randn(2)
            next_obs = np.random.randn(4)
            
            progress = step / 10.0
            info = {
                'progress': progress,
                'task_completed': step == 9
            }
            
            reward = handler.compute_reward(obs, action, next_obs, info)
            total_reward += reward
        
        # Final step should have high reward
        assert total_reward > 1000  # At least task success


if __name__ == "__main__":
    pytest.main([__file__, "-v"])