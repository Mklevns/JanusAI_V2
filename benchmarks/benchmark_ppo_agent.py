# benchmarks/benchmark_ppo_agent.py
"""
Performance benchmarks for the production PPO Agent.

Measures:
- Inference speed (actions/second)
- Memory usage
- Scaling with batch size
- Mixed precision speedup
- Recurrent vs feedforward performance
"""
import time
import psutil
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
import pandas as pd

from janus.agents.ppo_agent import PPOAgent, NetworkConfig

# Check GPU availability
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
if DEVICE == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("Running on CPU")


class AgentBenchmark:
    """Benchmark suite for PPO agents."""
    
    def __init__(self):
        self.results = []
        
    def benchmark_inference_speed(
        self, 
        agent: PPOAgent, 
        batch_sizes: List[int] = [1, 16, 64, 256, 1024],
        num_steps: int = 1000
    ) -> pd.DataFrame:
        """Benchmark action selection speed."""
        print("\n=== Inference Speed Benchmark ===")
        
        results = []
        obs_dim = agent.observation_dim
        
        for batch_size in batch_sizes:
            # Warmup
            obs = torch.randn(batch_size, obs_dim, device=agent.device)
            for _ in range(10):
                agent.act(obs)
                
            # Benchmark
            torch.cuda.synchronize() if DEVICE == 'cuda' else None
            start_time = time.time()
            
            for _ in range(num_steps):
                obs = torch.randn(batch_size, obs_dim, device=agent.device)
                actions, log_probs, _ = agent.act(obs)
                
            torch.cuda.synchronize() if DEVICE == 'cuda' else None
            elapsed_time = time.time() - start_time
            
            actions_per_second = (batch_size * num_steps) / elapsed_time
            time_per_action = elapsed_time / (batch_size * num_steps) * 1000  # ms
            
            results.append({
                'batch_size': batch_size,
                'actions_per_second': actions_per_second,
                'time_per_action_ms': time_per_action,
                'total_time': elapsed_time
            })
            
            print(f"Batch {batch_size:4d}: {actions_per_second:,.0f} actions/sec "
                  f"({time_per_action:.3f} ms/action)")
        
        return pd.DataFrame(results)
    
    def benchmark_memory_usage(
        self,
        network_sizes: List[List[int]] = [[64, 64], [128, 128], [256, 256], [512, 512]],
        obs_dim: int = 100,
        action_dim: int = 10,
        batch_size: int = 256
    ) -> pd.DataFrame:
        """Benchmark memory usage for different network sizes."""
        print("\n=== Memory Usage Benchmark ===")
        
        results = []
        
        for layer_sizes in network_sizes:
            # Get initial memory
            if DEVICE == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                initial_memory = torch.cuda.memory_allocated()
            else:
                process = psutil.Process()
                initial_memory = process.memory_info().rss
            
            # Create agent
            config = NetworkConfig(layer_sizes=layer_sizes)
            agent = PPOAgent(
                observation_dim=obs_dim,
                action_dim=action_dim,
                actor_config=config,
                critic_config=config,
                device=DEVICE
            )
            
            # Run inference to allocate all buffers
            obs = torch.randn(batch_size, obs_dim, device=DEVICE)
            for _ in range(10):
                agent.act(obs)
                agent.get_value(obs)
            
            # Get memory after allocation
            if DEVICE == 'cuda':
                torch.cuda.synchronize()
                final_memory = torch.cuda.memory_allocated()
            else:
                final_memory = process.memory_info().rss
            
            memory_used_mb = (final_memory - initial_memory) / 1024 / 1024
            
            # Count parameters
            actor_params = sum(p.numel() for p in agent.actor.parameters())
            critic_params = sum(p.numel() for p in agent.critic.parameters())
            total_params = actor_params + critic_params
            
            results.append({
                'layer_sizes': str(layer_sizes),
                'total_params': total_params,
                'actor_params': actor_params,
                'critic_params': critic_params,
                'memory_mb': memory_used_mb,
                'params_per_mb': total_params / memory_used_mb if memory_used_mb > 0 else 0
            })
            
            print(f"Network {layer_sizes}: {total_params:,} params, "
                  f"{memory_used_mb:.1f} MB")
            
            # Cleanup
            del agent
            if DEVICE == 'cuda':
                torch.cuda.empty_cache()
        
        return pd.DataFrame(results)
    
    def benchmark_mixed_precision(
        self,
        layer_sizes: List[int] = [256, 256],
        batch_sizes: List[int] = [64, 256, 1024],
        num_steps: int = 500
    ) -> pd.DataFrame:
        """Compare FP32 vs FP16 performance."""
        if DEVICE != 'cuda':
            print("\n=== Mixed Precision Benchmark (Skipped - No GPU) ===")
            return pd.DataFrame()
            
        print("\n=== Mixed Precision Benchmark ===")
        
        results = []
        config = NetworkConfig(layer_sizes=layer_sizes)
        
        for use_amp in [False, True]:
            agent = PPOAgent(
                observation_dim=100,
                action_dim=10,
                actor_config=config,
                critic_config=config,
                device='cuda',
                enable_amp=use_amp
            )
            
            for batch_size in batch_sizes:
                # Warmup
                obs = torch.randn(batch_size, 100, device='cuda')
                for _ in range(10):
                    agent.act(obs)
                
                # Benchmark
                torch.cuda.synchronize()
                start_time = time.time()
                
                for _ in range(num_steps):
                    obs = torch.randn(batch_size, 100, device='cuda')
                    actions, log_probs, _ = agent.act(obs)
                    values, _ = agent.get_value(obs)
                    
                torch.cuda.synchronize()
                elapsed_time = time.time() - start_time
                
                throughput = (batch_size * num_steps) / elapsed_time
                
                results.append({
                    'precision': 'FP16' if use_amp else 'FP32',
                    'batch_size': batch_size,
                    'throughput': throughput,
                    'time': elapsed_time
                })
                
                print(f"{'FP16' if use_amp else 'FP32'} - Batch {batch_size}: "
                      f"{throughput:,.0f} samples/sec")
        
        return pd.DataFrame(results)
    
    def benchmark_recurrent_vs_feedforward(
        self,
        sequence_lengths: List[int] = [1, 10, 50, 100],
        batch_size: int = 32,
        num_episodes: int = 100
    ) -> pd.DataFrame:
        """Compare recurrent vs feedforward performance."""
        print("\n=== Recurrent vs Feedforward Benchmark ===")
        
        results = []
        
        # Feedforward agent
        ff_config = NetworkConfig(layer_sizes=[128, 128])
        ff_agent = PPOAgent(
            observation_dim=20,
            action_dim=4,
            actor_config=ff_config,
            critic_config=ff_config,
            device=DEVICE
        )
        
        # Recurrent agent
        rnn_config = NetworkConfig(
            layer_sizes=[64],
            use_recurrent=True,
            recurrent_type='lstm',
            recurrent_hidden_size=128
        )
        rnn_agent = PPOAgent(
            observation_dim=20,
            action_dim=4,
            actor_config=rnn_config,
            critic_config=rnn_config,
            device=DEVICE
        )
        
        for seq_length in sequence_lengths:
            # Benchmark feedforward
            torch.cuda.synchronize() if DEVICE == 'cuda' else None
            start_time = time.time()
            
            for _ in range(num_episodes):
                for t in range(seq_length):
                    obs = torch.randn(batch_size, 20, device=DEVICE)
                    ff_agent.act(obs)
                    
            torch.cuda.synchronize() if DEVICE == 'cuda' else None
            ff_time = time.time() - start_time
            
            # Benchmark recurrent
            torch.cuda.synchronize() if DEVICE == 'cuda' else None
            start_time = time.time()
            
            for _ in range(num_episodes):
                hidden = rnn_agent.reset_hidden_states(batch_size)
                for t in range(seq_length):
                    obs = torch.randn(batch_size, 20, device=DEVICE)
                    _, _, hidden['actor'] = rnn_agent.act(
                        obs, actor_hidden=hidden['actor']
                    )
                    
            torch.cuda.synchronize() if DEVICE == 'cuda' else None
            rnn_time = time.time() - start_time
            
            ff_throughput = (num_episodes * seq_length * batch_size) / ff_time
            rnn_throughput = (num_episodes * seq_length * batch_size) / rnn_time
            
            results.append({
                'sequence_length': seq_length,
                'ff_time': ff_time,
                'rnn_time': rnn_time,
                'ff_throughput': ff_throughput,
                'rnn_throughput': rnn_throughput,
                'rnn_overhead': (rnn_time - ff_time) / ff_time * 100
            })
            
            print(f"Seq length {seq_length:3d}: "
                  f"FF={ff_throughput:,.0f} samples/sec, "
                  f"RNN={rnn_throughput:,.0f} samples/sec "
                  f"(overhead: {results[-1]['rnn_overhead']:.1f}%)")
        
        return pd.DataFrame(results)
    
    def benchmark_action_masking_overhead(
        self,
        batch_sizes: List[int] = [1, 16, 64, 256],
        num_actions: int = 100,
        mask_sparsity: List[float] = [0.0, 0.5, 0.9],
        num_steps: int = 1000
    ) -> pd.DataFrame:
        """Benchmark overhead of action masking."""
        print("\n=== Action Masking Overhead Benchmark ===")
        
        results = []
        
        config = NetworkConfig(layer_sizes=[128, 128])
        agent = PPOAgent(
            observation_dim=50,
            action_dim=num_actions,
            actor_config=config,
            critic_config=config,
            device=DEVICE
        )
        
        for batch_size in batch_sizes:
            for sparsity in mask_sparsity:
                # Create observations
                obs = torch.randn(batch_size, 50, device=DEVICE)
                
                # Create masks with specified sparsity
                if sparsity > 0:
                    mask = torch.rand(batch_size, num_actions, device=DEVICE) > sparsity
                    # Ensure at least one valid action per sample
                    mask[:, 0] = True
                else:
                    mask = None
                
                # Warmup
                for _ in range(10):
                    agent.act(obs, action_mask=mask)
                
                # Benchmark
                torch.cuda.synchronize() if DEVICE == 'cuda' else None
                start_time = time.time()
                
                for _ in range(num_steps):
                    agent.act(obs, action_mask=mask)
                    
                torch.cuda.synchronize() if DEVICE == 'cuda' else None
                elapsed_time = time.time() - start_time
                
                throughput = (batch_size * num_steps) / elapsed_time
                
                results.append({
                    'batch_size': batch_size,
                    'mask_sparsity': sparsity,
                    'has_mask': mask is not None,
                    'throughput': throughput,
                    'time': elapsed_time
                })
                
        df = pd.DataFrame(results)
        
        # Calculate overhead
        for batch_size in batch_sizes:
            no_mask = df[(df['batch_size'] == batch_size) & (df['mask_sparsity'] == 0.0)]['throughput'].iloc[0]
            
            for _, row in df[df['batch_size'] == batch_size].iterrows():
                overhead = (no_mask - row['throughput']) / no_mask * 100 if row['has_mask'] else 0
                print(f"Batch {batch_size}, Sparsity {row['mask_sparsity']:.1f}: "
                      f"{row['throughput']:,.0f} samples/sec "
                      f"(overhead: {overhead:.1f}%)")
        
        return df
    
    def create_plots(self, results: Dict[str, pd.DataFrame], save_path: str = "benchmark_results.png"):
        """Create visualization of benchmark results."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('PPO Agent Performance Benchmarks', fontsize=16)
        
        # Plot 1: Inference speed vs batch size
        if 'inference_speed' in results:
            ax = axes[0, 0]
            df = results['inference_speed']
            ax.plot(df['batch_size'], df['actions_per_second'], 'o-', linewidth=2, markersize=8)
            ax.set_xlabel('Batch Size')
            ax.set_ylabel('Actions per Second')
            ax.set_title('Inference Speed Scaling')
            ax.set_xscale('log')
            ax.grid(True, alpha=0.3)
        
        # Plot 2: Memory usage vs network size
        if 'memory_usage' in results:
            ax = axes[0, 1]
            df = results['memory_usage']
            x = range(len(df))
            ax.bar(x, df['memory_mb'])
            ax.set_xticks(x)
            ax.set_xticklabels(df['layer_sizes'], rotation=45)
            ax.set_xlabel('Network Architecture')
            ax.set_ylabel('Memory Usage (MB)')
            ax.set_title('Memory Usage by Network Size')
            ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Mixed precision speedup
        if 'mixed_precision' in results and not results['mixed_precision'].empty:
            ax = axes[1, 0]
            df = results['mixed_precision']
            fp32 = df[df['precision'] == 'FP32']
            fp16 = df[df['precision'] == 'FP16']
            
            width = 0.35
            x = np.arange(len(fp32))
            ax.bar(x - width/2, fp32['throughput'], width, label='FP32')
            ax.bar(x + width/2, fp16['throughput'], width, label='FP16')
            ax.set_xticks(x)
            ax.set_xticklabels(fp32['batch_size'])
            ax.set_xlabel('Batch Size')
            ax.set_ylabel('Throughput (samples/sec)')
            ax.set_title('Mixed Precision Performance')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Recurrent overhead
        if 'recurrent' in results:
            ax = axes[1, 1]
            df = results['recurrent']
            ax.plot(df['sequence_length'], df['rnn_overhead'], 'ro-', linewidth=2, markersize=8)
            ax.set_xlabel('Sequence Length')
            ax.set_ylabel('RNN Overhead (%)')
            ax.set_title('Recurrent Network Overhead')
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nBenchmark plots saved to {save_path}")
        
        return fig


def run_comprehensive_benchmark():
    """Run all benchmarks and generate report."""
    print("=" * 80)
    print("PPO Agent Comprehensive Performance Benchmark")
    print("=" * 80)
    
    benchmark = AgentBenchmark()
    results = {}
    
    # 1. Create a standard agent for most tests
    config = NetworkConfig(layer_sizes=[128, 128], activation="relu")
    agent = PPOAgent(
        observation_dim=100,
        action_dim=10,
        actor_config=config,
        critic_config=config,
        device=DEVICE
    )
    
    # 2. Run benchmarks
    results['inference_speed'] = benchmark.benchmark_inference_speed(agent)
    results['memory_usage'] = benchmark.benchmark_memory_usage()
    results['mixed_precision'] = benchmark.benchmark_mixed_precision()
    results['recurrent'] = benchmark.benchmark_recurrent_vs_feedforward()
    results['masking'] = benchmark.benchmark_action_masking_overhead()
    
    # 3. Generate summary statistics
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    
    # Best inference speed
    best_speed = results['inference_speed']['actions_per_second'].max()
    best_batch = results['inference_speed'].loc[
        results['inference_speed']['actions_per_second'].idxmax(), 'batch_size'
    ]
    print(f"Peak inference speed: {best_speed:,.0f} actions/sec (batch size: {best_batch})")
    
    # Memory efficiency
    if 'memory_usage' in results:
        most_efficient = results['memory_usage'].loc[
            results['memory_usage']['params_per_mb'].idxmax()
        ]
        print(f"Most memory efficient: {most_efficient['layer_sizes']} "
              f"({most_efficient['params_per_mb']:.0f} params/MB)")
    
    # Mixed precision speedup
    if 'mixed_precision' in results and not results['mixed_precision'].empty:
        fp32_avg = results['mixed_precision'][results['mixed_precision']['precision'] == 'FP32']['throughput'].mean()
        fp16_avg = results['mixed_precision'][results['mixed_precision']['precision'] == 'FP16']['throughput'].mean()
        speedup = fp16_avg / fp32_avg
        print(f"Mixed precision speedup: {speedup:.2f}x")
    
    # Create plots
    benchmark.create_plots(results)
    
    # Save detailed results
    with pd.ExcelWriter('ppo_agent_benchmark_results.xlsx') as writer:
        for name, df in results.items():
            if not df.empty:
                df.to_excel(writer, sheet_name=name, index=False)
    print("\nDetailed results saved to ppo_agent_benchmark_results.xlsx")
    
    return results


if __name__ == '__main__':
    results = run_comprehensive_benchmark()