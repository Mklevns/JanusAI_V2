#!/bin/bash
set -e

# Configuration
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=4
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Distributed settings
MASTER_ADDR=localhost
MASTER_PORT=29500
NNODES=1
NODE_RANK=0
NPROC_PER_NODE=4

# Launch distributed training
python -m torch.distributed.launch \
    --nproc_per_node=$NPROC_PER_NODE \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    janus/training/ppo/main.py \
    --config configs/production.yaml \
    --num-envs 64 \
    --total-timesteps 100000000 \
    --tensorboard \
    --wandb
