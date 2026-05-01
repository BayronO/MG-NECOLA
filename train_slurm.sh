#!/bin/bash

#SBATCH --job-name=MGSR
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=mrigpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --time=30:00:00

# Load environment and modules
module use /mnt/it_software/easybuild/modules/all
module load Anaconda3/2022.10
module load CUDA/12.4.0
source ~/.bashrc

# Activate Conda Environment
conda activate pylians

# ---- Distributed Training Environment Variables (fixed) ----
export MASTER_ADDR=127.0.0.1                      # <-- fixed: no scontrol
export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")  # (use your fixed port if you prefer)
export WORLD_SIZE=$((SLURM_NNODES * 4))           # 4 GPUs per node

echo "Master node: $MASTER_ADDR"
echo "Master Port: $MASTER_PORT"
echo "World size: $WORLD_SIZE"
echo "CUDA_VISIBLE_DEVICES before training: ${CUDA_VISIBLE_DEVICES:-"(unset)"}"

# Train model using torch.distributed
python m2m.py train \
  --train-in-patterns "/mnt/mridata/mrhurtad/bayron/data/train_links/in/*/cola_*.npy" \
  --train-tgt-patterns "/mnt/mridata/mrhurtad/bayron/data/train_links/tgt/*/quijote_*.npy" \
  --val-in-patterns "/mnt/mridata/mrhurtad/bayron/data/val_links/in/*/cola_*.npy" \
  --val-tgt-patterns "/mnt/mridata/mrhurtad/bayron/data/val_links/tgt/*/quijote_*.npy" \
  --in-norms cosmology.dis --tgt-norms cosmology.dis \
  --augment --crop 128 --pad 20 \
  --model vnet.VNet --callback-at . \
  --lr 1e-4 --optimizer Adam --optimizer-args '{"betas": [0.9, 0.999], "weight_decay": 1e-4}' \
  --reduce-lr-on-plateau --scheduler-args '{"factor": 0.5, "patience": 2, "threshold": 1e-3, "verbose": true}' \
  --batches 2 --epochs 150 --loader-workers 4 --div-data --div-shuffle-dist 1\
  --L-sub 250.0 \
  --lambda-lag  1.0 \
  --lambda-eul  3.0 \
  --lambda-grad 2.0 \
  --lambda-spec 1.0 \
  --spec-kmin 0.3 \
  --spec-alpha 2.0

echo "✅ Training completed with 4 GPUs!"
