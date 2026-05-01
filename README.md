# MG-NECOLA

**MG-NECOLA** is a field-level deep-learning emulator for modified gravity cosmologies. It upgrades fast approximate **MG-PICOLA** simulations to near **QUIJOTE-MG N-body** accuracy using a V-Net convolutional neural network.

This repository is a fork and extension of [`map2map`](https://github.com/eelregit/map2map), adapted for **Hu-Sawicki f(R) gravity**, massive-neutrino generalization tests, and physics-driven loss functions for large-scale structure reconstruction.

Paper:

**MG-NECOLA: A Field-Level Emulator for f(R) Gravity and Massive Neutrino Cosmologies**  
J. Bayron Orjuela-Quintana, Mauricio Reyes, Elena Giusarma, Marco Baldi, Neerav Kaushal, and César A. Valenzuela-Toledo  
[arXiv:2604.19613](https://arxiv.org/abs/2604.19613)

---

## Overview

High-resolution N-body simulations provide accurate predictions for non-linear structure formation, but they are computationally expensive for the large simulation ensembles required in modified gravity analyses.

MG-NECOLA bridges this gap by learning a field-level correction:

```text
MG-PICOLA  -->  QUIJOTE-MG-like N-body output
```

Instead of generating cosmological fields from scratch, the network acts as a **residual corrector**. It takes approximate MG-PICOLA displacement fields as input and predicts the residual displacement needed to match the high-fidelity QUIJOTE-MG simulation.

The learned residual is

```text
Delta_res = Delta_QUIJOTE-MG - Delta_MG-PICOLA
```

and the corrected displacement is obtained by adding this residual back to the MG-PICOLA input.

---

## Main Scientific Contributions

MG-NECOLA introduces a field-level emulator for modified gravity simulations with the following key results:

- Upgrades fast MG-PICOLA simulations to near N-body fidelity.
- Learns non-linear displacement residuals between MG-PICOLA and QUIJOTE-MG.
- Uses a V-Net architecture based on the original `map2map` framework.
- Extends the NECOLA/map2map approach from standard gravity to **Hu-Sawicki f(R) gravity**.
- Adds a modified training objective with:
  - Lagrangian displacement loss
  - Eulerian density loss
  - Gradient-matching loss
  - Fourier-space spectral loss
- Recovers the matter power spectrum and bispectrum with nearly sub-percent accuracy up to approximately `k ~ 1 h/Mpc`.
- Reduces the mean particle displacement error by approximately 30% relative to MG-PICOLA.
- Generalizes to cosmologies outside the training manifold.
- Recovers the LCDM/GR limit without introducing spurious modified-gravity signals.
- Captures massive-neutrino power suppression for `M_nu <= 0.4 eV`, despite being trained on massless-neutrino simulations.
- Achieves a total speed-up of roughly `1500x` relative to full N-body simulations.

---

## Physical Problem

Modified gravity models can alter the growth of cosmic structure through scale-dependent gravitational forces and screening mechanisms. These effects are especially important in the mildly non-linear and non-linear regimes, where approximate solvers such as COLA begin to lose accuracy.

In Hu-Sawicki f(R) gravity, the scalar degree of freedom enhances gravity in low-density regions while recovering General Relativity in high-density regions through chameleon screening. This makes the problem difficult for approximate solvers and motivates a field-level neural correction.

MG-NECOLA uses the speed of MG-PICOLA while correcting its small-scale errors using a neural emulator trained against QUIJOTE-MG.

---

## Data

The model is trained on paired simulations:

- **Input:** MG-PICOLA approximate simulations
- **Target:** QUIJOTE-MG high-fidelity N-body simulations

The paper uses the fixed-cosmology subset of QUIJOTE-MG for Hu-Sawicki f(R) gravity. Four scalar-field amplitudes are considered:

```text
fR0 = -5e-7, -5e-6, -5e-5, -5e-4
```

For training, the dataset contains 100 paired simulations:

```text
25 simulations per f(R) label x 4 labels = 100 simulations
```

The simulations are split as:

```text
70% training
20% validation
10% testing
```

The full displacement fields have shape:

```text
3 x 512^3
```

where the three channels correspond to the three Cartesian components of the displacement field.

During training, fields are decomposed into cubic sub-volumes:

```text
3 x 128^3
```

corresponding to sub-boxes of size approximately:

```text
250 Mpc/h per side
```

Each sub-volume is periodically padded by 20 voxels before entering the network.

---

## Model Architecture

MG-NECOLA uses a 3D V-Net architecture inherited from the `map2map` framework.

The architecture is an encoder-decoder network with:

- Two down-sampling stages
- Two up-sampling stages
- Residual connections
- Skip connections
- 3D convolutional layers
- Batch normalization
- Leaky ReLU activations
- Three input channels for the displacement components
- Three output channels for the predicted residual displacement components

The model receives padded input sub-volumes and outputs a corrected residual displacement field of size:

```text
3 x 128^3
```

The final corrected field is obtained through a global residual connection:

```text
Delta_corrected = Delta_MG-PICOLA + Delta_res
```

---

## Loss Function

The training objective combines real-space and Fourier-space constraints.

The real-space baseline loss is

```text
L_base = lambda_lag * L_lag + lambda_eul * L_eul + lambda_grad * L_grad
```

where:

- `L_lag` is the Lagrangian displacement loss.
- `L_eul` is the Eulerian overdensity loss obtained after mapping displacements to density.
- `L_grad` is a gradient-matching penalty that improves small-scale displacement structure.

A spectral loss is also included:

```text
L_spec = < w(k) [log P_pred(k) - log P_targ(k)]^2 >
```

This term focuses on high-k modes above

```text
k_min = 0.3 h/Mpc
```

The total loss used in this code is

```text
L_total = lambda_lag * L_lag
        + lambda_eul * L_eul
        + lambda_grad * L_grad
        + lambda_spec * L_spec
```

The default weights used in the provided training script are:

```text
lambda_lag  = 1.0
lambda_eul  = 3.0
lambda_grad = 2.0
lambda_spec = 1.0
```

The spectral loss uses:

```text
spec_kmin  = 0.3
spec_alpha = 2.0
L_sub      = 250.0
```

---

## Repository Structure

```text
.
├── m2m.py                  # Main map2map command-line interface
├── train.py                # Training loop with MG-NECOLA loss terms
├── test.py                 # Evaluation / inference routine
├── fields.py               # Dataset and field-loading utilities
├── gradient_loss.py        # Gradient-matching loss
├── spectral_loss.py        # Fourier-space power-spectrum loss
├── train_slurm.sh          # SLURM training script
├── models/                 # Neural-network architectures
├── data/                   # Dataset utilities
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/BayronO/MG-NECOLA.git
cd MG-NECOLA
```

Create or activate your Python environment. On the Michigan Tech cluster, the training script uses:

```bash
module use /mnt/it_software/easybuild/modules/all
module load Anaconda3/2022.10
module load CUDA/12.4.0
source ~/.bashrc
conda activate pylians
```

Install the required dependencies according to your local environment. The code requires PyTorch, NumPy, Matplotlib, and the dependencies inherited from `map2map`.

---

## Training

Training is launched through the `m2m.py` interface inherited from `map2map`.

### SLURM Training

The recommended way to train on the GPU cluster is:

```bash
sbatch train_slurm.sh
```

The provided SLURM script requests:

```text
partition:       mrigpu
nodes:           1
GPUs:            4
CPUs per task:   16
wall time:       30:00:00
```

It sets the distributed training variables:

```bash
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
export WORLD_SIZE=$((SLURM_NNODES * 4))
```

The training command used in `train_slurm.sh` is:

```bash
python m2m.py train \
  --train-in-patterns "/path/to/train/in/*/cola_*.npy" \
  --train-tgt-patterns "/path/to/train/tgt/*/quijote_*.npy" \
  --val-in-patterns "/path/to/val/in/*/cola_*.npy" \
  --val-tgt-patterns "/path/to/val/tgt/*/quijote_*.npy" \
  --in-norms cosmology.dis --tgt-norms cosmology.dis \
  --augment --crop 128 --pad 20 \
  --model vnet.VNet --callback-at . \
  --lr 1e-4 --optimizer Adam \
  --optimizer-args '{"betas": [0.9, 0.999], "weight_decay": 1e-4}' \
  --reduce-lr-on-plateau \
  --scheduler-args '{"factor": 0.5, "patience": 2, "threshold": 1e-3, "verbose": true}' \
  --batches 2 --epochs 150 --loader-workers 4 \
  --div-data --div-shuffle-dist 1 \
  --L-sub 250.0 \
  --lambda-lag 1.0 \
  --lambda-eul 3.0 \
  --lambda-grad 2.0 \
  --lambda-spec 1.0 \
  --spec-kmin 0.3 \
  --spec-alpha 2.0
```

The input and target patterns follow the convention:

```text
cola_*.npy      -> MG-PICOLA approximate displacement fields
quijote_*.npy   -> QUIJOTE-MG target displacement fields
```

---

## Checkpoints

During training, the code saves model states as:

```text
state_1.pt
state_2.pt
...
state_N.pt
```

A symbolic link points to the latest checkpoint:

```text
state.pt
```

Training logs are written to:

```text
logs/%x-%j.out
logs/%x-%j.err
```

where `%x` is the SLURM job name and `%j` is the job ID.

---

## Evaluation

The evaluation code loads a trained checkpoint and applies the model to test fields. The testing routine computes:

- Lagrangian displacement loss
- Eulerian density loss
- Total diagnostic loss
- Output fields assembled from model predictions

A typical test call follows the same `map2map` interface style and should provide:

```bash
python m2m.py test \
  --test-in-patterns "/path/to/test/in/*/cola_*.npy" \
  --test-tgt-patterns "/path/to/test/tgt/*/quijote_*.npy" \
  --in-norms cosmology.dis --tgt-norms cosmology.dis \
  --crop 128 --pad 20 \
  --model vnet.VNet --callback-at . \
  --load-state state.pt
```

Adjust the paths and checkpoint name according to your trained model location.

---

## Reported Results

According to the paper, MG-NECOLA achieves:

- Nearly sub-percent accuracy in the matter power spectrum up to `k ~ 1 h/Mpc`.
- Nearly sub-percent accuracy in the bispectrum for representative triangle configurations.
- Approximately 30% reduction in the mean particle displacement error compared with MG-PICOLA.
- Strong recovery of non-linear clustering lost by the approximate solver.
- Stable generalization across intermediate scalar-field strengths.
- Good performance across a broad Latin-hypercube parameter space.
- Accurate recovery of the LCDM limit.
- Accurate recovery of massive-neutrino suppression for `M_nu = 0.1, 0.2, 0.4 eV`.

The total time-to-solution is approximately:

```text
QUIJOTE-MG N-body:  ~1.8 x 10^6 CPU seconds
MG-PICOLA:          ~1.0 x 10^3 CPU seconds
MG-NECOLA:          ~1.2 x 10^3 seconds including MG-PICOLA + inference
```

This corresponds to a total speed-up of approximately:

```text
~1500x relative to full N-body
```

---

## Known Limitation

MG-NECOLA is a residual correction model. Its performance depends on the quality of the approximate MG-PICOLA input.

In strongly coupled scenarios involving both modified gravity and massive neutrinos, the paper finds that MG-PICOLA can produce degraded displacement fields. In that case, MG-NECOLA still recovers a significant fraction of the missing power, but it cannot fully reconstruct the true N-body result when the baseline solver fails too severely.

---

## Relation to map2map

This repository builds on the original `map2map` codebase:

```text
https://github.com/eelregit/map2map
```

The main extensions in this fork are:

- Application to modified gravity simulations.
- Pairing MG-PICOLA approximate fields with QUIJOTE-MG N-body targets.
- Residual learning for displacement-field correction.
- Gradient-matching loss for small-scale structure recovery.
- Spectral loss for power-spectrum consistency.
- SLURM-based multi-GPU training workflow.
- Physics validation using matter power spectrum, bispectrum, displacement errors, and extrapolation tests.

---

## Citation

If you use this repository, please cite:

```bibtex
@article{orjuelaquintana2026mgnecola,
  title = {MG-NECOLA: A Field-Level Emulator for f(R) Gravity and Massive Neutrino Cosmologies},
  author = {Orjuela-Quintana, J. Bayron and Reyes, Mauricio and Giusarma, Elena and Baldi, Marco and Kaushal, Neerav and Valenzuela-Toledo, Cesar A.},
  journal = {arXiv e-prints},
  year = {2026},
  eprint = {2604.19613},
  archivePrefix = {arXiv},
  primaryClass = {astro-ph.CO}
}
```

Please also cite the original `map2map` work if you use this fork.

---

## Acknowledgements

This code is based on `map2map`. The scientific application uses QUIJOTE-MG and MG-PICOLA simulations. The project acknowledges support from Michigan Technological University computing resources and collaborators involved in the MG-NECOLA paper.

---

## Status

This is an active research repository. Interfaces, paths, and training scripts may change as the project evolves.
