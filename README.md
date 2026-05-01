# MG-NECOLA

**MG-NECOLA** is a field-level deep-learning emulator for modified gravity cosmologies. It upgrades fast approximate **MG-PICOLA** simulations to near **QUIJOTE-MG N-body** accuracy using a 3D V-Net convolutional neural network, achieving a total speed-up of ~1500× relative to full N-body runs.

This repository is a fork and extension of [`map2map`](https://github.com/eelregit/map2map), adapted for **Hu-Sawicki f(R) gravity**, massive-neutrino generalization tests, and a physics-driven composite loss function for large-scale structure reconstruction.

**Paper:**

> **MG-NECOLA: A Field-Level Emulator for f(R) Gravity and Massive Neutrino Cosmologies**  
> J. Bayron Orjuela-Quintana, Mauricio Reyes, Elena Giusarma, Marco Baldi, Neerav Kaushal, and César A. Valenzuela-Toledo  
> [arXiv:2604.19613](https://arxiv.org/abs/2604.19613)

---

## Table of Contents

- [Overview](#overview)
- [Physical Motivation](#physical-motivation)
- [Key Results](#key-results)
- [Data](#data)
- [Model Architecture](#model-architecture)
- [Loss Function](#loss-function)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Computational Efficiency](#computational-efficiency)
- [Known Limitations](#known-limitations)
- [Relation to map2map and NECOLA](#relation-to-map2map-and-necola)
- [Citation](#citation)

---

## Overview

High-resolution N-body simulations provide accurate predictions for non-linear structure formation, but are computationally prohibitive for the large ensembles required in modified gravity (MG) analyses. MG-NECOLA bridges this gap by learning a **field-level residual correction**:

```
MG-PICOLA  ──[V-Net]──►  QUIJOTE-MG-like N-body output
```

Instead of generating cosmological fields from scratch, the network acts as a **residual corrector**. It takes approximate MG-PICOLA displacement fields as input and predicts the residual needed to match high-fidelity QUIJOTE-MG simulations:

```
Δ_res = Δ_QUIJOTE-MG − Δ_MG-PICOLA
```

The corrected displacement field is recovered by a global bypass connection:

```
Δ_corrected = Δ_MG-PICOLA + Δ_res
```

---

## Physical Motivation

In **Hu-Sawicki f(R) gravity**, an additional scalar degree of freedom mediates a fifth force that enhances gravity in low-density regions while recovering General Relativity in high-density environments via the **chameleon screening mechanism**. On sub-horizon scales, this modifies the Poisson equation through a scale- and time-dependent effective gravitational coupling:

```
G_eff(k, a) / G_N = 1 + (1/3) * k² / (k² + a²m²(a))
```

Gravity is enhanced by 4/3 on scales smaller than the scalaron Compton wavelength, while GR is recovered on large scales. These scale-dependent, environment-dependent effects are particularly challenging for approximate solvers like COLA, which rely on Lagrangian perturbation theory and lack the force resolution to solve the non-linear scalar field equations accurately. MG-NECOLA is designed to correct these small-scale errors.

---

## Key Results

Evaluated on the QUIJOTE-MG test set (10 independent realizations per f(R) label):

| Observable | MG-PICOLA error | MG-NECOLA error |
|---|---|---|
| Matter power spectrum P(k), k ≤ 1 h/Mpc | > 5% at k > 0.5 h/Mpc | **≲ 1%** |
| Bispectrum B(k₁, k₂, θ) | significant bias | **sub-percent** |
| Mean particle displacement ⟨δr⟩ | ~0.68 h⁻¹ Mpc | **~0.47 h⁻¹ Mpc (~30% improvement)** |

**Generalization tests (out-of-training-manifold):**

- Interpolation across scalar field strengths |fR0| ∈ [10⁻⁷, 10⁻⁴]: error < 1% up to k ~ 1 h/Mpc
- Latin hypercube (98 cosmologies, 6D parameter space {Ωm, Ωb, h, ns, σ8, fR0}): near-unity mean T(k) with tight 1σ variance
- ΛCDM limit (fR0 = 0): sub-percent recovery with **no spurious MG signal**
- Massive neutrinos (Mν = 0.1, 0.2, 0.4 eV), trained on massless: accurate suppression recovery up to Mν ≤ 0.4 eV

---

## Data

The model is trained on **100 paired simulations** from the fixed-cosmology QUIJOTE-MG suite, matching MG-PICOLA approximate runs to QUIJOTE-MG N-body targets particle-by-particle.

**Fixed cosmology:**
```
Ωm = 0.3175,  Ωb = 0.049,  h = 0.6711,  ns = 0.9624,  σ8 = 0.834,  Mν = 0
```

**f(R) labels:**
```
fR0 = {-5×10⁻⁷, -5×10⁻⁶, -5×10⁻⁵, -5×10⁻⁴}   (25 simulations each)
```

**Simulation specs:** 512³ CDM particles, box size L = 1000 h⁻¹ Mpc, evolved from z = 127 to z = 0.

**Data split:**
```
70 simulations  →  training
20 simulations  →  validation
10 simulations  →  testing
```

**Field layout:** Each full displacement field has shape `3 × 512³` (three Cartesian components). During training, fields are tiled into cubic sub-volumes of size `3 × 128³` (~250 h⁻¹ Mpc per side), periodically padded by 20 voxels before entering the network.

---

## Model Architecture

MG-NECOLA uses the **3D V-Net** architecture from `map2map` (`models/vnet.py`), implementing a symmetric encoder–decoder (U-shaped) design with residual and skip connections at multiple resolutions.

```
Input: 3 × 168³  (128³ sub-volume + 20 voxel periodic padding)
         │
    [Contracting block 1]  ── 3³ convs + residual ──►  166³ → 164³
         │ (stride-2 2³ conv)
         ▼ 82³
    [Contracting block 2]  ── 3³ convs + residual ──►  80³ → 78³
         │ (stride-2 2³ conv)
         ▼ 39³ (bottleneck)
    [Expansive block 1]    ── transposed conv ──►  70³
         │ + skip connection (crop + concat)
    [Expansive block 2]    ── transposed conv ──►  128³
         │
Output: 3 × 128³  (predicted residual displacement Δ_res)
```

## Loss Function

The training objective combines real-space and Fourier-space constraints:

```
L = λ_lag · L_lag  +  λ_eul · L_eul  +  λ_grad · L_grad  +  λ_spec · L_spec
```

| Term | Description |
|---|---|
| `L_lag` | RMSE of predicted vs. target **Lagrangian displacement field** |
| `L_eul` | RMSE of predicted vs. target **Eulerian overdensity** (mapped via `models/lag2eul.py`) |
| `L_grad` | Gradient-matching penalty — regularizes displacement predictions, preventing over-smoothing and recovering small-scale MG features |
| `L_spec` | Fourier-space power-spectrum loss, weighted as `w(k) = (k/k_Ny)^(2α) Θ(k − k_min)` to focus on non-linear scales |

The spectral loss is:
```
L_spec = ⟨ w(k) [log P_pred(k) − log P_targ(k)]² ⟩
```

**Default weights (from paper ablation study):**
```
λ_lag  = 1.0
λ_eul  = 3.0
λ_grad = 2.0
λ_spec = 1.0

spec_kmin  = 0.3 h/Mpc   (onset of mildly non-linear regime)
spec_alpha = 2.0
L_sub      = 250.0 h⁻¹ Mpc
```

The gradient term `L_grad` was identified as critical in the ablation study: removing it causes errors 2–2.5× larger in the non-linear regime (k > 0.5 h/Mpc) and breaks sub-percent accuracy. This is the principal departure from the original NECOLA loss.

---

## Repository Structure

The code lives inside the `map2map/` Python package. Only the most relevant files are highlighted; auxiliary modules inherited from upstream `map2map` (GAN utilities, alternative architectures, plotting helpers, etc.) are kept but not used by the MG-NECOLA pipeline.

```
map2map/
├── data/             # Dataset, distributed sampler, field normalizations
│   ├── fields.py     # FieldDataset: file globbing, cropping, padding, augmentation
│   ├── sampler.py    # DistFieldSampler for multi-GPU training
│   └── norms/        # Per-field normalization callbacks (e.g. cosmology.dis)
│
├── models/           # Network architectures and physics ops
│   ├── vnet.py       # ★ V-Net used by MG-NECOLA
│   ├── lag2eul.py    # Lagrangian → Eulerian density mapping (used in L_eul)
│   ├── power.py      # Power-spectrum utilities (diagnostics and L_spec)
│   ├── narrow.py     # Cropping for skip connections
│   └── ...           # Conv blocks, U-Net, GAN modules (unused here)
│
├── utils/            # gradient_loss, power_spectrum_loss, checkpoint I/O,
│                     #   plotting, dynamic attribute import
│
├── train.py          # ★ Training loop — DDP, composite loss, TensorBoard logging
├── test.py           # Evaluation / inference routine
├── main.py           # Top-level dispatch (train / test subcommands)
├── args.py           # Argument parser (all --lambda-*, --spec-*, --L-sub flags)
└── __init__.py
```

The two files most relevant to this fork are **`train.py`** (composite loss assembly and DDP loop) and **`models/vnet.py`** (the V-Net architecture). Dataset and augmentation logic in `data/fields.py` is inherited from upstream `map2map` with no MG-specific changes.

### What `train.py` does

`train.py` orchestrates multi-GPU distributed training:

1. **Spawn**: `node_worker` → `gpu_worker` spawns one process per GPU (`torch.multiprocessing.spawn`), pinning each to its own device with `torch.cuda.set_device`.
2. **Distributed init**: NCCL process group via `dist_init`, with global rank computed from `SLURM_NODEID` for multi-node compatibility.
3. **Data**: builds `FieldDataset` + `DistFieldSampler` for both train and validation, with periodic-padding crops of size 128³.
4. **Model**: instantiates the V-Net (`models.vnet.VNet`), wraps in `DistributedDataParallel`, and optionally restores from `checkpoint.pt`.
5. **Loss assembly** (per batch):
   - Lagrangian RMSE on raw displacement,
   - Eulerian RMSE after `lag2eul` projection,
   - `gradient_loss` on displacements,
   - `power_spectrum_loss` with `kmin`, `alpha`, `L_sub`,
   - combined with the four `λ_*` weights from `args.py`.
6. **Logging**: per-batch and per-epoch losses, gradient norms (first/last layer), field slices, and power spectra written to TensorBoard from rank 0.
7. **Scheduling**: `ReduceLROnPlateau` halves the learning rate after `patience` epochs without validation improvement.

---

## Installation

Clone the repository:
```bash
git clone https://github.com/BayronO/MG-NECOLA.git
cd MG-NECOLA
```

**Michigan Tech cluster:**
```bash
module use /mnt/it_software/easybuild/modules/all
module load Anaconda3/2022.10
module load CUDA/12.4.0
source ~/.bashrc
conda activate pylians
```

**Dependencies:** PyTorch (with CUDA), NumPy, Matplotlib, TensorBoard, and the `map2map` dependencies. Install according to your local environment.

---

## Training

Training is launched through the `m2m.py` CLI, which dispatches to `train.py`.

### SLURM

```bash
sbatch train_slurm.sh
```

The SLURM script requests:
```
partition:       mrigpu
nodes:           1
GPUs:            4 (one process per GPU via torch.multiprocessing.spawn)
CPUs per task:   16
```

### Training command

```bash
python m2m.py train \
  --train-in-patterns  "/path/to/train/in/*/cola_*.npy" \
  --train-tgt-patterns "/path/to/train/tgt/*/quijote_*.npy" \
  --val-in-patterns    "/path/to/val/in/*/cola_*.npy" \
  --val-tgt-patterns   "/path/to/val/tgt/*/quijote_*.npy" \
  --in-norms cosmology.dis --tgt-norms cosmology.dis \
  --augment --crop 128 --pad 20 \
  --model vnet.VNet --callback-at . \
  --lr 1e-4 --optimizer Adam \
  --optimizer-args '{"betas": [0.9, 0.999], "weight_decay": 1e-4}' \
  --reduce-lr-on-plateau \
  --scheduler-args '{"factor": 0.5, "patience": 2, "threshold": 1e-3, "verbose": true}' \
  --batches 2 --epochs 100 --loader-workers 4 \
  --div-data --div-shuffle-dist 1 \
  --L-sub 250.0 \
  --lambda-lag  1.0 \
  --lambda-eul  3.0 \
  --lambda-grad 2.0 \
  --lambda-spec 1.0 \
  --spec-kmin  0.3 \
  --spec-alpha 2.0
```

File naming convention:
```
cola_*.npy      →  MG-PICOLA approximate displacement fields  (input)
quijote_*.npy   →  QUIJOTE-MG N-body displacement fields      (target)
```

### Checkpoints and logs

Model states are saved as `state_1.pt`, `state_2.pt`, ..., with a symlink `checkpoint.pt` pointing to the latest. TensorBoard logs (loss curves, power spectra, field slices) are written to `runs/`.

---

## Evaluation

```bash
python m2m.py test \
  --test-in-patterns  "/path/to/test/in/*/cola_*.npy" \
  --test-tgt-patterns "/path/to/test/tgt/*/quijote_*.npy" \
  --in-norms cosmology.dis --tgt-norms cosmology.dis \
  --crop 128 --pad 20 \
  --model vnet.VNet --callback-at . \
  --load-state checkpoint.pt
```

The test routine computes Lagrangian displacement loss, Eulerian density loss, total loss, and assembles output fields from patches.

---

## Computational Efficiency

| Method | Hardware | Time per realization | Speed-up vs. N-body |
|---|---|---|---|
| QUIJOTE-MG (N-body) | CPU | ~1.8 × 10⁶ s (~500 CPU-h) | 1× |
| MG-PICOLA | CPU | ~1.0 × 10³ s | ~1800× |
| **MG-NECOLA** (COLA + inference) | CPU + GPU | **~1.2 × 10³ s** | **~1500×** |

Neural network inference alone takes ~180 seconds per realization on a single GPU. The total pipeline (MG-PICOLA + inference) remains ~1500× faster than full N-body, enabling generation of the large mock catalogs required for covariance matrix estimation in stage-IV galaxy surveys.

---

## Known Limitations

MG-NECOLA is a **residual corrector**: its performance is intrinsically tied to the quality of the MG-PICOLA input.

- In the **combined f(R) + massive neutrino** regime with high Mν (e.g., the Latin hypercube cases in Table 2 of the paper), MG-PICOLA itself breaks down, producing displacement fields that rapidly diverge from N-body truth even at mildly non-linear scales. In these cases, MG-NECOLA recovers a significant fraction of the missing power but cannot fully reconstruct the true N-body result.
- Performance at **k > 1 h/Mpc** extends slightly beyond the simulation trust region (kNy ≈ 1.61 h/Mpc; reliable to ~1.1 h/Mpc) and should be interpreted with caution.
- The model is trained on **fixed background cosmology**. Generalization to varying cosmologies (validated on a 6D Latin hypercube) maintains < 5% error, but accuracy degrades relative to the fixed-cosmology case.

---

## Relation to map2map and NECOLA

This repository builds on [`map2map`](https://github.com/eelregit/map2map) and the original [NECOLA](https://arxiv.org/abs/2111.02422) framework (Kaushal et al. 2022). The principal extensions introduced in MG-NECOLA are:

- Application to **Hu-Sawicki f(R) modified gravity** simulations.
- Pairing MG-PICOLA approximate fields with QUIJOTE-MG N-body targets for residual learning.
- **Composite loss function** replacing the NECOLA logarithmic product loss with an explicit gradient-matching term (`L_grad`) and a Fourier-space spectral term (`L_spec`).
- Multi-GPU distributed training (`DistributedDataParallel`) via SLURM (in `train.py`).
- Physics validation using matter power spectrum, bispectrum, displacement residuals, and out-of-distribution generalization tests.

---

## Citation

If you use this code, please cite the MG-NECOLA paper:

```bibtex
@article{orjuelaquintana2026mgnecola,
  title   = {MG-NECOLA: A Field-Level Emulator for f(R) Gravity and Massive Neutrino Cosmologies},
  author  = {Orjuela-Quintana, J. Bayron and Reyes, Mauricio and Giusarma, Elena
             and Baldi, Marco and Kaushal, Neerav and Valenzuela-Toledo, Cesar A.},
  journal = {arXiv e-prints},
  year    = {2026},
  eprint  = {2604.19613},
  archivePrefix = {arXiv},
  primaryClass  = {astro-ph.CO}
}
```

Please also cite the original `map2map` work and NECOLA if you use this fork:

```bibtex
@article{kaushal2022necola,
  title   = {NECOLA: Toward a Universal Field-level Cosmological Emulator},
  author  = {Kaushal, Neerav and Villaescusa-Navarro, Francisco and Giusarma, Elena and others},
  journal = {Astrophys. J.},
  volume  = {930},
  pages   = {115},
  year    = {2022},
  doi     = {10.3847/1538-4357/ac5c4a}
}
```

---

## Acknowledgements

This project uses QUIJOTE-MG simulations (Baldi & Villaescusa-Navarro 2025) and MG-PICOLA (Wright et al. 2017; Winther et al. 2017) as baseline solver. The authors thank Francisco Villaescusa-Navarro for insightful discussions and the IT Department at Michigan Technological University for computing support.

---

## Status

This is an active research repository. Training scripts, interfaces, and paths may evolve as the project develops. Trained models, predictions, and extracted statistics from the test and extrapolation sets are hosted at [github.com/BayronO/MG-NECOLA](https://github.com/BayronO/MG-NECOLA).
