# Model Parallelism for AFNO (FourCastNet) — Master's Thesis Code

> **This is a fork of [NVIDIA Modulus](https://github.com/NVIDIA/physicsnemo)** (now PhysicsNeMo).
> Upstream code is NVIDIA's. My contribution is described below. The original README is kept in [`README_upstream.md`](README_upstream.md).

**Thesis:** *Modelparallelismus in Adaptiv Fourier Neural Operator Transformern für skalierbare datengetriebene Wettervorhersagen*
Robin Maurer · M.Sc. Computer Science · Karlsruhe Institute of Technology (KIT), Scientific Computing Center · 07/2024 – 01/2025
Supervisors: Prof. Dr. Achim Streit, Prof. Dr. Veit Hagenmeyer, Dr. Charlotte Debus
📄 Thesis PDF: <!-- TODO: Link einfügen (PDF im Repo oder KITopen) -->

## What I did

FourCastNet forecasts global weather from ERA5 reanalysis data (0.25° resolution) using Adaptive Fourier Neural Operators (AFNO). I extended the AFNO backbone with **model parallelism** and measured whether it scales better than plain data parallelism on the HoreKa supercomputer.

- **Three parallelization strategies** implemented:
  - *Global* (macro): split across the model, only 2 extra communication operations per forward/backward pass
  - *Local, hidden dimension*: split the hidden dimension before the Fourier transform
  - *Local, diagonal blocks*: split the block-diagonal weights after the Fourier transform
- **72 configurations** evaluated: 3 architectures × 4 GPU counts (8, 16, 32, 64) × 3 model-parallel sizes (2, 4, 8) × 2 scaling modes (strong / weak)
- **Overhead analysis**: communication vs. data preprocessing vs. compute, tracked with Weights & Biases
- Stack: Python, PyTorch, `torch.distributed` (NCCL), MPI, Slurm, W&B

## Results (honest summary)

- Model parallelism showed **no significant scaling advantage** under the given conditions. For stand-alone AFNONet, **pure data parallelism was the most efficient** option.
- Best single configuration: global strategy, model-parallel size 2 on 8 GPUs — about **+10 % throughput** over the data-parallel baseline (relative scaled speedup 1.11).
- **Local (fine-grained) strategies reached lower loss** than the global strategy.
- Training on 64 GPUs took about 4.5 h but used the GPUs inefficiently; the 8-GPU model reached an 18.5 % lower loss.

This is a negative result: it documents where and why model parallelism does not pay off for AFNO, and gives a starting point for follow-up work (e.g. the successor Spherical Fourier Neural Operator).

## Where my code is

| Path | Content |
|---|---|
| `modulus/models/afno/distributed/afno_mpi.py` | Model-parallel AFNO layers |
| `modulus/distributed/utils_mpi.py` | Communication utilities (gather/scatter, process groups) |
| `modulus/models/afno/afno.py` | Changes to the AFNO model |
| `*_horeka.sh`, `train_horeka.sh` | Slurm job scripts for HoreKa |

<!-- TODO: Prüfen, ob dieser Stand (letzter Commit 15.08.2024) dem Endstand der Experimente entspricht. Falls nicht, Satz ergänzen: "Code state as of 08/2024; final experiments are documented in the thesis." -->

**Not my work:** everything else in this repository (NVIDIA upstream), including the `modulus/models/sfno/` files, which were copied from NVIDIA Modulus to prepare follow-up experiments. Parts of the communication setup are adapted from [Heat (Helmholtz Analytics Toolkit)](https://github.com/helmholtz-analytics/heat).

## License

Apache 2.0, as in upstream (see [`LICENSE.txt`](LICENSE.txt)).
