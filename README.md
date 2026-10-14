# Model Parallelism for AFNO — First Attempt (Master's Thesis)
 
> ⚠️ **Final thesis code and results: [FourCastNet fork](https://github.com/RamlatchxRamspeicher/FourCastNet).**
> This repository holds the first approach (07–08/2024), built on NVIDIA Modulus. Modulus did not run on the HoreKa system; after trying to prune it down, I moved the work to the original FourCastNet code base.
 
> **This is a fork of [NVIDIA Modulus](https://github.com/NVIDIA/physicsnemo)** (now PhysicsNeMo).
> Upstream code is NVIDIA's. My contribution is described below. The original README is kept in [`README_upstream.md`](README_upstream.md).
 
**Thesis:** *Modelparallelismus in Adaptiv Fourier Neural Operator Transformern für skalierbare datengetriebene Wettervorhersagen*
Robin Maurer · M.Sc. Computer Science · Karlsruhe Institute of Technology (KIT), Scientific Computing Center · 07/2024 – 01/2025
📄 Thesis: [Full text (German, PDF)](Modelparallelism_in_Adaptive_Fourier_Neural_Operator_Transformers_for_Scalable_Data_Driven_Weatherforecasting.pdf)
 
## What is here
 
A first implementation of model-parallel AFNO layers and MPI-based communication inside Modulus, plus Slurm scripts for HoreKa.
 
| Path | Content |
|---|---|
| `modulus/models/afno/distributed/afno_mpi.py` | Model-parallel AFNO layers (first version) |
| `modulus/distributed/utils_mpi.py` | Communication utilities (gather/scatter, process groups) |
| `modulus/models/afno/afno.py` | Changes to the AFNO model |
| `*_horeka.sh`, `train_horeka.sh` | Slurm job scripts for HoreKa |
 
**Not my work:** everything else in this repository (NVIDIA upstream), including the `modulus/models/sfno/` files, which were copied from NVIDIA Modulus to prepare follow-up experiments. Parts of the communication setup are adapted from [Heat (Helmholtz Analytics Toolkit)](https://github.com/helmholtz-analytics/heat).
 
## License
 
Apache 2.0, as in upstream (see [`LICENSE.txt`](LICENSE.txt)).