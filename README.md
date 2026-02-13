# ComPhy: Composing Physical Models with end-to-end Alignment
Code for the paper **[ComPhy: Composing Physical Models with end-to-end Alignment](https://openreview.net/forum?id=ER7zDJXtRI)** accepted at The Fourteenth International Conference on Learning Representations (ICLR 2026).

Please consider citing us

	@inproceedings{
        trenta2026comphy,
        title={ComPhy: Composing Physical Models with end-to-end Alignment},
        author={Alessandro Trenta and Andrea Cossu and Davide Bacciu},
        booktitle={The Fourteenth International Conference on Learning Representations},
        year={2026},
        url={https://openreview.net/forum?id=ER7zDJXtRI}
    }


## Overview
We introduce ComPhy, a modular framework for learning system of PDEs by assigning each module to a (some) specific equation(s) in the system. An alignment mechanism ensures that the modules are kept informed of each other, collectively building a solution. Implemented modules are PINNs and Neural Conservation Laws (NCL) models.

## Directory Structure

- **acoustics/** - Acoustic waves
- **euler-tori/** - Euler equations on Tori (see [euler-tori/README.md](euler-tori/README.md) for detailed instructions)
- **kovasznay_flow/** - Kovasznay flow (Navier-Stokes)
- **MHD/** - Magnetohydrodynamics
- **taylor_green_vortex/** - Taylor-Green vortex (Navier-Stokes)

## Installation

### Prerequisites

- Python 3.12 or higher
- Conda package manager

### Setup

Create and activate the conda environment:

```bash
conda create -n comphy python=3.12
conda activate comphy
pip install -r requirements.txt
```

Ensure you have PyTorch installed with CUDA support (if using GPU):

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126

```


## Running Experiments

To run experiments, first generate the data with the `generate.py` script an then use the `run.py` script available in acoustics, kovasznay_flow, MHD, and taylor_green_vortex folders.

#### Common Parameters

- **--device**: GPU/CPU device (e.g., `cuda:0`, `cuda:1`, `cpu`)
- **--weight_mode**: How to weight losses (`static` or `grad`)
- **--model_name**: Model architecture to use
- **--alignment_mode**: Type of alignment (`none`, `DERL`, `SOB`, `OUTL`)

#### Supported Models

Depends on the domain, but can include:
- **pinn**: Physics-Informed Neural Network
- **ncl**: Neural Conservation Laws
- **dualpinn**: ComPhy - 2xPINN
- **triplepinn**: ComPhy - 3xPINN
- **quadpinn**: ComPhy - 4xPINN
- **dualncl**: ComPhy - 2xNCL
- **pinnncl**: ComPhy - PINN+NCL

### Domain-Specific Instructions

#### Acoustics

```bash
cd acoustics
# Train models with specific parameters
python run.py --device cuda:0 --weight_mode grad --model triplepinn --alignment_mode OUTL
```

**Available Models**: `pinn`, `ncl`, `triplepinn`, `triplencl`

#### Kovasznay Flow

```bash
cd kovasznay_flow
# Train a model
python run.py --device cuda:0 --weight_mode static --model pinnncl --alignment_mode DERL
```

**Available Models**: `pinn`, `ncl`, `dualpinn`, `pinnncl`

#### MHD (Magnetohydrodynamics)

```bash
cd MHD
# Train a model
python run.py --device cuda:0 --weight_mode grad --model pinn --alignment_mode none
```

**Available Models**: `pinn`, `ncl`, `dualpinn`, `quadpinn`, `triplepinn`

#### Taylor-Green Vortex

```bash
cd taylor_green_vortex
# Train a model
python run.py --device cuda:0 --weight_mode static --model ncl --alignment_mode none
```

**Available Models**: `pinn`, `ncl`, `dualpinn`, `pinnncl`

#### Euler-Tori: Special Case

The Euler-Tori folder has a different structure with separate training scripts for each model. **See [euler-tori/README.md](euler-tori/README.md) for complete instructions.**

## Output and Results

Training results are typically stored in the root of each domain folder. After running experiments, use the evaluation and plotting scripts:

- **eval.py** or **eval_results.py**: Evaluate model performance and make plots 

## Notes

- Weight modes (`static` vs `grad`) affect how physical constraint losses are weighted during training
- Alignment modes affect how multiple physics objectives are composed together

## License

See LICENSE file for details.
