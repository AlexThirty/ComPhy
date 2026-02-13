# Euler-Tori: Euler Equations on Tori

This module implements the ComPhy models for solving Euler equations on tori domains. Unlike other domains in ComPhy, euler-tori uses individual training scripts for each model variant rather than a unified `run.py` interface.

## Overview

The Euler-Tori module contains implementations of:
- **PINN** (Physics-Informed Neural Network)
- **Dual PINN** (ComPhy - 2xPINN)
- **NCL** (Neural Conservation Laws)
- **Dual NCL** (ComPhy 2-NCL)
- **Triple PINN** (ComPhy - 3xPINN)
- **PINNNCL** (ComPhy - PINN+NCL) 

## Training Scripts

Each model has its own dedicated training script with specific parameters. All scripts accept command-line arguments for configuration.

### Common Arguments for All Training Scripts

- **--restart**: Restart training from scratch (overwrite existing results)
- **--device**: GPU device to use (e.g., `cuda:0`, `cuda:1`, `cuda:2`)
- **--weight_type**: Type of weight scaling (`grad` or others)
- **--mode**: Training mode (e.g., `DERL`, `SOB`)
- **--train_steps**: Number of maximum training steps per epoch
- **--epochs**: Number of epochs to train
- **--batch_size**: Batch size for training
- **--lr_init**: Initial learning rate
- **--name**: Experiment name
- **--inc_weight**: Weight for incompressibility/divergence loss
- **--init_weight**: Weight for initial condition loss
- **--mom_weight**: Weight for momentum loss
- **--div_weight**: Weight for divergence loss

## Training Individual Models

### Generate Solutions
First generate the data. The original .vtk files can be found in the original repo of NCL: [repo link](https://github.com/facebookresearch/neural-conservation-law)
```bash
python generate.py
```

### Train PINN

```bash
python train_pinn.py --restart --device cuda:0 --weight_type grad --epochs 600 --train_steps 1000
```

### Train Dual PINN

```bash
python train_dualpinn.py --restart --device cuda:0 --weight_type grad --mode DERL --epochs 600 --train_steps 1000
```

### Train Dual NCL (Neural Composite Learning)

```bash
python train_dualncl.py --restart --device cuda:0 --weight_type grad --mode SOB --epochs 600 --train_steps 1000
```

### Train Triple PINN

```bash
python train_triplepinn.py --restart --device cuda:0 --weight_type grad --mode DERL --epochs 600 --train_steps 1000
```

### Train ComPhy Model

```bash
python train_ComPhy.py --restart --device cuda:0 --weight_type grad --mode DERL --epochs 600 --train_steps 1000
```

### Train NCL

```bash
python train_ncl.py --restart --device cuda:0 --weight_type grad --epochs 600 --train_steps 1000
```
## Evaluation and Visualization
### Evaluate Results

```bash
python eval_results.py
```

Evaluates trained model performance and generates metrics.

## Loss Function Weights

You can customize the weight of different loss components:

- **--inc_weight**: Default ~1e-2, controls incompressibility constraint weight
- **--init_weight**: Default ~30, controls initial condition matching weight
- **--mom_weight**: Default ~3e-3, controls momentum equation weight
- **--div_weight**: Default ~1.0, controls divergence penalty weight

Example with custom weights:

```bash
python train_pinn.py --restart --device cuda:0 --inc_weight 1e-2 --init_weight 30 --mom_weight 3e-3 --div_weight 1.0
```