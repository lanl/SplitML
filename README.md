# SplitML

Signal Processing Library for Interference rejecTion via Machine Learning.

## Installation/setup

Set up Anaconda environment:
```
conda env create -f environment.yml
```

Activate environment:
```
conda activate splitml
```

Install Pytorch 1.12 (see [Pytorch website](https://pytorch.org/get-started/locally/) for OS, CPU/GPU options):
```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch-nightly
```

Install `splitml` package locally:
```
pip install -e .
```

## Copyright notice

© 2022. Triad National Security, LLC. All rights reserved.
This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.
