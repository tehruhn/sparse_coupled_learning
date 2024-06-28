# Sparse Coupled Learning Solver

## Tarun Raheja, Menachem Stern, Andrea Liu

This thesis explores the paradigm of embedding learning in physical networks, diverging from traditional digital computational approaches. By leveraging contrastive Hebbian learning and localized coupled learning mechanisms, we demonstrate that physical systems can evolve to exhibit learning capabilities. We develop efficient algorithms for linear networks and study their properties, exploring phenomena such as two-phase learning, the impact of network structure, and scaling behavior. Our findings highlight the importance of network design in creating efficient and effective learning systems. This work showcases the feasibility and benefits of in-material learning, opening up new avenues for innovations in adaptive materials, neuromorphic computing, and autonomous systems.


This repository contains a Python implementation of a Linear Network Solver capable of initializing from various sources, solving for particular tasks, and optimizing network conductances.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Classes](#classes)
  - [LinearNetwork](#linearnetwork)
  - [LinearNetworkSolver](#linearnetworksolver)
- [Experiments](#experiments)
  - [Topology Experiment](#topology-experiment)
- [Utilities](#utilities)
  - [Data Utils](#data-utils)
  - [Graph Utils](#graph-utils)

## Installation

To use this Linear Network Solver, you need to have Python installed. Clone the repository and install the required dependencies:

```bash
git clone git@github.com:tehruhn/sparse_coupled_learning.git
cd sparse_coupled_learning
pip install -r requirements.txt
```

It is recommended that you set up a conda environment as well. Miniconda is preferred : [Installation Guide for Miniconda](https://docs.anaconda.com/miniconda/miniconda-install/)
After setting it up, make an environment with Python 3.10:
```bash
conda create -n research python=3.10
```

## Usage

Import the necessary classes and create instances of `LinearNetwork` and `LinearNetworkSolver`:

```python
from CLSolver.LinearNetwork import LinearNetwork
from CLSolver.LinearNetworkSolver import LinearNetworkSolver

# Create a LinearNetwork instance
linNet = LinearNetwork(input_data)

# Create a LinearNetworkSolver instance
solver = LinearNetworkSolver(linNet)
```

## Classes

### LinearNetwork

The `LinearNetwork` class represents a Linear Network and can be initialized from different sources:
- A pickle file containing a graph dictionary
- A pickle file with a NetworkX graph object
- Directly from a NetworkX graph object
- Uninitialized

### LinearNetworkSolver

The `LinearNetworkSolver` class takes a `LinearNetwork` instance and solves it for a particular task. It provides methods for setting up training tasks, computing costs, updating conductances, and performing optimization trials.

## Experiments

### Topology Experiment

The `topology_experiment.py` script tests the performance of the Linear Network Solver on different network topologies. It generates random regression data and runs the solver on various topologies, comparing the final cost values.

## Utilities

### Data Utils

The `data_utils.py` module provides functions for generating random regression data and specific regression data for experiments.

### Graph Utils

The `graph_utils.py` module contains functions for creating different types of networks, such as wide networks, low connectivity networks, random networks, and networks with specific topologies. It also includes a function for drawing wide networks.
