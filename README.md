# Wiser than the Wisest of Crowds: The Asch Effect and Polarization Revisited

This repository contains code for our ECML-PKDD submission (link...).
We provide methods to minimize and maximize the MSE and the polarization in a network
through selecting stooges based on

1. a greedy heuristic proposed in our paper,
3. maximum degree, or
4. betweenness-centrality.

We further provide synthetic and real-world graphs on which we apply our methods.

## Running the Code

Installation:

> ???

Compile cython:

> python setup.py build_ext --inplace

We can run a small example using

> python local_run.py

which places a figure under `plots`. All results are cached in the `cache`
folder. Empty the folder to re-run experiments.

## Reproducing the Experiments

Code for plots produced in the paper are in [experiments.py](experiments.py) and
can be via the setups found in [local_run.py](local_run.py)

## File Structure

- [cython_mse_graph_calculator.pyx](cython_mse_graph_calculator.pyx): Computes approximate equilibirum opinions
- [mse_stooges_resistance_greedy.py](mse_stooges_resistance_greedy.py): Contains our greedy approach
- [graph_construction.py](graph_construction.py): Generate synthetic graphs and load real-world graphs
