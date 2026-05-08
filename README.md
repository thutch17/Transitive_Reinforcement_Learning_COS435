# COS435 Final Project: Replication and Benchmarking of Transitive Reinforcement Learning (TRL) on OGBench

By Luke Cho (lc1023@princeton.edu), Veronica Kuo (vk7364@princeton.edu), Cheryl Li (cl7672@princeton.edu), Tate Hutchins (th5879@princeton.edu), Maya Doron-Repa (md3317@princeton.edu)

This repository contains our implementation of TRL, along with the OGBench library used to run our benchmarks.

To ensure compatibility with the original OGBench implementation, we included a full copy of the original OGBench repository. We also used existing algorithms in the library to help scaffold our implementation of TRL and maintain consistency with the broader OGBench framework.

The original OGBench README has also been preserved for reference.

## Important Files

### Our TRL Implementation

Our implementation of TRL can be found at:

`ogbench-master/impls/agents/trl.py`

### Main File

The primary script used to run TRL experiments is:

`ogbench-master/impls/main.py`

### Hyperparameter Shell Files

We used multiple shell files to run different experiments. These can be found under:

`ogbench-master/impls`

The primary files are:

- `test_hyperparams.sh`
  - Used for smoke testing to ensure the algorithm functions correctly before running long evaluations.

- `eval_hyperparameters.sh`
  - Used for benchmarking on OGBench to reproduce results from Park et al. (2026) as well as evaluate several novel environments.

- `ablation_hyperparameters.sh`
  - Used for ablation studies reproducing results from Park et al.

### Data Collection Notebooks

We also used several Python/Colab notebooks to collect and visualize results.

The most important notebooks are:

- `ogbench_evaluations.ipynb`
  - Used for benchmarking on OGBench and evaluating novel environments in conjunction with `eval_hyperparameters.sh`.

- `ablation_evaluations.ipynb`
  - Used for ablation studies in conjunction with `ablation_hyperparameters.sh`.

- `graphs.ipynb`
  - Used to visualize experimental results for the final write-up.

Other notebooks, such as `trl_pointmaze_colab` and `trl_pointmaze_results`, were used for preliminary testing.

## Branches

### `smarter-subgoals`

This branch was used to develop extensions and modifications to the original TRL algorithm intended to improve performance.

Most files are organized similarly to the `main` branch.