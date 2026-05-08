### COS435 Final Project: Replication and Benchmarking of Transitive Reinforcement Learning (TRL) on OGBench

By Luke Cho (lc1023@princeton.edu), Veronica Kuo (vk7364@princeton.edu), Cheryl Li (cl7672@princeton.edu), Tate Hutchins (th5879@princeton.edu), Maya Doron-Repa (md3317@princeton.edu)

This is our public github repo that contains our implementation of TRL, along with the OGBench library that was used to run our benchmarks. 
To ensure that none of our code has issues with the original OGBench implementation, we have directly copied the original OGBench repository in full
We used other algorithms in the library to help scaffold our implementation of TRL and ensure compatibility with the overall OGBench library.
We have also kept the OGBench README for reference

## Important files that we implemented:
# Our implementation of TRL: 
Our implementation of TRL can be found in \ogbench-master\impls\trl.py

# Hyperparameter shell files:
We used multiple shell files to run different experiments. They can be found under \ogbench-master\utils
The files that we used are as follows:
test_hyperparams.sh: used for smoke testing to ensure our algorithm functions before running ~1 hour long evaluations
eval_hyperparameters.sh: used for benchmarking on OGBench for reproduction of the results in Park et al. 2026, as well as a few novel environments
ablation_hyperparameters.sh: used for the ablation studies for reproduction of the results in Park et al. 2026

# Data Collection Notebooks:
We also used several python (Colab) notebooks to collect and visualize our data.
The most important notebooks we used are as follows:
ogbench_evaluations.ipynb: used for benchmarking on OGBench for reproduction of the results in Park et al. 2026, as well as a few novel environments in conjunction with eval_hyperparameters
ablation_evaluations.ipynb: used for the ablation studies for reproduction of the results in Park et al. 2026 in conjunction with ablation_hyperparameters.sh
graphs.ipynb: used to visualize data for the write-up
Other notebooks such as trl_pointmaze_colab and trl_pointmaze_results were used for preliminary testing

## Branches:
# smarter-subgoals:
This branch was used to develop extensions and modifications to the original TRL algorithm to help improve performance
Most of the files should be arranged similarly to the main branch