# MPPI

This submodule was written as part of our work, ["Model Predictive Adversarial Imitation Learning for Planning from Observation"](https://arxiv.org/abs/2507.21533v1). Model Predictive Path Integral (MPPI) control is a component of the introduced MPAIL algorithm and was written wholly independent of any upstream (MPAIL) features.

This implementation is designed for integration with Isaac Lab as well as deployment on-robot as a `torch.nn.Module`. It is parallelized across environments/agents in addition to standard batch trajectory parallelization.
