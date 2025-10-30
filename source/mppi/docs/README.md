# MPPI

This submodule was written as part of our work, ["Model Predictive Adversarial Imitation Learning for Planning from Observation"](https://arxiv.org/abs/2507.21533v1). Model Predictive Path Integral (MPPI) control is a component of the introduced MPAIL algorithm and was written wholly independent of any upstream (MPAIL) features.

This implementation is designed for integration with Isaac Lab as well as deployment on-robot as a `torch.nn.Module`. It is parallelized across environments/agents in addition to standard batch trajectory parallelization.

Here's an episode's rollout visualization of 4 agents with "go-to (10,10)" and "target-velocity 2 m/s" costs. Each has their own elevation map as measured by IsaacLab's height scanner sensor.

https://github.com/user-attachments/assets/b26381a7-c262-4816-86ad-573adeef975c

This is what the simulation looks like. Notice that it's a screen-recording and is in real-time:

https://github.com/user-attachments/assets/38470b7e-f0a4-4546-9294-73d4e953db80

If you use this in your work, please consider citing:

@misc{han2025modelpredictiveadversarialimitation,
      title={Model Predictive Adversarial Imitation Learning for Planning from Observation}, 
      author={Tyler Han and Yanda Bao and Bhaumik Mehta and Gabriel Guo and Anubhav Vishwakarma and Emily Kang and Sanghun Jung and Rosario Scalise and Jason Zhou and Bryan Xu and Byron Boots},
      year={2025},
      eprint={2507.21533},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2507.21533}, 
}
