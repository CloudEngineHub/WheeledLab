'''
Inspired by "Full-Order Sampling-Based MPC for Torque-Level Locomotion Control via Diffusion-Style Annealing" by
Haoru Xue, Chaoyi Pan, Zeji Yi, Guannan Qu, Guanya Shi (Carnegie Mellon University, USA)
https://ieeexplore.ieee.org/abstract/document/11127320
'''

import torch

from typing import TYPE_CHECKING

from .delta_sampling import DeltaSampling

if TYPE_CHECKING:
    from .sampling_cfg import AnnealingDeltaSamplingCfg

class AnnealingDeltaSampling(torch.nn.Module):
    """
    Class for Dynamics modelling
    """
    def __init__(
        self,
        sampling_cfg: 'AnnealingDeltaSamplingCfg',
        num_envs: int,
        dtype = torch.float32,
        device = torch.device("cuda"),
    ):
        super().__init__()
        self.dtype = dtype
        self.d = device
        self.cfg = sampling_cfg

        self.nu = self.cfg.control_dim
        self.K = self.cfg.num_rollouts
        self.T = self.cfg.num_timesteps
        self.num_envs = num_envs

        self.sigma_0 = torch.zeros((self.num_envs, self.nu, self.nu), device=self.d, dtype=self.dtype)

        # more general way to set sigma_0, allows for modularity across action spaces
        for i in range(self.nu):
            self.sigma_0[:, i, i] = self.cfg.noise[i]

        self.sigma_0_inv = torch.inverse(self.sigma_0)
        self.sigma_0_mu = torch.zeros((self.num_envs, self.nu), dtype=self.dtype, device=self.d)

        ## for torchscript we have to initialize these things to same shape and size as what we'll use later
        torch.manual_seed(0)

        random_noise = torch.randn((self.num_envs, self.K, self.T, self.nu), device=self.d, dtype=self.dtype)
        matmul = torch.einsum('nktd,ndh->nkth', random_noise, self.sigma_0)
        self.noise = matmul + self.sigma_0_mu[:, None, None, :]

        self.max_delta = self.cfg.max_delta
        self.max_control = self.cfg.max_control

    def sample(self, iter: int, prev_controls: torch.Tensor=None, state=None):
        '''
        :param: state: torch.Tensor of shape (num_envs, state_dim)
        :param: U: torch.Tensor of shape (num_envs, T, nu)
        :param: (Optional) prev_controls: torch.Tensor of shape (num_envs, T, nu)

        init_controls is the initial guess
        sampling is done in the delta control space
        add this to the previous delta_controls
        integrate delta_controls and add previous controls to get controls
        find new noise after clamping
        find perturbation cost
        return controls, perturbation cost
        '''

        if prev_controls is None:
            prev_controls = torch.zeros((self.num_envs, self.T, self.nu), device=self.d, dtype=self.dtype)

        # Compute delta controls from previous control
        U_mu = torch.diff(
            prev_controls, dim=-2,
            append=torch.zeros_like(prev_controls[:, -1:, :])
        ) # [num_envs, T, nu]
        u0 = prev_controls[:, 0:1, :]

        #  Added Noise
        _noise = torch.randn((self.num_envs, self.K, self.T, self.nu), device = self.d, dtype = self.dtype)
        _noise = torch.einsum('nktd,ndh->nkth', _noise, self.sigma_0)
        schedule = torch.ones((self.T), device=self.d, dtype=self.dtype)

        # Annealing schedule
        if self.cfg.h_anneal_rate:
            schedule *= (torch.arange(self.T, device=self.d, dtype=self.dtype) - self.T) / (self.T * self.cfg.h_anneal_rate) # [T]

        if self.cfg.n_anneal_rate:
            schedule *= (self.cfg.opt_iters - iter) / (self.cfg.opt_iters * self.cfg.n_anneal_rate) # scalar

        _noise = _noise * schedule.unsqueeze(-1)

        self.noise = _noise + self.sigma_0_mu[:, None, None, :]
        # scale and add mean

        delta_U = U_mu.unsqueeze(-3) + self.noise

        # Delta control space constraints through clamping
        if self.max_delta is not None:
            for i in range(len(self.max_delta)):
                delta_U[..., i] = torch.clamp(delta_U[..., i], -self.max_delta[i], self.max_delta[i])

        controls = prev_controls[:, 0:1, :].unsqueeze(-2) + torch.cumsum(delta_U, dim=-2) # [num_envs, K, T, nu]

        # Control space constraints through clamping
        if self.max_control is not None:
            for i in range(len(self.max_control)):
                ## for mushr: car can't go in reverse, can't have more than 50 % speed
                controls[..., i] = torch.clamp(controls[..., i], self.max_control[i][0], self.max_control[i][1])

        return controls