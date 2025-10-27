import torch
from typing import TYPE_CHECKING

from .delta_sampling import DeltaSampling

if TYPE_CHECKING:
    from .sampling_cfg import SamplingCfg

class DirectSampling(DeltaSampling):
    """
    Class for Dynamics modelling
    """
    def __init__(
        self,
        sampling_cfg: 'SamplingCfg',
        num_envs: int,
        dtype = torch.float32,
        device = torch.device("cuda"),
    ):
        super().__init__(sampling_cfg=sampling_cfg, num_envs=num_envs, dtype=dtype, device=device)
        self.temperature = self.cfg.temperature
        self.scaled_dt = self.cfg.scaled_dt

        self.CTRL_NOISE = torch.zeros((self.nu, self.nu), device=self.d, dtype=self.dtype)
        for i in range(self.nu):
            self.CTRL_NOISE[i, i] = self.cfg.noise[i]

        self.CTRL_NOISE_inv = torch.inverse(self.CTRL_NOISE)
        self.CTRL_NOISE_MU = torch.zeros((self.num_envs, self.nu), dtype=self.dtype, device=self.d)

        ## for torchscript we have to initialize these things to same shape and size as what we'll use later
        torch.manual_seed(0)

        self.normal_noise = torch.randn((self.num_envs, self.K, self.T, self.nu), device=self.d, dtype=self.dtype)
        matmul = torch.einsum('nktd,dh->nkth', self.normal_noise, self.CTRL_NOISE)
        self.noise = matmul + self.CTRL_NOISE_MU[:, None, None, :]

    def sample(self, prev_controls: torch.Tensor=None):
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

        self._mean = prev_controls[:, 0:1, :].unsqueeze(-2) # [num_envs, 1, T, nu]

        #  Added Noise
        self.normal_noise = torch.randn((self.num_envs, self.K, self.T, self.nu), device = self.d, dtype = self.dtype)
        scaled_noise = torch.einsum('nktd,dh->nkth', self.normal_noise, self.CTRL_NOISE)
        self.noise = scaled_noise + self.CTRL_NOISE_MU[:, None, None, :]

        controls = self._mean + self.noise # [num_envs, K, T, nu]

        # Control space constraints through clamping
        if self.max_control is not None:
            for i in range(len(self.max_control)):
                controls[..., i] = torch.clamp(controls[..., i], self.max_control[i][0], self.max_control[i][1])

        return controls