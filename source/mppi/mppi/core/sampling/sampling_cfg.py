from dataclasses import dataclass, MISSING

from .delta_sampling import DeltaSampling
from .direct_sampling import DirectSampling
from .annealing_delta_sampling import AnnealingDeltaSampling
from typing import Type, List, Tuple

@dataclass
class SamplingCfg:

    control_dim: int = MISSING
    """ Control dimension """

    noise: list[int] = MISSING
    """ Noise """

    max_control: List[Tuple[float, float]] = MISSING
    """ Max control values """

    num_rollouts: int = MISSING
    """ Number of rollouts """

    num_timesteps: int = MISSING
    """ Number of timesteps. Equal to Horizon minus 1 """

@dataclass
class DirectSamplingCfg(SamplingCfg):
    """ Direct sampling configuration """

    class_type: type[DirectSampling] = DirectSampling

@dataclass(kw_only=True)
class DeltaSamplingCfg(SamplingCfg):
    """Delta sampling configuration"""

    max_delta: List[float] = MISSING
    """ Max delta per step """

    class_type: type[DeltaSampling] = DeltaSampling

@dataclass(kw_only=True)
class AnnealingDeltaSamplingCfg(DeltaSamplingCfg):
    """ Annealing delta sampling configuration """

    opt_iters: int = MISSING # maybe same as opt_iters?

    h_anneal_rate: float = MISSING

    n_anneal_rate: float = MISSING

    class_type: type[AnnealingDeltaSampling] = AnnealingDeltaSampling
