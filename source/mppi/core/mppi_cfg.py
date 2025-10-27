from dataclasses import dataclass

from mppi.core.cost import CostBaseCfg
from mppi.core.dynamics import SimpleCarDynamicsCfg
from mppi.core.sampling import DeltaSamplingCfg
from mppi.core.maps import BEVMapCfg
from mppi.core.vis import RolloutVisConfig

@dataclass
class MPPICfg:

    seed: int = None
    '''Seed for random number generator'''

    debug: bool = None
    '''Debug flag'''

    temperature: float = None
    '''Temperature of MPPI optimization step'''

    opt_iters: int = None
    '''Number of optimization iterations per action'''

    u_per_command: int = None
    '''Number of control commands per action'''

    cost_cfg : CostBaseCfg = None
    '''Rollout evaluator configuration'''

    dynamics_cfg: SimpleCarDynamicsCfg = None
    '''Dynamics model configuration'''

    sampling_cfg: DeltaSamplingCfg = None
    '''MPPI Sampling procedure configuration'''

    map_cfg: BEVMapCfg = None
    '''Map configuration'''

    vis_cfg: RolloutVisConfig = None
    '''Visualization configuration. None disables visualization'''