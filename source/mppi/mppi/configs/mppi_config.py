from typing import List
from dataclasses import dataclass, field

from mppi.core import (
    MinimalCostCfg,
    SimpleCarDynamicsNoActionCfg,
    DeltaSamplingCfg,
    BEVMapCfg,
    MPPICfg,
    RolloutVisConfig
)


@dataclass
class MinCostConfig(MinimalCostCfg):
    goal_w: float                   = 1.
    speed_w: float                  = 10.
    goal_pos: list = field(default_factory=lambda :[10., 10., 0.])
    target_speed: float             = 2.


@dataclass
class DynamicsConfig(SimpleCarDynamicsNoActionCfg):

    feat_dim: int               = None     # feature dimension
    concatenate_feats: bool            = False  # concatenate features to rollout states
    wheelbase: float                = 0.33   # wheelbase
    throttle_to_wheelspeed: float   = 3.0   # throttle to wheelspeed
    steering_max: float             = 0.488   # maximum steering angle
    dt: float                       = 0.05   # time step

@dataclass
class DynamicsNoActionConfig(SimpleCarDynamicsNoActionCfg):
    '''
    Configuration class for SimpleCarDynamics
    '''
    feat_dim: int               = None     # feature dimension
    concatenate_feats: bool            = False  # concatenate features to rollout states
    wheelbase: float                = 0.325   # wheelbase
    throttle_to_wheelspeed: float   = 3.0   # throttle to wheelspeed
    steering_max: float             = 0.4   # maximum steering angle
    dt: float                       = 0.1   # time step

@dataclass
class SamplingConfig(DeltaSamplingCfg):

    control_dim: int                = 2     # control dimension
    noise: List[int]                = field(default_factory=lambda: [1.0, 0.5]) # noise
    scaled_dt: float                = 0.1   # scaled dt
    max_delta: List[float]          = field(default_factory=lambda: [0.2, 0.2])  # max delta per step
    max_control: List[tuple]        = field(default_factory=lambda: [(0.0, 1.0), (-1.0, 1.0)])  # max control values
    num_rollouts: int               = 1024  # number of rollouts
    num_timesteps: int              = 20    # number of timesteps

@dataclass
class MapConfig(BEVMapCfg):
    map_length_px: int              = 20   # gym map length (pixels)
    map_res_m_px: float             = 3./19.   # gym map resolution (meters per pixel).
    map_res_hitl: float             = 0.25  # map resolution hitl
    feature_dim: int                = 4     # feature dimension


@dataclass
class VisConfig(RolloutVisConfig):
    vis_n_envs: int                 = 4     # number of environments to visualize
    vis_n_rollouts: int             = 10     # number of rollouts to visualize
    xlim: tuple                = (-1, 1)
    ylim: tuple                = (-1, 1)
    show_velocity: bool             = False
    show_elevation: bool            = False
    cost_range: tuple               = None  # cost range for visualization


@dataclass
class MPPIConfig(MPPICfg):

    seed: int                       = 0
    temperature: float              = 1.
    opt_iters: int                  = 1
    u_per_command: int              = 1
    debug: bool                     = False

    cost_cfg: MinCostConfig         = MinCostConfig()
    dynamics_cfg: DynamicsConfig     = DynamicsConfig()
    sampling_cfg: SamplingConfig     = SamplingConfig()
    map_cfg: MapConfig               = MapConfig()
    vis_cfg: VisConfig               = VisConfig()
