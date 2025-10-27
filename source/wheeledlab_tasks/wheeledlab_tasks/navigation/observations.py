import torch

import isaaclab.envs.mdp as mdp
from isaaclab.envs import ManagerBasedEnv
from isaaclab.utils import configclass
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.noise import (
    AdditiveUniformNoiseCfg as Unoise,
    AdditiveGaussianNoiseCfg as Gnoise,
)

from wheeledlab.envs.mdp import root_euler_xyz
import isaaclab.utils.math as math_utils

def map_from_height_scan(env, map_length_px:int, sensor_cfg:SceneEntityCfg):
    height_scan = mdp.height_scan(env, sensor_cfg)
    height_map = height_scan.reshape(-1, map_length_px, map_length_px)
    return height_map

def flat_normal_map(env, map_length_px:int, sensor_cfg:SceneEntityCfg):
    ''' Returns a flat normal map by querying the shape from the height scan '''
    height_scan = map_from_height_scan(env, map_length_px, sensor_cfg)
    # height_scan = square_elev_from_height_scan(env, sensor_cfg) # for RayCaster (not camera)
    normal_map = torch.zeros((*height_scan.shape, 3), device=height_scan.device)
    normal_map[..., -1] = 1.
    return normal_map

def terrain_feature_map(env: ManagerBasedEnv, map_length_px:int, sensor_cfg:SceneEntityCfg):
    ''' Returns a terrain feature map where the features are a concatenation of the height map and the normal map '''
    height_map = map_from_height_scan(env, map_length_px, sensor_cfg)[..., None]
    normal_map = flat_normal_map(env, map_length_px, sensor_cfg)
    terrain_map = torch.cat((height_map, normal_map), dim=-1)

    return terrain_map

def gt_features(env: ManagerBasedEnv, map_length_px:int, sensor_cfg:SceneEntityCfg):
    ''' Returns the ground truth features. Using middle of the map as the ground truth '''
    terrain_map = terrain_feature_map(env, map_length_px, sensor_cfg)
    feats = terrain_map[:, map_length_px//2, map_length_px//2]
    return feats

@configclass
class MappingCfg(ObsGroup):
    """Observations for mapping group."""

    feature_map = ObsTerm(
        func=terrain_feature_map,
        params={
                "sensor_cfg": SceneEntityCfg("height_scanner"),
                "map_length_px": 20,
            },
        noise=Unoise(n_min=-0.1, n_max=0.1),
        clip=(-100., 100.),
    )

    def __post_init__(self):
        self.concatenate_terms = True # Collapse obs dict
        self.enable_corruption = False

@configclass
class Proprioception(ObsGroup):

    root_pos_w_term = ObsTerm( # meters
        func=mdp.root_pos_w,
        noise=Gnoise(mean=0., std=0.1),
    )

    root_euler_xyz_term = ObsTerm( # radians
        func=root_euler_xyz,
        noise=Gnoise(mean=0., std=0.1),
    )

    base_lin_vel_term = ObsTerm( # m/s
        func=mdp.base_lin_vel,
        noise=Gnoise(mean=0., std=0.5),
    )

    base_ang_vel_term = ObsTerm( # rad/s
        func=mdp.base_ang_vel,
        noise=Gnoise(std=0.4),
    )

    def __post_init__(self):
        self.concatenate_terms = True # Collapse obs dict
        self.enable_corruption = False

@configclass
class MappingObsCfg:
    """Default observation configuration (no sensors; no corruption)"""

    proprioception: Proprioception = Proprioception() # Proprioceptive state

    exteroception: MappingCfg = MappingCfg() # Exteroceptive state

def root_euler_yaw(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Returns the yaw of the root euler angles."""
    xyz_tuple = math_utils.euler_xyz_from_quat(mdp.root_quat_w(env, asset_cfg))
    return xyz_tuple[2].unsqueeze(-1)

@configclass
class BareProprioceptionCfg(ObsGroup):
    """Bare proprioception configuration"""
    root_pos_w_term = ObsTerm( # meters
        func=mdp.root_pos_w,
        noise=Gnoise(mean=0., std=0.1),
    )


    base_lin_vel_term = ObsTerm( # m/s
        func=mdp.base_lin_vel,
        noise=Gnoise(mean=0., std=0.5),
    )

    root_euler_xyz_term = ObsTerm( # radians
        func=root_euler_yaw,
        noise=Gnoise(mean=0., std=0.1),
    )

@configclass
class Figure8GailObsCfg:
    """Observation configuration for Figure 8 GAIL"""

    policy: BareProprioceptionCfg = BareProprioceptionCfg()

@configclass
class ExpertObsCfg:

    policy: Proprioception = Proprioception()

    def __post_init__(self):
        setattr(self.policy, "gt_features",
            ObsTerm(
                func=gt_features,
                params={
                    "sensor_cfg": SceneEntityCfg("height_scanner"),
                    "map_length_px": 20,
                },
                noise=Unoise(n_min=-0.1, n_max=0.1),
                clip=(-100., 100.),
            )
        )