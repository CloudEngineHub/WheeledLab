import torch
import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.assets import AssetBaseCfg, ArticulationCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils import configclass
from isaaclab.managers import (
    RewardTermCfg as RewTerm,
    TerminationTermCfg as DoneTerm,
    EventTermCfg as EventTerm,
    SceneEntityCfg
)

from wheeledlab_assets import MUSHR_SUS_CFG
from wheeledlab_tasks.drifting.mushr_drift_env_cfg import (
    DriftTerrainImporterCfg, MushrDriftRLEnvCfg, DriftEventsCfg
)
from wheeledlab_tasks.common import Mushr4WDActionCfg
from .observations import MappingObsCfg, ExpertObsCfg, Figure8GailObsCfg


## SHARED PARAMS
GOAL_POS = [10.0, 10.0, 0.0]
##

@configclass
class SceneCfg(InteractiveSceneCfg):

    terrain = DriftTerrainImporterCfg()

    robot: ArticulationCfg = MUSHR_SUS_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    # spawn a red cone
    cfg_cone = AssetBaseCfg(
        prim_path="/World/goal",
        init_state=AssetBaseCfg.InitialStateCfg(pos=GOAL_POS),
        spawn=sim_utils.ConeCfg(
            radius=0.15,
            height=0.5,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        )
    )

@configclass
class ScannerSceneCfg(SceneCfg):

    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/mushr_nano/base_link",
        offset=RayCasterCfg.OffsetCfg(
            pos=(0.0, 0.0, 20.0),
        ),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(size=[2., 2.], resolution=2./19),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

## Events


def out_of_bounds(env, x_lims: tuple = (-1., 11.), y_lims: tuple = (-1., 11.)):
    pos = mdp.root_pos_w(env)
    in_x_bounds = torch.logical_and(pos[...,0] > x_lims[0], pos[...,0] < x_lims[1])
    in_y_bounds = torch.logical_and(pos[...,1] > y_lims[0], pos[...,1] < y_lims[1])
    return torch.logical_not(torch.logical_and(in_x_bounds, in_y_bounds))

@configclass
class NavTerminationsCfg:

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # out_of_bounds = DoneTerm(
    #     func=out_of_bounds,
    # )

## Rewards

def dist2goal(env, goal: list, offset: float=0.):
    goal = torch.tensor(goal, device=env.device)
    pos = mdp.root_pos_w(env)
    return torch.linalg.norm(goal - pos, dim=-1) + offset

def progress2goal(env, goal: list):
    pos = mdp.root_pos_w(env)
    vel = mdp.root_lin_vel_w(env)
    goal_pos = torch.tensor(goal, device=env.device)
    # goal_pos = goal_pos[:, :2] # we only need the x, y coordinates

    goal_vector = goal_pos - pos
    proj_scal = torch.sum(vel * goal_vector, dim=-1) / torch.norm(goal_vector, dim=-1)
    return proj_scal

@configclass
class NavRewardsCfg:

    goal = RewTerm(
        func=dist2goal,
        weight=-1.0,
        params={
            "goal": GOAL_POS,
            "offset": -math.hypot(*GOAL_POS),
        }
    )

    # goal = RewTerm(
    #     func=progress2goal,
    #     weight=1.0,
    #     params={
    #         "goal": GOAL_POS
    #     }
    # )

def reset_list_root_state_uniform(
    env,
    env_ids,
    pose_range: dict[str, list[tuple[float, float]]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    # bin the env_ids into number of ranges specified
    num_ranges = len(pose_range["x"])
    # reset the first half to the first pose range
    # and the second half to the second pose range
    for i in range(num_ranges):
        # get the env_ids for this half
        start = i * (len(env_ids) // num_ranges)
        end = (i + 1) * (len(env_ids) // num_ranges)
        if i == num_ranges - 1:
            end = len(env_ids)
        env_id = env_ids[start:end]
        # get the pose range for this half
        _pose_range = {k: v[i] for k, v in pose_range.items()}
        mdp.reset_root_state_uniform(
            env,
            env_id,
            pose_range=_pose_range,
            velocity_range=velocity_range,
            asset_cfg=asset_cfg,
        )

@configclass
class NavEventsCfg:
    # on startup

    reset_root_state = EventTerm(
        func=mdp.reset_root_state_uniform,
        params={
            "pose_range": {
                'x': (-1.2, 1.0),
                'y': (-1.6, 1.5),
                'yaw': (math.pi/2 - 0.1, math.pi/2 + 0.1),
            },
            "velocity_range": {
                'x': (0.0, 0.0),
                'y': (0.0, 0.0),
            },
        },
        mode="reset",
    )

    # reset_root_state = EventTerm(
    #     func=reset_list_root_state_uniform,
    #     params={
    #         "pose_range": {
    #             # 'x': [(-1.2,-1.1), (-.7, -.6)],
    #             # 'y': [(-1.6, -1.5), (1.4, 1.5)],
    #             'x': [
    #                 (-1.2,-1.1),
    #                 # (-.7, -.6), # Figure 8
    #                 (0.9, 1.0)
    #             ],
    #             'y': [
    #                 (-1.6, -1.5),
    #                 # (1.4, 1.5), # Figure 8
    #                 (-1.2, -1.1)
    #             ],
    #             'yaw': [
    #                 (math.pi/2 - 0.1, math.pi/2 + 0.1),
    #                 # (math.pi + math.pi/2 - 0.1, math.pi + math.pi/2 + 0.1), # Figure 8
    #                 (math.pi + math.pi/2 - 0.1, math.pi + math.pi/2 + 0.1),
    #             ],
    #         },
    #         "velocity_range": {
    #             'x': (0.0, 0.0),
    #             'y': (0.0, 0.0),
    #         },
    #     },
    #     mode="reset",
    # )


## Task configurations

@configclass
class NavEnvCfg(ManagerBasedRLEnvCfg):

    num_envs: int = 512
    env_spacing: float = 0.0

    observations = MISSING
    actions: Mushr4WDActionCfg = Mushr4WDActionCfg()

    rewards: NavRewardsCfg = NavRewardsCfg()
    events: NavEventsCfg = DriftEventsCfg()
    terminations: NavTerminationsCfg = NavTerminationsCfg()

    def __post_init__(self):
        self.viewer.eye = [6., -8., 8.0]
        self.viewer.lookat = [3.0, 3.0, 0.]

        self.sim.dt = 0.01  # 100 Hz
        self.decimation = 10  # 10 Hz
        self.actions.throttle_steer.scale = (3.0, 0.488)
        self.sim.render_interval = self.decimation * 2
        self.episode_length_s = 10

        self.scene = ScannerSceneCfg(num_envs=self.num_envs, env_spacing=self.env_spacing)

@configclass
class ExpertNavEnvCfg(NavEnvCfg):

    observations: ExpertObsCfg = ExpertObsCfg()
    
@configclass
class Figure8GAILEnvCfg(NavEnvCfg):
    observations: Figure8GailObsCfg = Figure8GailObsCfg()

@configclass
class MappingNavEnvCfg(NavEnvCfg):

    observations: MappingObsCfg = MappingObsCfg()

#######################
# OOD EVALUATION TASK #
#######################

@configclass
class OODNavEventsCfg:
    reset_root_state = EventTerm(
        func=mdp.reset_root_state_uniform,
        params={
            "pose_range": {
                "x": (-20.0, 20.0),
                "y": (-20.0, 20.0),
                "yaw": (-math.pi, math.pi),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
            }
        },
        mode="reset",
    )

@configclass
class OODMappingNavEnvCfg(MappingNavEnvCfg):
    events: OODNavEventsCfg = OODNavEventsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.viewer.eye = [0., 0., 40.0]
        self.viewer.lookat = [0.0, 0., 0.]

@configclass
class OODExpertNavEnvCfg(ExpertNavEnvCfg):
    events: OODNavEventsCfg = OODNavEventsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.viewer.eye = [0., 0., 45.0]
        self.viewer.lookat = [0.0, 1., 0.]