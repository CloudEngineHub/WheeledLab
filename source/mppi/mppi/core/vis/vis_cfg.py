from dataclasses import dataclass
from typing import Type

from .rollout_vis import RolloutsVisualization

@dataclass
class RolloutVisConfig:

    class_type: Type[RolloutsVisualization] = RolloutsVisualization

    vis_rollouts: bool = None

    vis_n_envs: int = None

    vis_n_rollouts: int = None

    xlim: tuple = None

    ylim: tuple = None

    show_velocity: bool = None

    show_elevation: bool = None

    cost_range: tuple = None

    show_trajectory_trace: bool = None