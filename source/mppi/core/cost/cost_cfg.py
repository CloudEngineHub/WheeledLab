import torch.nn as nn
from dataclasses import dataclass
from typing import Callable, Any, List, Tuple, Type, Optional

from .minimal_cost import MinimalCost
from .nn_cost import NNCost

@dataclass
class CostBaseCfg:

    class_type: Type = None
    '''class type of cost function'''

@dataclass
class NNCostCfg(CostBaseCfg):

    model_factory: Callable[[Any], nn.Sequential] = None
    '''model factory to create model for costing'''

    model_kwargs: dict = None
    '''model arguments if model_factory is used'''

    clip_costs: Tuple[float, float] = None
    '''clip costs to this range'''

    class_type: Type[NNCost] = NNCost

@dataclass
class MinimalCostCfg(CostBaseCfg):

    goal_w: float = None
    '''weight on terminal goal cost'''

    goal_pos: list = None
    '''goal position'''

    speed_w: float = None
    '''weight with which target speed will be tracked'''

    target_speed: float = None
    '''target speed in m/s'''

    class_type: Type[MinimalCost] = MinimalCost
