from dataclasses import dataclass

from .car_dynamics import SimpleCarDynamics, SimpleCarDynamicsNoAction

@dataclass
class SimpleCarDynamicsCfg:

    wheelbase: float
    '''wheelbase'''

    throttle_to_wheelspeed: float
    '''throttle to wheelspeed'''

    steering_max: float
    '''maximum steering angle'''

    dt: float
    '''time step'''

    feat_dim: int
    '''feature dimension. If None, no features are concatenated'''

    concatenate_feats: bool
    '''concatenate features to rollout states'''

    x_dim: int = 14
    '''agent state dimension'''

    class_type: type = SimpleCarDynamics

@dataclass
class SimpleCarDynamicsNoActionCfg(SimpleCarDynamicsCfg):
    '''
    Configuration class for SimpleCarDynamics
    '''
    x_dim: int = 12
    '''agent state dimension excludes action'''

    class_type: type = SimpleCarDynamicsNoAction
    '''class type for the dynamics model'''
