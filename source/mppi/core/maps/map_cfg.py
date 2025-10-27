from dataclasses import MISSING, dataclass

from .bev_map import BEVMap

@dataclass
class BEVMapCfg:

    map_length_px: int
    ''' map length (pixels) '''

    map_res_m_px: float
    ''' map resolution (meters per pixel). '''

    feature_dim: int
    ''' feature dimension '''

    class_type: type = BEVMap
