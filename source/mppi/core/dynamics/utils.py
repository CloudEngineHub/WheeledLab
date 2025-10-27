import torch
import math
import torch.nn.functional as F

def create_rollout_tensor(
    x0: torch.Tensor,
    controls: torch.Tensor,
    device: str = None,
    dtype: torch.dtype = None
):
    '''
    args:
        x0: initial state of shape (B, x_dim)
        controls: controls of shape (B, ..., T, a_dim)
            T is the number of time steps, a_dim is the action dimension
    returns:
        xs: rollout tensor of shape (B, ..., T+1, x_dim)
            where T+1 is the number of time steps + 1 for the initial state
            
    This method only rolls out x0. It does not apply the controls to the state.
    '''

    # Determine device
    _device = controls.device if device is None else device
    _dtype = controls.dtype if dtype is None else dtype

    # reshape x0 to (num_envs, ..., x_dim)
    _B = x0.shape[0]
    _batch_shape = list(controls.shape[1:-1])
    _batch_shape[-1] = _batch_shape[-1] + 1  # Horizon is one more than the number of controls

    x0 = x0.to(device=_device, dtype=_dtype)
    xs = x0.unsqueeze(-2).expand(
        _B, math.prod(_batch_shape), x0.shape[-1]
    ).view(_B, *_batch_shape, x0.shape[-1]).clone() # [num_envs, ..., x_dim]

    return xs

def euler_to_quat(rpy: torch.Tensor) -> torch.Tensor:
    """
    Convert roll-pitch-yaw to quaternion (w, x, y, z).
    Assumes right-handed coordinate system with:
    - Roll about X axis
    - Pitch about Y axis
    - Yaw about Z axis
    
    Args:
        rpy: Tensor of shape [..., 3], where [..., 0] = roll,
             [..., 1] = pitch, [..., 2] = yaw
    
    Returns:
        Tensor of shape [..., 4] representing quaternions in (w, x, y, z) format.
    """
    roll, pitch, yaw = rpy.unbind(-1)

    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return torch.stack((w, x, y, z), dim=-1)

def quat_to_euler(quat: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion (w, x, y, z) to roll-pitch-yaw.
    Returns angles in radians.

    Args:
        quat: Tensor of shape [..., 4] representing (w, x, y, z)

    Returns:
        Tensor of shape [..., 3] representing (roll, pitch, yaw)
    """
    w, x, y, z = quat.unbind(-1)

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    _eps = 1e-6  # Small epsilon to avoid numerical issues with asin
    sinp = 2 * (w * y - z * x)
    pitch = torch.where(
        sinp.abs() >= 1,
        torch.sign(sinp) * (torch.pi / 2.0),
        # torch.asin(sinp)
        torch.asin(
            torch.clamp(sinp, -1.0 + _eps, 1.0 - _eps)
        )
    )

    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return torch.stack((
        roll % (2 * torch.pi),
        pitch % (2 * torch.pi),
        yaw % (2 * torch.pi)),
        dim = -1
    )
    # return torch.stack((roll, pitch, yaw), dim=-1)

def quat_normalize(q):
    """Normalize quaternion to unit norm."""
    return q / q.norm(dim=-1, keepdim=True)

def quat_multiply(q1, q2):
    """
    Quaternion product: q = q1 * q2
    Assumes wxyz format: [w, x, y, z]
    """
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return torch.stack((w, x, y, z), dim=-1)

def quaternion_distance(q1, q2):
    """Quaternion distance: 1 - |<q1, q2>|"""
    q1 = quat_normalize(q1)
    q2 = quat_normalize(q2)
    return 1.0 - torch.abs(torch.sum(q1 * q2, dim=-1))
