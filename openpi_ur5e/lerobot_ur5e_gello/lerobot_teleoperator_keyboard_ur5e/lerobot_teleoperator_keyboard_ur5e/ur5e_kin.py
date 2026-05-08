"""UR5e forward kinematics and geometric Jacobian (pure Python, no robot connection).

Uses standard DH parameters from the UR5e product datasheet.
Provides a geometric Jacobian + damped-pseudoinverse for Cartesian → joint-space mapping.

The geometric Jacobian directly relates joint velocities to [linear_vel, angular_vel]
in the base frame, avoiding axis-angle singularities that plague numerical approaches
near ±180° rotations.
"""

import numpy as np

# UR5e standard DH parameters (from UR product documentation)
_D     = np.array([0.1625,  0.0,     0.0,     0.1333,  0.0997,  0.0996])
_A     = np.array([0.0,    -0.425,  -0.3922,  0.0,     0.0,     0.0   ])
_ALPHA = np.array([np.pi/2, 0.0,     0.0,     np.pi/2, -np.pi/2, 0.0  ])


def _dh_matrix(theta: float, d: float, a: float, alpha: float) -> np.ndarray:
    """Single standard-DH homogeneous transformation."""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0.0, sa,       ca,      d     ],
        [0.0, 0.0,      0.0,     1.0   ],
    ])


def forward_kinematics(q: np.ndarray) -> np.ndarray:
    """6 joint angles (rad) → 4×4 TCP pose in base frame."""
    T = np.eye(4)
    for i in range(6):
        T = T @ _dh_matrix(q[i], _D[i], _A[i], _ALPHA[i])
    return T


def geometric_jacobian(q: np.ndarray) -> np.ndarray:
    """Compute the 6×6 geometric Jacobian at configuration q.
    
    The geometric Jacobian maps joint velocities to spatial velocities:
        [v_x, v_y, v_z, omega_x, omega_y, omega_z]^T = J(q) * dq
    
    where v is the linear velocity and omega is the angular velocity of the
    end-effector, both expressed in the base frame.
    
    This avoids singularities in the axis-angle representation that occur
    near ±180° rotations.
    """
    J = np.zeros((6, 6))
    
    # Compute all intermediate transforms T_0_i for i = 0..6
    T = [np.eye(4)]  # T_0_0 = identity (base frame)
    for i in range(6):
        T.append(T[-1] @ _dh_matrix(q[i], _D[i], _A[i], _ALPHA[i]))
    
    # End-effector position in base frame
    p_ee = T[6][:3, 3]
    
    # For each joint (all revolute for UR5e):
    for i in range(6):
        # z-axis of frame i (rotation axis of joint i+1 in base frame)
        z_i = T[i][:3, 2]
        # Position of joint i origin in base frame
        p_i = T[i][:3, 3]
        
        # Linear velocity contribution: z_i × (p_ee - p_i)
        J[:3, i] = np.cross(z_i, p_ee - p_i)
        # Angular velocity contribution: z_i
        J[3:, i] = z_i
    
    return J


def cartesian_to_joint_delta(
    q: np.ndarray, dx: np.ndarray, damping: float = 0.01
) -> np.ndarray:
    """Convert 6-D Cartesian delta [dx dy dz wx wy wz] → joint delta.
    
    Uses the geometric Jacobian with damped least-squares pseudo-inverse (DLS)
    for robust inversion even near singularities.
    
    The input dx is interpreted as:
        [dx, dy, dz] = desired linear displacement in base frame
        [wx, wy, wz] = desired angular displacement (rotation vector) in base frame
    
    Args:
        q: Current joint angles (6,)
        dx: Desired Cartesian delta [vx, vy, vz, wx, wy, wz] (6,)
        damping: DLS damping factor (higher = more stable but less accurate)
    
    Returns:
        dq: Joint angle deltas (6,)
    """
    J = geometric_jacobian(q)
    # Damped least-squares: dq = J^T (J J^T + λ²I)^{-1} dx
    JJT = J @ J.T
    J_pinv = J.T @ np.linalg.inv(JJT + damping**2 * np.eye(6))
    return J_pinv @ dx


# Keep old API for backward compatibility
def numerical_jacobian(q: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """6×6 numerical Jacobian (DEPRECATED - use geometric_jacobian instead).
    
    Warning: This uses axis-angle representation which is singular at ±180°.
    Kept for backward compatibility only.
    """
    from scipy.spatial.transform import Rotation
    
    def _pose_vector(T):
        return np.concatenate([T[:3, 3], Rotation.from_matrix(T[:3, :3]).as_rotvec()])
    
    p0 = _pose_vector(forward_kinematics(q))
    J = np.zeros((6, 6))
    for i in range(6):
        q_eps = q.copy()
        q_eps[i] += eps
        J[:, i] = (_pose_vector(forward_kinematics(q_eps)) - p0) / eps
    return J
