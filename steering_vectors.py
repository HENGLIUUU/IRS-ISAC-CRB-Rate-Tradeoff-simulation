"""Centered-ULA steering vectors and angular derivatives."""

import numpy as np


def steering_vector(N, theta, spacing=0.5):
    """
    Generate ULA steering vector with midpoint reference.

    [Eq.(5)]: a(theta) with centered ULA
    The midpoint of the array is the phase reference point.

    Args:
        N: Number of antennas
        theta: Direction angle (rad)
        spacing: Antenna spacing in wavelengths (default 0.5 = lambda/2)

    Returns:
        a: Steering vector (N, 1), complex-valued

    (N - 1 - 2*n) 这个结构让相位关于阵列中点对称：
    - n=0(第 1 根天线）：相位 = -pi*(N-1)*sinθ
    - n=N-1(最后 1 根天线）：相位 = pi*(N-1)*sinθ
    - 中间的某根天线相位 = 0(阵列中点,参考点)
    spacing变量包含了da/λ的值,当spacing=0.5 时，相邻天线相位差为 pi*sinθ,满足奈奎斯特采样定理
    """
    n = np.arange(N)
    # Centered ULA: phases are symmetric around the midpoint
    phase = -np.pi * (N - 1 - 2 * n) * spacing * np.sin(theta)
    return np.exp(1j * phase).reshape(-1, 1)


def steering_vector_derivative(N, theta, spacing=0.5):
    """
    Derivative of steering vector w.r.t. theta.

    [Eq.(64), Appendix A]:
        b_dot = (j * pi * spacing * cos(theta)) * D * b(theta)
    where D = diag(-(N-1), -(N-3), ..., (N-1))

    Args:
        N: Number of antennas
        theta: Direction angle (rad)
        spacing: Antenna spacing in wavelengths

    Returns:
        b_dot: Derivative vector (N, 1), complex-valued
    """
    b = steering_vector(N, theta, spacing)
    n = np.arange(N)
    D = np.diag(-(N - 1) + 2 * n)
    b_dot = 1j * np.pi * spacing * np.cos(theta) * D @ b
    return b_dot
