
import math
import numpy as np

def linear_model(x, a, b):
    """Linear model: y = a*x + b."""
    return x * a + b

def square_model(x, b, a, c):
    """Generic power + linear term + bias: (x)**a + b*x + c."""
    return (x)**a + b*x + c

def calculate_theta2_classic(dist, size, width, length, alpha_deg):
    """Classic theta^2 estimator (degrees^2).

    Parameters
    ----------
    dist : float
        Distance from image CoG to (test) direction, deg.
    size : float
        Image size (arb. units).
    width, length : float
        Hillas ellipse params (deg).
    alpha_deg : float
        Angle between ellipse main axis and vector to (test) direction, deg.

    Returns
    -------
    float | None
        theta^2 estimate or None if inputs are invalid.
    """
    if any(v is None for v in [dist, size, width, length, alpha_deg]):
        return None
    if np.isnan(dist) or np.isnan(alpha_deg):
        return None

    # Ksi selection (legacy heuristic). Kept for backward compatibility.
    if dist < 0.9045:
        ksi = 1.3
    elif 0.9045 < dist < 1.206:
        ksi = 1.6
    elif dist > 1.206:
        if size < 700.0:
            ksi = 2.0
        elif 700.0 < size < 1000.0:
            ksi = 1.9
        elif 1000.0 < size < 1250.0:
            ksi = 1.75
        elif 1250.0 < size < 1500.0:
            ksi = 1.5
        else:
            ksi = 1.3
    elif dist > 1.3:
        return None
    else:
        return None

    disp = ksi * (1.0 - width / max(length, 1e-9))
    theta2 = disp ** 2 + dist ** 2 - 2.0 * disp * dist * math.cos(alpha_deg / 57.3)
    return theta2
