# Written by: Chusei Kiumi

# Third-party imports
import numpy as np
from numba import jit


@jit(nopython=True)
def prob_list(angles, Δ):
    probs = [abc(θ, (1 if θ == 0 else np.sign(θ)) * Δ) for θ in angles]
    return [list(np.abs(probs) / np.sum(np.abs(probs))) for probs in probs]


@jit(nopython=True)
def abc(theta, delta):
    a = (1 + np.cos(theta) - (np.cos(delta) + 1) /
         np.sin(delta) * np.sin(theta)) / 2
    b = np.sin(theta) / np.sin(delta)
    c = (1 - np.cos(theta) - np.sin(theta) * np.tan(delta / 2)) / 2
    return np.array([a, b, c])


@jit(nopython=True)
def gamma(angles, Δ):
    gam = [
        (np.cos((np.sign(θ) * Δ / 2) - θ)) / np.cos(np.sign(θ) * Δ / 2) for θ in angles
    ]
    return np.prod(np.array(gam))
