# Written by: Chusei Kiumi, updated by Hugo PAGES
# Date: 2024-01-05

# Third-party imports
from numba import jit
import numpy as np
from scipy.stats import binom


# def batch_sampling(probs, batch_size):
#     return mp.Pool(mp.cpu_count()).map(sample_from_prob, [probs] * batch_size)

def batch_sampling(probs, M_sample):
    index = []
    for _ in range(M_sample):
        index.append(sample_from_prob(probs))
    return index


@jit(nopython=True)
def custom_random_choice(prob):
    cdf = np.cumsum(prob)
    r = np.random.random()
    idx = np.searchsorted(cdf, r)
    return idx + 1

@jit(nopython=True)
def sample_from_prob(probs):
    res = []
    for i in range(probs.shape[0]):
        res2 = []
        for j in range(probs.shape[1]):
            val = custom_random_choice(probs[i][j])
            if val != 1:
                res2.append((j, val))
        res.append(res2)
    return res


def resample(res):
    s = np.concatenate([c * (2 * binom.rvs(1, p, size=100) - 1)
                       for (c, p) in res])
    choices = np.reshape(
        s[np.random.choice(len(s), 1000 * 10000)], (10000, 1000))
    return np.mean(choices, axis=1)
