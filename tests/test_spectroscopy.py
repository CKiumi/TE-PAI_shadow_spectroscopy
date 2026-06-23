from pai_shadow.tools_box.plots_functions import *
from pai_shadow.tools_box.data_file_functions import *
from pai_shadow.shadow_spectro.spectroscopy import Spectroscopy
import warnings
import numpy as np
import sys
import time
import os
folder = os.path.dirname(os.path.dirname(
    os.path.abspath(os.path.dirname(sys.argv[0]))))
print(folder)
sys.path.append(folder)
warnings.filterwarnings('ignore')


def test_spectroscopy():
    np.random.seed(42)
    Nt = 1000
    dt = 0.01
    num_observables = 10
    t = np.linspace(0, Nt * dt, Nt)
    freqs = [1, 5, 8]
    omega = 2*np.pi*np.array(freqs)
    data_matrix = np.array([np.sin(2 * np.pi * f * t) for f in freqs]).T
    data_matrix += 0.2 * np.random.randn(Nt, len(freqs))
    random_signals = np.random.randn(Nt, num_observables - len(freqs))
    data_matrix = np.hstack((data_matrix, random_signals))
    # Initialize the Spectroscopy class
    spectroscopy = Spectroscopy(Nt=Nt, dt=dt, cutoff=4)
    # Run spectroscopy analysis
    solution, frequencies = spectroscopy.Spectroscopy(data_matrix)
    pos = np.argsort(np.abs(solution))[::-1]
    omega_spectro = np.array(
        [frequencies[pos[0]], frequencies[pos[1]], frequencies[pos[2]]])
    avg_error = np.mean(np.abs(np.sort(omega)-np.sort(omega_spectro)))
    assert avg_error < 0.1
