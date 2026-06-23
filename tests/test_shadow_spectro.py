from qiskit.circuit.random import random_circuit, random_clifford_circuit
from pai_shadow.hamil import Hamiltonian
from tqdm import tqdm
from pai_shadow.shadow_spectro.shadow_spectro import ShadowSpectro
from pai_shadow.shadow_spectro.spectroscopy import Spectroscopy
from pai_shadow.shadow_spectro.classical_shadow import ClassicalShadow
from pai_shadow.tools_box.quantum_tools import *
from pai_shadow.tools_box.plots_functions import plot_matrix, plot_spectre
from pai_shadow.tools_box.data_file_functions import save_parameters, save_to_pickle_file
import numpy as np
import sys
import time
import os

folder = os.path.dirname(os.path.dirname(
    os.path.abspath(os.path.dirname(sys.argv[0]))))
sys.path.append(folder)


def test_arg_type():
    Nt = 100
    dt = 0.1
    nq = 4
    k = 3
    shadow_size = 35
    shadow = ClassicalShadow()
    spectro = Spectroscopy(Nt, dt)
    shadow_spectro = ShadowSpectro(shadow, spectro, nq, k, shadow_size)

    assert shadow_spectro.No == 174, " No should be equal to 174 with nq = 4, k=3"
    assert isinstance(shadow_spectro.q_Pauli,
                      list), "shadow_spectro.q_Pauli should be a list"
    for obs in shadow_spectro.q_Pauli:
        assert isinstance(
            obs, tuple), "shadow_spectro.q_Pauli[k] should be a tuple"
        assert len(
            obs) == 4, "len(shadow_spectro.q_Pauli[k]) should be equal to the number of qubits"
        for i in range(4):
            assert isinstance(
                obs[i], str), "shadow_spectro.q_Pauli[k][i] should be a str"
    qc = random_circuit(4, 4)
    qc_test = qc.copy()

    snapshots_Clifford, snapshots_bits_string = shadow_spectro.get_snapshots_classical_shadow(
        qc)
    assert qc_test == qc, "Quantum circuit should not be modified"
    assert isinstance(snapshots_Clifford,
                      list), "snapshots_Clifford should be a list"
    assert len(
        snapshots_Clifford) == 35, "len(snapshots_Clifford) should be equal to shadow sizes"
    for cliff in range(snapshots_Clifford):
        assert isinstance(
            cliff, list), "snapshots_Clifford[k] should be a list"
        assert len(
            cliff) == 4, "len(snapshots_Clifford[k]) should be equal to number of qubits"
        for i in range(4):
            assert len(
                cliff[i]) == 3, "len(snapshots_Clifford[k][i]) should be equal to 3 "

    assert isinstance(snapshots_bits_string,
                      list), "snapshots_bits_string should be a list"
    assert len(
        snapshots_bits_string) == 35, "len(snapshots_bits_string) should be equal to shadow sizes"
    for bit_str in range(snapshots_bits_string):
        assert isinstance(
            bit_str, str), "snapshots_bits_string[k] should be a str"
        assert len(
            bit_str) == 4, "len(snapshots_bits_string[k]) should be equal to number of qubits"

    fk = shadow_spectro.expectation_value_q_Pauli(
        snapshots_Clifford, snapshots_bits_string, density_matrix=True)
    assert isinstance(fk, np.ndarray)
    assert len(fk) == 174
    assert isinstance(fk[0], float)


def test_expectation_value_q_Pauli():
    Nt = 100
    dt = 0.1
    nq = 4
    k = 4
    shadow_size = 750
    shadow = ClassicalShadow()
    spectro = Spectroscopy(Nt, dt)
    shadow_spectro = ShadowSpectro(shadow, spectro, nq, k, shadow_size)
    qc = random_clifford_circuit(4, 4)
    snapshots_Clifford, snapshots_bits_string = shadow_spectro.get_snapshots_classical_shadow(
        qc)
    fk = shadow_spectro.expectation_value_q_Pauli(
        snapshots_Clifford, snapshots_bits_string, density_matrix=True)
    error = []
    for n, obs in enumerate(shadow_spectro.q_Pauli):
        ideal = get_expectation_value(qc, "".join(obs))
        error.append(np.abs(fk[n] - ideal))

    assert np.mean(error) < 0.2

    fk = shadow_spectro.expectation_value_q_Pauli(
        snapshots_Clifford, snapshots_bits_string, density_matrix=False)
    error = []
    for n, obs in enumerate(shadow_spectro.q_Pauli):
        ideal = get_expectation_value(qc, "".join(obs))
        error.append(np.abs(fk[n] - ideal))
    assert np.mean(error) < 0.2
