# Written by: Hugo PAGES 
# Date: 2024-01-05

# Standard library imports
import os
import gc
import time
import datetime
import pickle
import multiprocessing as mp
from dataclasses import dataclass

# Third-party imports
import numpy as np
from tqdm import tqdm
from qiskit import QuantumCircuit

# Local application imports
from pai_shadow.te_pai.te_pai import TE_PAI
from pai_shadow.shadow_spectro.classical_shadow import ClassicalShadow
from pai_shadow.shadow_spectro.spectroscopy import Spectroscopy
from pai_shadow.shadow_spectro.shadow_spectro import ShadowSpectro
from pai_shadow.tools_box.quantum_tools import get_expectation_value, get_depth_from_qasm

# Commented optional/debug tools
# from Monitoring import PerformanceMonitor
# import psutil
# import objgraph
# import tracemalloc


@dataclass
class TE_PAI_Shadow_Spectroscopy:
    """
    TE-PAI Shadow Spectroscopy class for analyzing quantum systems via spectral 
    decomposition using Time-Evolution Probabilistic Angle Interpolation (TE-PAI) 
    and classical shadows.

    This class performs shadow spectroscopy through time-evolved quantum circuits 
    sampled using TE-PAI and analyzed with classical shadows to extract spectral 
    properties of a quantum system.
    """
   
    def __init__(self, hamil, delta: float, dt: float, Nt, shadow_size: int,
                 trotter_steps:float=1.e-4, PAI_error:float=0.1, N_trotter_max : int =8000,M_sample_max: int =1000,
                 init_state: QuantumCircuit = None, K: int = 3, noise_coef:list[float,float] =None, num_qubits: int = None):
        """Class constructor of TE_PAI_Shadow_Spectroscopy.

        Parameters
        ----------
        hamil : Hamiltonian
            The Hamiltonian of the quantum system to be analyzed.
        delta : float
            Angular precision parameter for the PAI method.
        dt : float
            Time interval between steps for the spectroscopy.
        Nt : int
            Number of time steps in the spectroscopy process.
        shadow_size : int
            Number of snapshots (measurements) per circuit for classical shadows.
        trotter_steps : float, optional
            Trotterization step size for time evolution. Default is 1.e-4.
        PAI_error : float, optional
            Error threshold for the PAI method. Default is 0.1.
        N_trotter_max : int, optional
            Maximum number of Trotter steps per circuit. Default is 8000.
        M_sample_max : int, optional
            Maximum number of TE-PAI sampled circuits per time step. Default is 1000.
        init_state : QuantumCircuit, optional
            Initial quantum state to apply before time evolution. Default is None.
        K : int, optional
            Specifies k-Pauli observables used for shadow estimation. Default is 3.
        noise_coef : list[float,float], optional
            Probability p1, p2 for noise model. See classical shadow 
        """
        self.delta = delta
        self.init_state = init_state
        self.dt = dt
        self.Nt = Nt
        self.shadow_size = shadow_size
        self.hamil = hamil
        if num_qubits is not None:
            self.nq = num_qubits
        else:
            self.nq = hamil.num_qubits
        self.classical_shadow = ClassicalShadow(noise_error=noise_coef)
        self.spectro = Spectroscopy(Nt, dt)
        self.Shadow_Spectro = ShadowSpectro(
            self.classical_shadow, self.spectro, self.nq, K, self.shadow_size
        )
        self.No = self.Shadow_Spectro.No
        self.T = np.linspace(0, self.Nt * self.dt, self.Nt)
        self.density_matrix = False
        self.num_processes = min(30, int(mp.cpu_count() * 0.20))
        self.trotter_steps = trotter_steps
        self.PAI_error = PAI_error
        self.N_trotter_max=N_trotter_max
        self.M_sample_max=M_sample_max

        
        
        
    def loop_snapshots_circuit_TE_PAI_density_matrix(self,ms):
        """
        Compute the weighted density matrix snapshot for a given TE-PAI circuit index.
        (Used for multiprocessing, if density_matrix=True)

        Parameters
        ----------
        ms : int
            Index of the TE-PAI circuit sample.

        Returns
        -------
        np.ndarray
            The weighted density matrix corresponding to the circuit sample.
        """
        Rho = self.Shadow_Spectro.classical_shadow(self.C[ms], density_matrix=True)*self.GAMMA[ms]
        return Rho



    def loop_snapshots_circuit_TE_PAI_expectation_value(self, ms):
        """
        Compute the expectation values from classical shadows for a given TE-PAI circuit index.
        (Used for multiprocessing, if density_matrix=False)

        Parameters
        ----------
        ms : int
            Index of the TE-PAI circuit sample.

        Returns
        -------
        np.ndarray
            The weighted expectation values for the snapshot.
        """
        snapshots_shadow = self.Shadow_Spectro.classical_shadow(
            self.C[ms]
        )
        snapshot_expectation_values = (
            self.Shadow_Spectro.expectation_value_q_pauli(snapshots_shadow)
            * self.GAMMA[ms]
        )
        return snapshot_expectation_values

    def loop_time_evolve_te_pai_shadow(self):
        """
        Perform parallelized time evolution over all TE-PAI circuit samples and compute 
        the expectation value of all the Q-Pauli observable using either density matrix-based or shadow-based expectation values.

        Returns
        -------
        np.ndarray
            Averaged observable values for the current time step.
        """

        if self.density_matrix:
            Rho = np.zeros((2**self.nq, 2**self.nq), dtype="complex128")
            fkt = np.zeros(self.No)
            with mp.Pool(self.num_processes) as pool:
                Rho = np.sum(
                    pool.map(
                        self.loop_snapshots_circuit_TE_PAI_density_matrix,
                        range(self.TE_PAI_sample),
                        chunksize=self.chunk_size,
                    ),
                    axis=0,
                )
            fkt = self.Shadow_Spectro.expectation_value_q_pauli(
                Rho / self.TE_PAI_sample
            )
            return fkt
        else:
            expectation_values = []
            with mp.Pool(self.num_processes) as pool:
                expectation_values += pool.map(
                    self.loop_snapshots_circuit_TE_PAI_expectation_value,
                    self.ms_array,
                    chunksize=self.chunk_size,
                )

            fkt = np.sum(expectation_values, axis=0) / self.TE_PAI_sample
        return fkt


    def calculate_Data_matrix_te_pai_shadow_spectro(self, density_matrix: bool = False, save_D:int =None, serialize=False) -> np.ndarray:
        """
        Generate the full data matrix for TE-PAI shadow spectroscopy by calculating the expactation value of all Q-Pauli observables
        system through all time steps.

        Parameters
        ----------
        density_matrix : bool, optional
            If True, computes full density matrices instead of classical shadows. 
            Note: introduces exponential memory complexity. Default is False.
        save_D : int, optional
            If specified, saves the data matrix every `save_D` time steps. Default is None.
        serialize : bool, optional
            If True, circuits are serialized (QASM-based) to speed up processing. 
            Default is False.

        Returns
        -------
        np.ndarray
            The full data matrix `D`, shape (Nt, No), where `Nt` is the number of time 
            steps and `No` is the number of observables.
        """
        self.density_matrix = density_matrix
        D = []
        if isinstance(save_D, int):
            folder_name = r"{0}{1}".format(
                "data", datetime.datetime.now().strftime("%Y_%m_%d/")
            )
            if not os.path.exists(folder_name):
                folder_name = os.path.join(folder_name, f"Acquisition_1")
                os.makedirs(folder_name)
            else:
                n = len(os.listdir(folder_name))
                folder_name = os.path.join(folder_name, f"Acquisition_{n+1}")
                os.makedirs(folder_name)
        self.depth = []
        for n, t in tqdm(enumerate(self.T), desc="Computing time evolution"):
            trotter = TE_PAI(self.hamil, self.nq, self.delta, t,
                                trotter_steps=self.trotter_steps, PAI_error=self.PAI_error,N_trotter_max=self.N_trotter_max, 
                                init_state=self.init_state, serialize=serialize, M_sample_max=self.M_sample_max)
            trotter.gen_te_pai_circuits()
            self.depth.append(trotter.get_average_depth())
            self.C, self.GAMMA = trotter.TE_PAI_Circuits, trotter.GAMMA
            self.TE_PAI_sample=len(self.C)
            self.ms_array = [i for i in range(self.TE_PAI_sample)]
            self.chunk_size = max(1, self.TE_PAI_sample // (self.num_processes*5))
            D.append(self.loop_time_evolve_te_pai_shadow())
            if isinstance(save_D, int):
                if (n % save_D == 0 or (n + 1) == self.Nt) and n != 0:
                    my_dict = dict()
                    my_dict["D"] = D

                    with open(
                        os.path.join(folder_name, "Matrix_D") + str(n) + ".pickle", "wb"
                    ) as file:
                        pickle.dump(my_dict, file)
        print("Average depth of the TE_PAI circuits:", np.mean(self.depth))
        return np.array(D)

    def main_te_pai_shadow_spectro(
        self,
        Ljung: bool = True,
        density_matrix: bool = False,
        save_D: int = None,
        serialize=False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Main function to perform TE-PAI shadow spectroscopy.

        This method executes the full time evolution, computes the data matrix, and 
        performs spectral analysis (optionally applying the Ljung-Box test).

        Parameters
        ----------
        Ljung : bool, optional
            Whether to apply the Ljung-Box test to reduce noise in the spectrum. Default is True.
        density_matrix : bool, optional
            Whether to compute density matrices for each snapshot (introduces exponential complexity). 
            Default is False.
        save_D : int, optional
            If set, the data matrix is saved every `save_D` steps. Default is None.
        serialize : bool, optional
            If True, uses serialized QASM circuits for simulation to improve performance. 
            Default is False.

        Returns
        -------
        tuple of np.ndarray
            - Intensity: The spectral intensity at each frequency.
            - Frequencies: Corresponding frequency values.
        """
        self.density_matrix = density_matrix
        self.Ljung = Ljung
        self.depth=[]
        D = self.calculate_Data_matrix_te_pai_shadow_spectro(
            self.density_matrix,
            save_D=save_D,
            serialize=serialize,
        )
        frequencies, solution = self.spectro.Spectroscopy(D, Ljung)
        return frequencies,solution

   
