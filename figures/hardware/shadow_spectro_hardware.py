# Written by: Hugo PAGES
# Date: 2024-01-05

# Standard library imports
from itertools import chain
import time
from typing import Union
import os, sys 

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Third-party imports
import numpy as np
from tqdm import tqdm
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import UnitaryGate
from qiskit_ibm_runtime import SamplerV2 as Sampler, Batch, Options
import multiprocessing as mp 
from pai_shadow.tools_box.data_file_functions import get_data_file, save_to_file
import secrets

class ShadowSpectro_Hardware():
    """
    Class for performing Shadow Spectroscopy on real quantum hardware.

    This class interfaces a shadow spectroscopy routine with a quantum backend
    (either real or simulated) to generate, execute, and analyze time-evolved
    quantum circuits for spectral analysis of a given Hamiltonian.
    
    Attributes:
        hardware: An object managing backend access and job submission (custom `Hardware` class).
        shadow_spectro: An instance of a shadow spectroscopy class.
        shadow: The classical shadow module used for circuit modifications and reconstruction.
        shadow_size: Number of classical shadows used at each time step.
        nq: Number of qubits in the system.
    """
    def __init__(self, hardware, shadow_spectro, post_process_mode: bool=False):
        """
        Initialize the ShadowSpectro_Hardware class.
        
        Args:
            hardware: A hardware interface class responsible for transpilation and job management.
            shadow_spectro: An instance of a shadow spectroscopy class containing parameters and methods.
        """
        
        if post_process_mode:
            self.hardware = hardware
            self.backend = hardware.backend
            
        else:
            self.hardware = hardware
            self.backend = hardware.backend if hasattr(hardware, 'backend') else None
            self.X = np.array([[0, 1],  [1, 0]])
            self.Y = np.array([[0, -1j], [1j, 0]])
            self.Z = np.array([[1, 0],  [0, -1]])
            self.I = np.array([[1, 0],  [0, 1]])
            self.S = np.array([[1, 0],  [0, 1j]])
            self.H = np.array([[1/np.sqrt(2), 1/np.sqrt(2)],
                            [1/np.sqrt(2), -1/np.sqrt(2)]])
            self.V = self.H@self.S@self.H@self.S
            self.W = self.V@self.V
            self.gate_set = {"X": self.X, "Y": self.Y, "Z": self.Z, "I": self.I, "S": self.S,
                            "H": self.H, "V": self.V, "W": self.W}
            self.Clifford_Gate_set = [
                "III", "XII", "YII", "ZII", "VII", "VXI", "VYI", "VZI",
                "WXI", "WYI", "WZI", "HXI", "HYI", "HZI", "HII",
                "HVI", "HVX", "HVY", "HVZ", "HWI", "HWX",
                "HWY", "HWZ", "WII"]
            self.precompiled_cliffords = {}
            for label in self.Clifford_Gate_set:
                matrix = np.linalg.multi_dot([self.gate_set[gate] for gate in label])
                unitary = UnitaryGate(matrix, label=label)
                qc = QuantumCircuit(1)
                qc.append(unitary, [0])
                transpiled =  transpile(qc, basis_gates=self.backend.configuration().basis_gates, optimization_level=3)
                self.precompiled_cliffords[label] = transpiled
        
        
        self.shadow_spectro = shadow_spectro
        self.shadow = shadow_spectro.shadow
        self.shadow_size = self.shadow_spectro.shadow_size
        self.nq = self.shadow_spectro.nq
        self.num_processes = min(40, int(mp.cpu_count() * 0.40))
        self.chunksize = max(1, self.shadow_size// (self.num_processes * 5))
        mp.set_start_method("spawn", force=True)
 
       

    def get_time_evolve_circuits(self, hamil, init_state: Union[np.ndarray, list, QuantumCircuit] = None, verbose: bool = True,  N_Trotter_steps: int = 1000):
        """
        Generate time-evolution circuits with random Clifford layers for classical shadows.

        Args:
            hamil: A function or object that returns a UnitaryGate or generates a circuit via `.gen_quantum_circuit(t)`.
            init_state: Initial state as a state vector, list, or QuantumCircuit.
            verbose: Whether to show a progress bar.
            backend: Optional backend used for Clifford randomization.
            N_trotter_step: Number of Trotter steps if Trotterization is used.

        Returns:
            Tuple:
                - List of QuantumCircuits to execute on hardware.
                - List of corresponding Clifford gates applied at each step.
        """
        self.init_state=init_state
        if isinstance(self.init_state, (np.ndarray, list)):
            init_circ = QuantumCircuit(self.nq)
            init_circ.initialize(self.init_state, list(range(self.nq)), normalize=True)
            self._cached_init_circ = transpile(init_circ, optimization_level=3, basis_gates=self.backend.configuration().basis_gates)

        elif isinstance(self.init_state, QuantumCircuit):
            self._cached_init_circ = transpile( self.init_state.copy(),optimization_level=3, basis_gates=self.backend.configuration().basis_gates)

        T = np.linspace(0, self.shadow_spectro.Nt *
                        self.shadow_spectro.dt, self.shadow_spectro.Nt)
        self.file_name = f"Shadow_spectro_nq{self.nq}_Nt{self.shadow_spectro.Nt}_dt{self.shadow_spectro.dt:.2}_NTrot{N_Trotter_steps}_Ns{self.shadow_size}"
        Time_evolve_Clifford_array = []
        job_res=[]
        if not self.hardware.is_fake: 
            batch=Batch(backend=self.hardware.backend)
            print(batch.details())
            self.sampler = Sampler(mode=batch)
            print("waiting 2 min starting...")
            time.sleep(120)
        else: 
            self.sampler = Sampler(self.hardware.backend)

        for k, t in tqdm(enumerate(T), desc="Generate circuit Time evolution", disable=not verbose):
            
            Time_evolve_circuit_array=[]
            circ = hamil.gen_quantum_circuit(
                t, init_state=self._cached_init_circ,  N_Trotter_steps=N_Trotter_steps)

            self.C=transpile( circ,optimization_level=3, basis_gates=self.backend.configuration().basis_gates)
            Clifford_Gate_array = []
            with mp.Pool(processes=self.num_processes) as pool :
                res=pool.map(self.classical_shadow, [i for i in range(self.shadow_size)], chunksize=self.chunksize)
            Clifford_Gate_array, Time_evolve_circuit_array=zip(*res)           
            
            Time_evolve_Clifford_array.append(Clifford_Gate_array)
            print("depth circuits : ", Time_evolve_circuit_array[0].depth())
            N=int(len(Time_evolve_circuit_array)/250)
            if N==0:
                N=1
            list_pubs_for_sampling= self.split_list_into_n_sublists(Time_evolve_circuit_array, N)
            
            sub_job=[]
            if not self.hardware.is_fake and ((k+1)%10==0):
                # If the backend is QPU, we need to create a new batch every 20 circuits
                print("batch closing...")
                time.sleep(120)
                batch.close()
                print("Creation of a new batch...")
                batch=Batch(backend=self.hardware.backend)
                print(batch.details())
                self.sampler = Sampler(mode=batch)
                print("waiting 2 min starting...")
                time.sleep(120)
                
            for n, sub_list in enumerate(list_pubs_for_sampling): 
                job = self.sampler.run(sub_list, shots=1)  
                print("number of pub:", len(sub_list))
                print(f"Done sending sub circuits to backend {n+1}/{len(list_pubs_for_sampling)}")
                self.hardware.print_job_info(job)
                if self.hardware.is_fake:  # If the backend is fake, we can get the data directly
                    sub_job.append(self.hardware.get_data_from_results(job.result()))
                else:  # If the backend is QPU, we need to get the job id. We fetch the result from the job id later
                    sub_job.append(job.job_id())
            
            job_res.append(sub_job)
            print("#################################################")
        save_to_file(self.file_name+"_data","Hardware_simulation",  # We suppose that the file is run in Frontier_Lab, such as 'python Hardware_simulation/post_process.py'
            use_auto_structure=False,
            format="pickle",
            Nt= self.shadow_spectro.Nt, dt= self.shadow_spectro.dt, nq=self.nq, Time_evolve_Clifford_array=Time_evolve_Clifford_array,shadow_size=self.shadow_size, job_res=job_res)    


    def deflatten_Time_evolve_bit_string_array(self, flattenTime_evolve_bit_string_array):
        """
        Restructure flat measurement results into time-evolution organized structure.

        Args:
            flattenTime_evolve_bit_string_array: A flat list of measurement results.

        Returns:
            Nested list: [Nt][shadow_size] formatted bitstrings.
        """
        Time_evolve_bit_string_array = []
        for nt in range(self.shadow_spectro.Nt):
            tmp = []
            for shadow in range(self.shadow_size):
                tmp.append(flattenTime_evolve_bit_string_array[0])
                flattenTime_evolve_bit_string_array.pop(0)
            Time_evolve_bit_string_array.append(tmp)
        return Time_evolve_bit_string_array

    def get_data_matrix_from_hardware(self, job_id: str, Clifford: list[list[UnitaryGate]], density_matrix: bool = True, verbose: bool = False):
        """
        Reconstruct the data matrix from hardware results, using classical shadows.

        Args:
            job_id: Job identifier(s) returned from hardware submission.
            Clifford: Nested list of Clifford gates used in measurements.
            density_matrix: If True, reconstruct the density matrix at each step.
            verbose: Whether to print progress.
            res_fake_backend: Optional list of precomputed bitstrings (used in simulation mode).

        Returns:
            Tuple:
                - Reconstructed spectral signal (array).
                - Extracted frequency components (array).
        """
        
        Bit_string_array=self.get_res_from_hardware(job_id)
        D = []
        for t in tqdm(range(self.shadow_spectro.Nt), desc="Spectral cross_correlation", disable=not verbose):
            if density_matrix:
                Rho = np.zeros((2**self.nq, 2**self.nq), dtype='complex128')
                for i in range(self.shadow_size):
                    Rho += self.shadow.snapshot_density_matrix(
                        Clifford[t][i], Bit_string_array[t][i])
                Rho = Rho/self.shadow_size
                fkt = self.shadow_spectro.expectation_value_q_pauli(Rho, multiprocessing=True)
                D.append(fkt.tolist())
            else:
                fkt = self.shadow_spectro.expectation_value_q_pauli((Clifford[t], Bit_string_array[t]),multiprocessing=True)
                D.append(fkt.tolist())
        D = np.array(D)
        save_to_file("data_matrix_Trotter_fez_1250", folder_name="data",use_auto_structure=False, D=D)
        frequencies,solution = self.shadow_spectro.spectro.Spectroscopy(D)

        return  frequencies,solution

    def split_list_into_n_sublists(self,lst, n):
        """
        Splits a list `lst` into `n` sublists, distributing elements as evenly as possible.
        
        Args:
            lst (list): The list to split.
            n (int): Number of sublists to split into.
        
        Returns:
            List[List]: A list of `n` sublists.
        """
        k, r = divmod(len(lst), n)  # k = size of each sublist, r = remainder
        sublists = []
        start = 0
        for i in range(n):
            end = start + k + (1 if i < r else 0)
            sublists.append(lst[start:end])
            start = end
        return sublists
    
    
    
    
    
    def get_res_from_hardware(
        self,
        job_id: str,
        Folder_path="data/Hardware_simulation/",
    ):
        Measurement_bit_string = []
        if self.hardware.is_fake :
            for sub_list in job_id: 
                for sub_sub_list in sub_list : 
                    for bit in sub_sub_list:

                        Measurement_bit_string.append(bit)

        else:
            raw_data = []
            for n,id in enumerate(job_id):
                print("data from job id: ", n, "/", len(job_id)-1)
                for sub_id in id:
                    raw_data.append(self.hardware.get_sampler_result(sub_id))
                
            for sublist in raw_data:
                for bits in sublist:
                    Measurement_bit_string.append(bits)
                    
            save_to_file(
                "measurement_bit_string",
                folder_name="Hardware_simulation",  # We suppose that the file is run in Frontier_Lab, such as 'python Hardware_simulation/post_process.py'
                use_auto_structure=False,
                format="pickle",
                Measurement_bit_string=Measurement_bit_string)            
                    
        Bit_string_array = self.deflatten_Time_evolve_bit_string_array(
            Measurement_bit_string
        )

        return Bit_string_array
    

     
     
    def classical_shadow(self,_):
        circuit_copy = QuantumCircuit(self.nq)
        circuit_copy.compose(self.C, inplace=True)
        clifford_labels = []
        for qubit in range(self.nq):
            label = self.random_clifford_gate()
            clifford_circ = self.precompiled_cliffords[label]
            circuit_copy.compose(clifford_circ, qubits=[qubit], inplace=True)
            clifford_labels.append(label)

        circuit_copy.measure_all()
        transpile_circ_for_backend=transpile(circuit_copy, backend=self.backend, optimization_level=1) 
        return clifford_labels, transpile_circ_for_backend
        
        
    def random_clifford_gate(self, idx: int = None) -> UnitaryGate:
        """Get a random clifford gate from the Clifford gate set"""
        if idx is None:
            return secrets.choice(self.Clifford_Gate_set)
        return self.Clifford_Gate_set[idx]