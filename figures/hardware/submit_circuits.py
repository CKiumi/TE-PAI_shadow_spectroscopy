# Third-party imports
from itertools import product
from matplotlib import pyplot as plt
import numpy as np

# Local application imports
from te_pai_shadow_spectro_hardware import TE_PAI_Shadow_Spectro_Hardware
from Hardware import Hardware
from shadow_spectro_hardware import ShadowSpectro_Hardware
from pai_shadow.te_pai_shadow import TE_PAI_Shadow_Spectroscopy
from pai_shadow.shadow_spectro.shadow_spectro import ShadowSpectro
from pai_shadow.shadow_spectro.spectroscopy import Spectroscopy
from pai_shadow.shadow_spectro.classical_shadow import ClassicalShadow
from pai_shadow.hamil import Hamiltonian
from pai_shadow.tools_box.quantum_tools import *
from pai_shadow.tools_box.data_file_functions import get_data_file


numQs = 4

def J(_):
    return -0.1

def J2(_):
    return -2

terms = [("ZZ", [k, (k + 1)], J)for k in range(numQs-1)]
terms += [("X", [k], J2) for k in range(numQs)]

if __name__ == "__main__":
    # # Parameters :
    k: int = 3  # K Pauli observable
    shadow_size: int = 1 # Shadow size
    Nt: int = 80 # Number of time steps
    delta: float = np.pi / 2**5 # Trotter Delta
    dt: float = 3/Nt  # Time step

    # Hamiltonian:
    hamil = Hamiltonian(numQs, terms)
    
    token = ""
    is_save = False
    name_backend="ibm_kingston"
 
    hardware= Hardware(token=token,initiate_service=True, set_backend=True, is_fake=True, name_backend=name_backend)
    
    # Initial_state = ground_components + excited_components
    Initial_state = QuantumCircuit(numQs)
    Initial_state.h(i for i in range(numQs-1))
    # Creation Shadow spectro
    shadow = ClassicalShadow()
    spectro = Spectroscopy(Nt, dt)
    shadow_spectro = ShadowSpectro(shadow, spectro, numQs, k, shadow_size)
    
    
    
    te_pai_shadow = TE_PAI_Shadow_Spectroscopy(
        hamil,
        delta,
        dt,
        Nt,
        shadow_size,
        K=k,
        trotter_steps=0.015,
        N_trotter_max=10,
        M_sample_max=100,
        PAI_error=0.001,
        init_state=Initial_state ,
    )
    
    shadow_spectro_hardware=ShadowSpectro_Hardware(hardware, shadow_spectro)




    # TE_PAI_Shadow_spectro Hardware
    TE_PAI_Shadow_Hardware = TE_PAI_Shadow_Spectro_Hardware(
        hardware, te_pai_shadow)




    """SCRIPT FOR TE_PAI SHADOW SPECTROSCOPY"""
    TE_PAI_Shadow_Hardware.send_time_evolve_circuits(verbose=True)

    """SRIPT FOR SHADOW SPECTRO"""
    # shadow_spectro_hardware.get_time_evolve_circuits(hamil, Initial_state,verbose=True,N_Trotter_steps=150)
    
    
    
