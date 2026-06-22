# Written by: Yuto Morohoshi, udpdated by Hugo PAGES

# Standard library imports
import multiprocessing as mp
import gc
import collections

# Third-party imports
import numpy as np
import matplotlib.pyplot as plt

# Local application imports
from te_pai_shadow_spectro_hardware import TE_PAI_Shadow_Spectro_Hardware
from Hardware import Hardware
from pai_shadow.te_pai_shadow import TE_PAI_Shadow_Spectroscopy
from pai_shadow.shadow_spectro.shadow_spectro import ShadowSpectro
from pai_shadow.shadow_spectro.spectroscopy import Spectroscopy
from pai_shadow.shadow_spectro.classical_shadow import ClassicalShadow
from pai_shadow.hamil import Hamiltonian
from pai_shadow.hamil import Ising_Hamil
from pai_shadow.tools_box.quantum_tools import *
from pai_shadow.tools_box.data_file_functions import (
    get_data_file,
    get_keywords_pickle_file,
    save_to_file,get_file_parent_folder_path
)
from pai_shadow.tools_box.plots_functions import plot_spectre, plot_multiple_data

if __name__ == "__main__":
    
    path_data=""
    path_res_jobs=""
    Folder_path=get_file_parent_folder_path(path_data)
    Nt, dt, numQs, Time_evolve_Clifford_array, GAMMA,M_sample, shadow_size=get_data_file(path_data)




    # Parameters :
    k: int = 3  # K Pauli observable
    file_name = f"nq{numQs}_Nt{Nt}_dt{dt:.2}_NsTE{shadow_size}"


    te_pai_shadow = TE_PAI_Shadow_Spectroscopy(None,None,dt,Nt, shadow_size,K=k, num_qubits=numQs)

    job_id= get_data_file(path_res_jobs)
    
    
    # -------- Hardware settings --------
    token = ""  # Replace with your actual token if needed
    is_save = True
    hardware = Hardware(token=token,initiate_service=False,is_fake=False, set_backend=True)  # Hardware class to retrieve data from the job.id
   
    # -------- Hardware settings --------


    TE_PAI_Shadow_Hardware = TE_PAI_Shadow_Spectro_Hardware(hardware, te_pai_shadow, post_process_mode=True) 

    
        
    TE_PAI_Shadow_Hardware.M_sample = M_sample
    
    # -------- Post processing --------
    
    Bit_string = TE_PAI_Shadow_Hardware.get_res_from_hardware(
        job_id[0]
    )
    
    
    
    
    print("#############################################")
    print("#############################################")
    
    solution, frequencies = TE_PAI_Shadow_Hardware.post_processing(GAMMA, Time_evolve_Clifford_array,Bit_string, is_verbose=True, is_density_matrix=False)
    

  
    
    # -------- save and plot --------
    save_to_file(name=file_name+"res_spectro", folder_name=Folder_path, use_auto_structure=False,solution=solution, frequency=frequencies )
    plot_spectre(frequencies, solution,Energy_gap=[4], save_as=file_name+"_Spetrum_Hardware", Folder=Folder_path)
    plt.show()
     # -------- save and plot --------




   