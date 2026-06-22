# Written by: Hugo PAGES
# Date: 2024-01-05

# Local application imports
from itertools import product
import pai_shadow.hamiltonian as Hamil
from pai_shadow.te_pai_shadow.te_pai_shadow_spectroscopy import TE_PAI_Shadow_Spectroscopy
from pai_shadow.tools_box.plots_functions import *
from pai_shadow.tools_box.quantum_tools import *
from pai_shadow.tools_box.data_file_functions import *
from pai_shadow.shadow_spectro import ClassicalShadow, Spectroscopy, ShadowSpectro

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np

# Standard library imports
import warnings
warnings.filterwarnings('ignore')





numQs = 6

def J(t):
    return 1

terms = [
    (gate, [k, k + 1], J if gate ==
        "XX" else J if gate == "YY" else J)
    for k, gate in product(range(numQs-1), ["XX", "YY", "ZZ"])  #Heisenberg Hamiltonian
    ]
################# Hamiltonian definition ################# 


if __name__ == "__main__":
    
    ################# Parameters ################# 
    delta : float = (np.pi / (2**6)) 
    shadow_size_TE_PAI : int = 1
    shadow_size : int = 1500
    N_trotter_max : int = 150
    trotter_step : float = 0.005
    PAI_error : float = 0.001
    k : int = 4
    Nt : int = 70
    dt : float = 2/Nt
    M_sample_max : int =1500
    folder : str ="data"
    print(f"dt={dt}, Nt={Nt}, shadow_size_TE_PAI={shadow_size_TE_PAI}, shadow_size={shadow_size}, N_trotter_max={N_trotter_max}, trotter_step={trotter_step}, PAI_error={PAI_error}, M_sample_max={M_sample_max}")
    ################# Parameters ################# 
    
    
    ######## Generation Initial State ########
    hamil = Hamil.Hamiltonian(numQs, terms)
    ground_energy, first_excited_energy, ground_components, excited_components=hamil.get_ground_and_excited_state(n=40)
    print("Theoretical energy gap : ",np.abs(ground_energy-first_excited_energy))
    Initial_state= ground_components + excited_components
    
    
    
    ################  Simulations ################ 

    shadow = ClassicalShadow()
    spectro = Spectroscopy(Nt, dt)
    shadow_spectro = ShadowSpectro(shadow, spectro, numQs, k, shadow_size)
    te_pai_shadow = TE_PAI_Shadow_Spectroscopy(
        hamil, delta, dt, Nt, shadow_size_TE_PAI, K=k, trotter_steps=trotter_step, N_trotter_max=N_trotter_max, PAI_error=PAI_error, init_state=Initial_state, M_sample_max=M_sample_max)
    
    # TE_PAI Shadow spectro
    solution_TE_PAI, frequencies_TE_PAI = te_pai_shadow.main_te_pai_shadow_spectro(
        density_matrix=True, serialize=True)
     # plot and save
    file_name = f"spectrum_nq{numQs}_J{J(1)}_Nt{Nt}_dt{dt:.2}_NTrot{N_trotter_max}_Msample{M_sample_max}_NsTE{shadow_size_TE_PAI}_Ns{shadow_size}_delta{delta:.2}"
    plot_spectre(frequencies_TE_PAI, solution_TE_PAI, label="TE_PAI shadow",
                 save_as=file_name+"_spectre_TE_PAI", Folder=folder)
    plt.show()
    # Trotter shadow spectro 
    solution, frequencies = shadow_spectro.shadow_spectro(
        hamil, init_state=Initial_state, N_Trotter_steps=N_trotter_max, density_matrix=False, serialize=True, multiprocessing=True)
    
    ################  Simulations ################ 

    ################  Save and plot  ################ 
    
    # save data
    save_to_file(file_name, folder_name=folder, format="pickle",use_auto_structure=False,
                             Nt=Nt, dt=dt, solution=solution, frequencies=frequencies,
                             solution_TE_PAI=solution_TE_PAI, frequencies_TE_PAI=frequencies_TE_PAI,
                             Energy_gap=hamil.energy_gap(), J=J(1))

   
    plot_spectre(frequencies, solution, save_as=file_name +
                 "spectre_shadow", Folder=folder)
    plot_multiple_data([frequencies_TE_PAI, frequencies], [solution_TE_PAI, solution], labels=["TE_PAI", "Shadow"],
                       title="TE_PAI and Trotter spectrums",
                       save_as=file_name+"spectre_Multi", Folder=folder)

    # plt.show()