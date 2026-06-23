import warnings
import time, sys, os
folder = os.path.dirname(os.path.dirname(
    os.path.abspath(os.path.dirname(sys.argv[0]))))
sys.path.append(folder)
warnings.filterwarnings('ignore')
import numpy as np
from pai_shadow.tools_box.quantum_tools import get_q_local_Pauli, get_expectation_value, deserialize_circuit
from pai_shadow.tools_box.plots_functions import *
from pai_shadow.te_pai.te_pai import TE_PAI
from pai_shadow.hamiltonian.ising_hamil import Ising_Hamil
from pai_shadow.hamiltonian.heisenberg_hamil import Heisenberg_Hamil
from qiskit import QuantumCircuit
import sys
import time
import os


def test_arg_type():
    nq = 4
    J = 0.1
    g = 5
    delta = np.pi / (2**10)
    hamil = Ising_Hamil(nq, J, g)
    Initial_state=hamil.ground_state()
    t=0  # test of type and length of every attributs
    trotter = TE_PAI(hamil, nq, delta,t, trotter_steps=1.e-4, error=0.1,N_trotter_max=2000, init_state=Initial_state, serialize=False)
    assert trotter.M_sample== 1200
    assert trotter.N== 1
    trotter.gen_te_pai_circuits()
    assert isinstance(trotter.TE_PAI_Circuits, list)
    assert len(trotter.TE_PAI_Circuits)==1200
    assert isinstance(trotter.GAMMA, np.ndarray)
    assert len(trotter.GAMMA)==1200
    assert isinstance(trotter.angles, np.ndarray) 
    assert len(trotter.angles)== trotter.N
    assert isinstance(trotter.angles[0], np.ndarray)
    assert len(trotter.angles[0])== len(hamil.terms)
    assert isinstance(trotter.angles[0][0],float)
    assert isinstance(trotter.gamma, np.float64)
    assert isinstance(trotter.probs, np.ndarray)
    assert len(trotter.probs)==trotter.N
    assert isinstance(trotter.probs[0], np.ndarray)
    assert len(trotter.probs[0])==len(hamil.terms)
    assert isinstance(trotter.probs[0][0], np.ndarray)
    assert len(trotter.probs[0][0])==3
    assert isinstance(trotter.terms, np.ndarray) 
    assert isinstance(trotter.terms[0], np.ndarray) 
    assert len(trotter.terms[0])== len(hamil.terms)
    assert isinstance(trotter.terms[0][0], np.ndarray) 
    assert isinstance(trotter.terms[0][0][0], str) 
    assert isinstance(trotter.terms[0][0][1], list) 
    assert isinstance(trotter.terms[0][0][2], float) 
    for ms in range(trotter.M_sample):
        assert isinstance(trotter.TE_PAI_Circuits[ms],QuantumCircuit)
        assert isinstance(trotter.GAMMA[ms], float)
    obs="XIII"
    res_avg, res_std=trotter.get_expectation_value(obs)
    assert isinstance(res_avg, float)
    assert isinstance(res_std, float)
    t=4 #test of maximum length for big t
    trotter = TE_PAI(hamil, nq, delta,t, init_state=Initial_state, serialize=False)
    assert trotter.M_sample== 5000
    assert trotter.N== 8000
    t=0.1 #test serialization
    trotter = TE_PAI(hamil, nq, delta,t, trotter_steps=1.e-4, error=0.1,N_trotter_max=1, init_state=Initial_state, serialize=True)    
    trotter.gen_te_pai_circuits()
    assert isinstance(trotter.TE_PAI_Circuits, list)
    assert len(trotter.TE_PAI_Circuits)==1200
    for ms in range(trotter.M_sample):
        assert isinstance(trotter.TE_PAI_Circuits[ms],str)
        try: 
            deserialize_circuit(trotter.TE_PAI_Circuits[ms])
        except Exception as e:
            assert False, print(f"{e}")
    t=0.1 #test "run_te_pai" method: gen Qc and calculate expectation value. 
    trotter = TE_PAI(hamil, nq, delta,t, trotter_steps=1.e-4, error=0.1,N_trotter_max=1000, init_state=Initial_state, serialize=True)    
    res_avg, res_std=trotter.run_te_pai(obs)  
    assert isinstance(res_avg, float)
    assert isinstance(res_std, float)
        
        
        
        
def test_TE_PAI():
    nq = 6
    delta = np.pi /(2**6)
    # hamil = Ising_Hamil(nq,0.1,1)
    trotter_steps=0.012
    N_trotter_max=200
    M_sample_max=1500
    PAI_error=0.01
    hamil=Heisenberg_Hamil(nq,1,1,1)
    ground_energy, first_excited_energy, ground_components, excited_components =  hamil.get_ground_and_excited_state(n=15)
    print("energy gap:", first_excited_energy-ground_energy)
    
    observables = get_q_local_Pauli(nq, 1)[:3]
    qc = QuantumCircuit(nq)
    qc.h(i for i in range(nq-1))
    res_PAI=[]
    res_Ideal=[]
    observables=["X"]
    T=np.linspace(0, 3, 4)
    # T=[3.5] 
    for t in T:
        trotter = TE_PAI(hamil, nq, delta, t,init_state=qc,trotter_steps=trotter_steps, PAI_error=PAI_error,N_trotter_max=N_trotter_max,serialize=True, M_sample_max=M_sample_max)
        trotter.gen_te_pai_circuits()
        circ_hamil=hamil.gen_quantum_circuit(t,N_Trotter_steps=N_trotter_max,init_state=qc)
        print(trotter.get_average_depth())
        for obs in observables:
            res_PAI.append(trotter.get_expectation_value("".join(obs), multiprocessing=False)[0])
            res_Ideal.append(get_expectation_value(circ_hamil,"".join(obs)))
            if np.abs(res_Ideal[-1]-res_PAI[-1]) > 0.1:
                print("ERROR:")
            print(f"t={np.round(t,2)}, {"".join(obs)}, PAI={res_PAI[-1]}, Ideal={res_Ideal[-1]}, error={np.abs(res_Ideal[-1]-res_PAI[-1])}")

def test_number_of_gate():
    import matplotlib.pyplot as plt
    numQs=10
    delta = np.pi / (2**7)
    hamil = Ising_Hamil(numQs, 0.8, 0.5)
    T =np.linspace(0, 5, 150)
    qc = QuantumCircuit(numQs)
    qc.h([i for i in range(numQs)])
    Gate_max=[]
    for t in T: 
        print("time : ", t)
        trotter = TE_PAI(hamil, numQs, delta, t,init_state=qc, PAI_error=0.1,trotter_steps=0.015,N_trotter_max=350,serialize=False, M_sample_max=800)
        trotter.gen_te_pai_circuits()
        count=[]
        for circ in trotter.TE_PAI_Circuits:
            count.append(circ.size())
        print("max:",np.max(count)) 
        print("min:",np.min(count))
        print("avg:",np.mean(count))   
        print("std:",np.std(count))
        Gate_max.append(np.max(count))

    plot_data(T,Gate_max,title="Maximum number of gate per iteration, for 4 Qubits Transverse Ising Hamiltonian")
    plt.show() 
        
test_TE_PAI()