import sys
import os
folder = os.path.dirname(os.path.dirname(
    os.path.abspath(os.path.dirname(sys.argv[0]))))
sys.path.append(folder)
import time 
from pai_shadow.hamiltonian.hamiltonian import Hamiltonian
from pai_shadow.hamiltonian.heisenberg_hamil import Heisenberg_Hamil
from pai_shadow.hamiltonian.ising_hamil import Ising_Hamil
from pai_shadow.hamiltonian.spin_chain_hamil import Spin_Chain_Hamil
from pai_shadow.tools_box.quantum_tools import deserialize_circuit
from qiskit import QuantumCircuit
from pai_shadow.shadow_spectro.classical_shadow import ClassicalShadow


def test_serialization():
    T=1
    classical=ClassicalShadow()
    import numpy as np
    Init_state=QuantumCircuit(4)
    Init_state.rx(np.pi,[i for i in range(3)])
    Heisenberg_hamil=Heisenberg_Hamil(4,1,1,1, boundarie_conditions=True)
    Heisenberg_circ=Heisenberg_hamil.gen_quantum_circuit(T, trotter_step=1,init_state=Init_state, serialize=False) 
    Heisenberg_circ_seri=Heisenberg_hamil.gen_Qasm_circuit(T, trotter_step=1, init_state=Init_state) 
    print( Heisenberg_circ_seri)
    deserialize_Heisenberg=deserialize_circuit(Heisenberg_circ_seri)
    print(Heisenberg_circ)
    print( deserialize_Heisenberg)
    shadow_circ, gate_set_list=classical.add_clifford_gate_to_qasm(Heisenberg_circ_seri)
    print(shadow_circ)
    print(classical.deserialize_circuit(shadow_circ))
    
#     assert Heisenberg_circ== deserialize_Heisenberg, "error Heiseberg Hamiltonian "
#     # Ising Hamiltonian
#     Ising_hamil = Ising_Hamil(4, 5,5)
#     print(Ising_hamil.energy_gap())
#     Ising_circ = Ising_hamil.gen_quantum_circuit(T, trotter_step=1, serialize=False)
#     Ising_circ_seri = Ising_hamil.gen_Qasm_circuit(T, trotter_step=1) 
#     deserialize_Ising = deserialize_circuit(Ising_circ_seri)
#     print(Ising_circ)
#     print(deserialize_Ising)
#     assert Ising_circ == deserialize_Ising, "errorIsing Hamiltonian "
#     # Spin Chain Hamiltonian
#     Spin_chain_hamil = Spin_Chain_Hamil(4, [0.1]*4)
#     Spin_chain_circ = Spin_chain_hamil.gen_quantum_circuit(T, trotter_step=1, serialize=False)
#     Spin_chain_circ_seri = Spin_chain_hamil.gen_Qasm_circuit(T, trotter_step=1) 
#     deserialize_Spin_Chain = deserialize_circuit(Spin_chain_circ_seri)
#     print( Spin_chain_circ)
#     print(deserialize_Spin_Chain)
#     assert Spin_chain_circ == deserialize_Spin_Chain, "error Spin chain Hamiltonian "
    
test_serialization()