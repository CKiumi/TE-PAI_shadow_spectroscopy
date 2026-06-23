import sys
import pathlib
# Hardware (real-device) code lives in the independent figures/ tree, not in the
# installed pai_shadow package. Put figures/ on the path to import it.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "figures"))
from hardware.hardware import Hardware




def test_initiate_service():
    try:
        Hardware()
        assert True
    except Exception as e:
        assert False, f"Error encountered: {e}"


def test_send_sampler_real_backend():
    hardware = Hardware()
    from qiskit.circuit.random import random_circuit
    qc = [None]*5
    for i in range(5):
        qc[i] = random_circuit(
            num_qubits=6, depth=5).measure_all(inplace=False)
    try:
        hardware.send_sampler_pub(qc)
        assert True
    except Exception as e:
        assert False, f"Error encountered: {e}"


def test_get_sampler_result():
    hardware = Hardware()
    from qiskit.circuit.random import random_circuit
    qc = []
    for _ in range(5):
        qc.append(random_circuit(num_qubits=6,
                  depth=5).measure_all(inplace=False))
    id = hardware.send_sampler_pub(qc, nshots=1)
    res = hardware.get_sampler_result(id)
    assert isinstance(res, list) and len(
        res) == 5, "Error for nshot=1, should return list[str] of length 5 "
    id = hardware.send_sampler_pub(qc, nshots=4000)
    res = hardware.get_sampler_result(id)
    assert isinstance(res, list) and len(res) == 5 and isinstance(res[0], dict)
