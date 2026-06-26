"""Qulacs-native circuit IR and simulation.

A :class:`Circuit` is a flat, picklable list of :class:`Gate` specs on a fixed
number of qubits, with an optional initial state vector. The gate specs stay
lightweight (plain dataclasses) so TE-PAI can generate them by the thousand and
ship a :class:`~pai_shadow.te_pai.TEPAI` instance across process boundaries; the
circuit knows how to run *itself* on qulacs via :meth:`statevector`,
:meth:`expectation` and :meth:`sample`.

Conventions
-----------
* Amplitudes are little-endian: qubit 0 is the least-significant bit.
* Rotation sign convention ``R_P(theta) = exp(-i theta/2 P)``. qulacs uses the
  opposite sign (``RX(i, t) = exp(+i t/2 X)``), so angles are negated when
  translated. Two-qubit Pauli rotations (RXX/RYY/RZZ) are symmetric, so the
  order of their two qubit arguments is irrelevant.
* A ``pauli`` observable is a length-``num_qubits`` string over ``IXYZ`` where
  ``pauli[i]`` acts on qubit ``i``; sampled bitstrings come back with qubit
  ``n-1`` left-most and qubit ``0`` right-most.

Performance
-----------
Native qulacs gate objects are cached by ``(name, param, qubits)`` and applied
**directly to the state** (no per-circuit ``QuantumCircuit`` is built). TE-PAI
reuses a tiny set of gates (angles are exactly ``+/-delta`` or ``pi``), so this
removes the dominant per-circuit construction cost. For the TE-PAI hot loop,
:func:`weighted_expectations` builds the qulacs ``Observable`` once and reuses a
single state buffer across all circuits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import List, Optional, Sequence, Tuple

import numpy as np
from qulacs import DensityMatrix, Observable, QuantumState
from qulacs.gate import (
    RX, RY, RZ, H, S, Sdag, X, Y, Z,
    DenseMatrix, PauliRotation,
    DepolarizingNoise, TwoQubitDepolarizingNoise,
    BitFlipNoise, DephasingNoise, AmplitudeDampingNoise,
)

ONE_QUBIT_ROTATIONS = {"RX", "RY", "RZ"}
TWO_QUBIT_ROTATIONS = {"RXX", "RYY", "RZZ"}
NOISE_KINDS = ("depolarizing", "bitflip", "phaseflip", "amplitude_damping")


# --------------------------------------------------------------------------- #
#  Gate / noise specs (lightweight, picklable)                                 #
# --------------------------------------------------------------------------- #
@dataclass
class Gate:
    """A single operation: ``name`` on ``qubits`` with optional ``param``.

    For ``name == "U"`` an explicit 2x2 ``matrix`` (custom single-qubit unitary,
    e.g. a classical-shadow Clifford) must be supplied.
    """

    name: str
    qubits: Tuple[int, ...]
    param: Optional[float] = None
    matrix: Optional[np.ndarray] = None


@dataclass
class NoiseSpec:
    """Per-gate noise channel applied after each targeted gate.

    Parameters
    ----------
    p1, p2:
        Error probabilities for single- and two-qubit gates.
    kind:
        ``"depolarizing"``, ``"bitflip"``, ``"phaseflip"`` (dephasing) or
        ``"amplitude_damping"``. Only depolarizing has a genuine two-qubit
        version; for the other kinds the single-qubit channel is applied to each
        qubit of a two-qubit gate.
    one_qubit_gates, two_qubit_gates:
        Gate names that receive 1- and 2-qubit noise. Defaults follow the
        paper's time-evolution gateset (RX, RY, RZ / RXX, RYY, RZZ).
    """

    p1: float = 0.0
    p2: float = 0.0
    kind: str = "depolarizing"
    one_qubit_gates: Sequence[str] = ("RX", "RY", "RZ")
    two_qubit_gates: Sequence[str] = ("RXX", "RYY", "RZZ")

    def __post_init__(self):
        if self.kind not in NOISE_KINDS:
            raise ValueError(f"Unknown noise kind {self.kind!r}; expected one of {NOISE_KINDS}.")

    def is_noiseless(self) -> bool:
        return self.p1 == 0.0 and self.p2 == 0.0


def _noiseless(noise: NoiseSpec | None) -> bool:
    return noise is None or noise.is_noiseless()


# --------------------------------------------------------------------------- #
#  qulacs translation (cached native gates)                                    #
# --------------------------------------------------------------------------- #
_ROT_1Q = {"RX": RX, "RY": RY, "RZ": RZ}
_FIXED_1Q = {"H": H, "S": S, "SDG": Sdag, "X": X, "Y": Y, "Z": Z}
_PAULI_ID = {"X": 1, "Y": 2, "Z": 3}
_NOISE_1Q = {
    "depolarizing": DepolarizingNoise,
    "bitflip": BitFlipNoise,
    "phaseflip": DephasingNoise,
    "amplitude_damping": AmplitudeDampingNoise,
}


@lru_cache(maxsize=200_000)
def _native_cached(name, param, qubits):
    """Cached qulacs gate for parameter gates (reused across circuits/states)."""
    if name in ONE_QUBIT_ROTATIONS:
        return _ROT_1Q[name](qubits[0], -param)              # negate: qulacs uses +i sign
    if name in TWO_QUBIT_ROTATIONS:
        pid = _PAULI_ID[name[1]]
        return PauliRotation(list(qubits), [pid, pid], -param)
    return _FIXED_1Q[name](qubits[0])


def native_gate(g: Gate):
    """qulacs gate for a :class:`Gate` (cached, except custom ``U`` matrices)."""
    if g.name == "U":
        return DenseMatrix(g.qubits[0], np.asarray(g.matrix, dtype=complex))
    return _native_cached(g.name, g.param, g.qubits)


@lru_cache(maxsize=100_000)
def _noise_cached(kind, qubits, p):
    """Cached qulacs noise gate(s) after a gate on ``qubits`` with rate ``p``."""
    if len(qubits) == 2:
        if kind == "depolarizing":
            return (TwoQubitDepolarizingNoise(qubits[0], qubits[1], p),)
        return (_NOISE_1Q[kind](qubits[0], p), _NOISE_1Q[kind](qubits[1], p))
    return (_NOISE_1Q[kind](qubits[0], p),)


def _make_observable(pauli: str, nq: int) -> Optional[Observable]:
    """qulacs ``Observable`` for a Pauli string, or ``None`` if all-identity."""
    terms = " ".join(f"{p} {i}" for i, p in enumerate(pauli) if p != "I")
    if not terms:
        return None
    obs = Observable(nq)
    obs.add_operator(1.0, terms)
    return obs


def _int_to_bitstring(value: int, n: int) -> str:
    # qubit n-1 left-most, qubit 0 right-most (little-endian convention).
    return "".join(str((value >> i) & 1) for i in reversed(range(n)))


# --------------------------------------------------------------------------- #
#  Circuit                                                                      #
# --------------------------------------------------------------------------- #
@dataclass
class Circuit:
    """A qulacs-native quantum circuit (gate specs + optional init state)."""

    num_qubits: int
    gates: List[Gate] = field(default_factory=list)
    init_state: Optional[np.ndarray] = None  # length-2**n statevector, optional

    # -- builders -------------------------------------------------------- #
    def add(self, name: str, qubits: Sequence[int], param: float | None = None,
            matrix: np.ndarray | None = None) -> "Circuit":
        self.gates.append(Gate(name, tuple(qubits), param, matrix))
        return self

    def rx(self, q: int, theta: float) -> "Circuit": return self.add("RX", [q], theta)
    def ry(self, q: int, theta: float) -> "Circuit": return self.add("RY", [q], theta)
    def rz(self, q: int, theta: float) -> "Circuit": return self.add("RZ", [q], theta)
    def rxx(self, a: int, b: int, theta: float) -> "Circuit": return self.add("RXX", [a, b], theta)
    def ryy(self, a: int, b: int, theta: float) -> "Circuit": return self.add("RYY", [a, b], theta)
    def rzz(self, a: int, b: int, theta: float) -> "Circuit": return self.add("RZZ", [a, b], theta)
    def h(self, q: int) -> "Circuit": return self.add("H", [q])
    def s(self, q: int) -> "Circuit": return self.add("S", [q])
    def x(self, q: int) -> "Circuit": return self.add("X", [q])
    def y(self, q: int) -> "Circuit": return self.add("Y", [q])
    def z(self, q: int) -> "Circuit": return self.add("Z", [q])
    def unitary(self, q: int, matrix: np.ndarray) -> "Circuit": return self.add("U", [q], matrix=matrix)

    def rotation(self, pauli: str, qubits: Sequence[int], theta: float) -> "Circuit":
        """Append a Pauli rotation from a Pauli label: 'X'->RX, 'ZZ'->RZZ, ..."""
        return self.add("R" + pauli.upper(), qubits, theta)

    def __len__(self) -> int:
        return len(self.gates)

    def depth(self) -> int:
        """Circuit depth (longest path of gates sharing qubits)."""
        layer = [0] * self.num_qubits
        for g in self.gates:
            d = max(layer[q] for q in g.qubits) + 1
            for q in g.qubits:
                layer[q] = d
        return max(layer) if self.gates else 0

    # -- simulation ------------------------------------------------------ #
    def _init_vector(self) -> Optional[np.ndarray]:
        if self.init_state is None:
            return None
        v = np.asarray(self.init_state, dtype=complex)
        return v / np.linalg.norm(v)

    def _new_state(self, density: bool):
        state = DensityMatrix(self.num_qubits) if density else QuantumState(self.num_qubits)
        v = self._init_vector()
        if v is not None:
            state.load(v)
        else:
            state.set_zero_state()
        return state

    def _apply(self, state, noise: NoiseSpec | None) -> None:
        """Apply the circuit's (cached) gates directly to ``state``."""
        noisy = not _noiseless(noise)
        for g in self.gates:
            native_gate(g).update_quantum_state(state)
            if not noisy:
                continue
            nq = len(g.qubits)
            if nq == 1 and g.name in noise.one_qubit_gates and noise.p1 > 0:
                for ng in _noise_cached(noise.kind, g.qubits, noise.p1):
                    ng.update_quantum_state(state)
            elif nq == 2 and g.name in noise.two_qubit_gates and noise.p2 > 0:
                for ng in _noise_cached(noise.kind, g.qubits, noise.p2):
                    ng.update_quantum_state(state)

    def statevector(self) -> np.ndarray:
        """Ideal (noiseless) output state vector."""
        state = self._new_state(density=False)
        self._apply(state, None)
        return state.get_vector()

    def expectation(self, pauli: str, noise: NoiseSpec | None = None) -> float:
        """Expectation value ``<pauli>``.

        With a non-trivial :class:`NoiseSpec` the **exact noisy** value is
        computed via density-matrix evolution (no sampling).
        """
        obs = _make_observable(pauli, self.num_qubits)
        if obs is None:  # all-identity observable
            return 1.0
        density = not _noiseless(noise)
        state = self._new_state(density=density)
        self._apply(state, noise)
        return float(np.real(obs.get_expectation_value(state)))

    def sample(self, shots: int = 1, noise: NoiseSpec | None = None) -> List[str]:
        """Measure all qubits, returning ``shots`` bitstrings.

        With a non-trivial :class:`NoiseSpec` each shot is an independent noisy
        (trajectory) realisation.
        """
        n = self.num_qubits
        if _noiseless(noise):
            state = self._new_state(density=False)
            self._apply(state, None)
            return [_int_to_bitstring(s, n) for s in state.sampling(shots)]
        out: List[str] = []
        for _ in range(shots):
            state = self._new_state(density=False)
            self._apply(state, noise)
            out.append(_int_to_bitstring(state.sampling(1)[0], n))
        return out

    def sample_one(self, noise: NoiseSpec | None = None) -> str:
        """Convenience: a single measurement bitstring."""
        return self.sample(1, noise)[0]

    def evolved_state(self, noise: NoiseSpec | None = None) -> QuantumState:
        """qulacs ``QuantumState`` after applying the circuit.

        Without noise this is the deterministic output state, so callers can
        ``.copy()`` it and take many cheap measurements of the same state (e.g.
        classical-shadow snapshots) without re-applying the (possibly deep)
        circuit each time. With a :class:`NoiseSpec` each call is one independent
        stochastic (trajectory) realisation -- statistically identical to sampling
        a shot from the exact noisy density matrix, but far cheaper for the many
        distinct circuits TE-PAI generates -- so it is re-evolved per shot.
        """
        state = self._new_state(density=False)
        self._apply(state, noise)
        return state

    def evolved_density(self, noise: NoiseSpec | None = None) -> DensityMatrix:
        """qulacs ``DensityMatrix`` after applying the circuit with **exact**
        noise channels.

        qulacs applies each noise gate as the full CPTP channel on the density
        matrix (not a stochastic trajectory), so this is the exact noisy output
        state. It is evolved once and can be measured with ``.sampling(n, seed)``
        for ``n`` single shots -- cheaper than re-evolving per shot when one
        circuit (e.g. a Trotter circuit) needs many shots.
        """
        state = self._new_state(density=True)
        self._apply(state, noise)
        return state


# --------------------------------------------------------------------------- #
#  Batched evaluation (TE-PAI hot path)                                         #
# --------------------------------------------------------------------------- #
def weighted_expectations(circuits, weights, pauli: str,
                          noise: NoiseSpec | None = None) -> np.ndarray:
    """Per-circuit ``weight * <pauli>`` for a list of circuits sharing the same
    qubit count and initial state.

    The qulacs ``Observable`` is parsed once and, in the noiseless case, a single
    state buffer is reused across all circuits, so the per-circuit cost is just
    gate application. Returns an array of length ``len(circuits)`` whose mean is
    the unbiased estimate.
    """
    weights = np.asarray(weights, dtype=float)
    if len(circuits) == 0:
        return np.empty(0)
    nq = circuits[0].num_qubits
    obs = _make_observable(pauli, nq)
    if obs is None:  # <I> = 1 for every circuit
        return weights.copy()

    out = np.empty(len(circuits))
    if _noiseless(noise):
        v = circuits[0]._init_vector()
        state = QuantumState(nq)
        for k, (c, w) in enumerate(zip(circuits, weights)):
            if v is not None:
                state.load(v)
            else:
                state.set_zero_state()
            for g in c.gates:
                native_gate(g).update_quantum_state(state)
            out[k] = w * np.real(obs.get_expectation_value(state))
    else:
        for k, (c, w) in enumerate(zip(circuits, weights)):
            state = c._new_state(density=True)
            c._apply(state, noise)
            out[k] = w * np.real(obs.get_expectation_value(state))
    return out


def make_observables(paulis: Sequence[str], nq: int) -> List[Optional[Observable]]:
    """Pre-parse a list of Pauli strings into reusable qulacs ``Observable`` objects."""
    return [_make_observable(p, nq) for p in paulis]


def weighted_exact_data_row(circuits, weights, observables,
                            noise: NoiseSpec | None = None) -> np.ndarray:
    """Exact (infinite-shot) TE-PAI data-matrix row for many observables.

    Each circuit is evolved **once** as a statevector and the *exact* expectation
    of every observable is read off that single state, then accumulated with the
    circuit's quasiprobability weight. With a :class:`NoiseSpec` the evolution is a
    single stochastic (trajectory) realisation -- the noise gates act on the
    statevector -- so averaging over the ``M`` circuits recovers the noisy
    expectation. Returns ``(1/M) * sum_s w_s * <O>_s`` per observable
    (length ``len(observables)``).

    Versus a classical-shadow snapshot this removes the single-shot ``3**k``
    measurement variance, leaving only the TE-PAI sampling variance
    ``gamma**2 / M`` (plus, when noisy, the trajectory variance) -- the regime in
    which the spectra are clean (paper Fig. 1/2). The cost is evaluating every
    observable per circuit rather than one shared random measurement. Using a
    statevector trajectory (not a density matrix) keeps this affordable.
    """
    weights = np.asarray(weights, dtype=float)
    No = len(observables)
    acc = np.zeros(No)
    if len(circuits) == 0:
        return acc
    nq = circuits[0].num_qubits
    apply_noise = None if _noiseless(noise) else noise
    v = circuits[0]._init_vector()
    state = QuantumState(nq)                       # reused buffer (statevector)
    for c, w in zip(circuits, weights):
        if v is not None:
            state.load(v)
        else:
            state.set_zero_state()
        c._apply(state, apply_noise)               # trajectory shot when noisy
        for i, obs in enumerate(observables):
            acc[i] += w if obs is None else w * np.real(obs.get_expectation_value(state))
    return acc / len(circuits)
