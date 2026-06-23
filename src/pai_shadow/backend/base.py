"""Abstract simulation backend and noise specification."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Sequence

import numpy as np

from .circuit import Circuit


NOISE_KINDS = ("depolarizing", "bitflip", "phaseflip", "amplitude_damping")


@dataclass
class NoiseSpec:
    """Per-gate noise channel applied after each targeted gate.

    Parameters
    ----------
    p1, p2:
        Error probabilities for single- and two-qubit gates.
    kind:
        Noise channel: ``"depolarizing"``, ``"bitflip"``, ``"phaseflip"``
        (dephasing) or ``"amplitude_damping"``. Only depolarizing has a genuine
        two-qubit version; for the other kinds the single-qubit channel is
        applied to each qubit of a two-qubit gate.
    one_qubit_gates, two_qubit_gates:
        Gate names that receive 1- and 2-qubit noise. Defaults follow the
        paper's time-evolution gateset (RX, RZ / RXX, RYY, RZZ).
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


class Backend(ABC):
    """Common interface every simulation backend implements.

    All methods take a backend-independent :class:`Circuit`. Conventions:
    amplitudes are little-endian (qubit 0 is the least-significant bit); a
    ``pauli`` observable is a length-``num_qubits`` string over ``IXYZ`` where
    ``pauli[i]`` acts on qubit ``i``; sampled bitstrings are returned with
    qubit ``n-1`` left-most and qubit ``0`` right-most.
    """

    name: str = "backend"

    def __init__(self, noise: NoiseSpec | None = None):
        self.noise = noise or NoiseSpec()

    @abstractmethod
    def statevector(self, circuit: Circuit) -> np.ndarray:
        """Ideal (noiseless) output state vector of ``circuit``."""

    @abstractmethod
    def expectation(self, circuit: Circuit, pauli: str) -> float:
        """Expectation value ``<pauli>`` of the output state.

        With a non-trivial :class:`NoiseSpec` the **exact noisy** value is
        returned via density-matrix simulation (no sampling).
        """

    @abstractmethod
    def sample(self, circuit: Circuit, shots: int = 1) -> List[str]:
        """Measure all qubits, returning ``shots`` bitstrings.

        If a non-trivial :class:`NoiseSpec` is configured, each shot is an
        independent noisy realisation.
        """

    def sample_one(self, circuit: Circuit) -> str:
        """Convenience: a single measurement bitstring."""
        return self.sample(circuit, shots=1)[0]
