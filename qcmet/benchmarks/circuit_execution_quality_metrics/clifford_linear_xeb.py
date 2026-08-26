"""Here we implement the linear cross-entropy benchmarking with Clifford circuits following PRA 108, 052613.

We have implemented a 'cycle' as that of the of the 1D chain (Fig 1a). This is controlled by the depth parameter. 
A cycle thus consists of 4 layers: single qubit clifford layer, entangling layer, single qubit clifford layer, entangling layer.
The entangling layers are brickwork overlapped.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, List

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, StabilizerState, random_clifford

from qcmet.benchmarks.circuit_execution_quality_metrics.linear_xeb import LinearXEB

if TYPE_CHECKING:
    from qcmet.core import FileManager


class CliffordLinearXEB(LinearXEB):
    """Linear XEB benchmark specialized to Clifford circuits."""

    def __init__(
        self,
        qubits: int | List[int],
        depth: int | List[int],
        num_circuits: int,
        seed: int | None = None,
        save_path: str | Path | FileManager | None = None,
    ):
        """Initialize the Clifford linear XEB benchmark."""
        super().__init__(
            qubits=qubits,
            depth=depth,
            num_circuits=num_circuits,
            seed=seed,
            save_path=save_path,
        )
        self.name = "CliffordLinearXEB"

    def _random_single_qubit_clifford_layer(self) -> QuantumCircuit:
        """Generate a random single-qubit Clifford layer across all qubits."""
        circuit = QuantumCircuit(self.num_qubits)
        for i in range(self.num_qubits):
            circuit.compose(
                random_clifford(1, seed=self.rng).to_circuit(),
                qubits=[i],
                inplace=True,
            )
        return circuit

    def _cnot_layer(
                    self,
                    parity: int,
                ) -> QuantumCircuit:
        """Construct one open-chain nearest-neighbour CNOT layer.

        parity=0 gives (0,1), (2,3), ...
        parity=1 gives (1,2), (3,4), ...
        """
        if parity not in (0, 1):
            raise ValueError("parity must be either 0 or 1")

        circuit = QuantumCircuit(self.num_qubits)

        for control in range(parity, self.num_qubits - 1, 2):
            circuit.cx(control, control + 1)

        return circuit

    def build_circuit(self, num_qubits: int, depth: int) -> QuantumCircuit:
        """Build a random Clifford circuit of the specified depth and qubit count."""
        circuit = QuantumCircuit(num_qubits)
        for _ in range(depth):
            circuit.compose(self._random_single_qubit_clifford_layer(), inplace=True)
            circuit.compose(self._cnot_layer(0), inplace=True)
            circuit.compose(self._random_single_qubit_clifford_layer(), inplace=True)
            circuit.compose(self._cnot_layer(1), inplace=True)
        return circuit

    def _generate_circuits(self) -> List[QuantumCircuit]:
        """Generate random Clifford circuits for the configured depths."""
        circuits = []
        depths = self.config["depth"]
        if isinstance(depths, int):
            depths = [depths]

        for depth in depths:
            for _ in range(self.config["num_circuits"]):
                circuit = self.build_circuit(self.num_qubits, depth)
                circuit.measure_all()
                circuits.append(circuit)

        return circuits

    @staticmethod
    def _gf2_nullspace(matrix: np.ndarray) -> list[np.ndarray]:
        """Return a basis for the nullspace of a binary matrix."""
        matrix = np.asarray(matrix, dtype=np.uint8).copy() % 2
        n_rows, n_columns = matrix.shape
        pivot_columns = []
        pivot_row = 0

        for column in range(n_columns):
            candidates = np.flatnonzero(matrix[pivot_row:, column])
            if len(candidates) == 0:
                continue

            row = pivot_row + int(candidates[0])
            matrix[[pivot_row, row]] = matrix[[row, pivot_row]]
            for other_row in range(n_rows):
                if other_row != pivot_row and matrix[other_row, column]:
                    matrix[other_row] ^= matrix[pivot_row]

            pivot_columns.append(column)
            pivot_row += 1
            if pivot_row == n_rows:
                break

        free_columns = [
            column for column in range(n_columns) if column not in pivot_columns
        ]
        basis = []
        for free_column in free_columns:
            vector = np.zeros(n_columns, dtype=np.uint8)
            vector[free_column] = 1
            for row, pivot_column in enumerate(pivot_columns):
                vector[pivot_column] = matrix[row, free_column]
            basis.append(vector)
        return basis

    @staticmethod
    def _diagonal_stabilizer_constraints(
        stabilizer_state: StabilizerState,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the (z_masks, signs) pair describing the diagonal stabilizer subgroup.

        This is the circuit-dependent, bitstring-independent part of computing a
        computational-basis probability from a stabilizer tableau, so it is computed
        once per circuit.
        """
        num_qubits = stabilizer_state.num_qubits
        assert num_qubits is not None

        clifford = stabilizer_state.clifford
        stabilizer_x = np.asarray(clifford.stab_x, dtype=np.uint8)
        stabilizer_z = np.asarray(clifford.stab_z, dtype=np.uint8)
        stabilizer_labels = clifford.to_labels()[num_qubits:]

        # A product of stabilizers is diagonal in the computational basis iff
        # its combined X support vanishes.
        z_only_basis = CliffordLinearXEB._gf2_nullspace(stabilizer_x.T)

        z_masks = np.zeros((len(z_only_basis), num_qubits), dtype=np.uint8)
        signs = np.ones(len(z_only_basis), dtype=np.int64)
        for row, combination in enumerate(z_only_basis):
            product = Pauli("I" * num_qubits)
            for index, selected in enumerate(combination):
                if selected:
                    product = product.compose(Pauli(stabilizer_labels[index]))

            signs[row] = -1 if product.to_label().startswith("-") else 1
            z_masks[row] = stabilizer_z.T @ combination % 2

        return z_masks, signs

    @staticmethod
    def probability_of_bitstring(stabilizer_state: StabilizerState, bitstring: str) -> float:
        """Return one computational-basis probability from a stabilizer tableau."""
        num_qubits = stabilizer_state.num_qubits
        assert num_qubits is not None

        if len(bitstring) != num_qubits or set(bitstring) - {"0", "1"}:
            raise ValueError("bitstring must contain one binary digit per qubit")

        z_masks, signs = CliffordLinearXEB._diagonal_stabilizer_constraints(stabilizer_state)
        raw_bits = np.asarray([int(bit) for bit in bitstring], dtype=np.uint8)

        parity = (z_masks @ raw_bits % 2).astype(np.int64)
        eigenvalues = signs * (-1) ** parity
        if np.any(eigenvalues != 1):
            return 0.0

        return 2.0 ** (-(num_qubits - len(signs)))

    def _ideal_probabilities(self,
                             circuit: QuantumCircuit,
                             observed_bitstrings: List[str],
                             ) -> dict[str, float]:
        
        circuit_without_measurements = circuit.remove_final_measurements(
            inplace=False
        )
        assert circuit_without_measurements is not None
        stabilizer_state = StabilizerState(circuit_without_measurements)
        num_qubits = stabilizer_state.num_qubits
        assert num_qubits is not None

        if not observed_bitstrings:
            return {}

        # Computed once per circuit: this used to be recomputed (at O(n^3) cost,
        # including a full stabilizer-tableau relabeling) for every single observed
        # bitstring, which dominated runtime for large qubit counts / shot counts.
        z_masks, signs = self._diagonal_stabilizer_constraints(stabilizer_state)
        base_probability = 2.0 ** (-(num_qubits - len(signs)))

        bits_matrix = np.asarray(
            [[int(bit) for bit in bitstring] for bitstring in observed_bitstrings],
            dtype=np.uint8,
        )
        parity = ((bits_matrix @ z_masks.T) % 2).astype(np.int64)
        eigenvalues = signs[np.newaxis, :] * (-1) ** parity
        satisfies_all_constraints = np.all(eigenvalues == 1, axis=1)

        return {
            bitstring: (base_probability if satisfied else 0.0)
            for bitstring, satisfied in zip(observed_bitstrings, satisfies_all_constraints)
        }

    def _cross_entropy_fidelity(
                                self,
                                circuit: QuantumCircuit,
                                counts: dict[str, int],
                            ) -> float:
        """Calculate linear XEB as defined in Phys. Rev. A 108, 052613."""
        shots = sum(counts.values())
        if shots == 0:
            raise ValueError("Cannot calculate XEB from zero shots")

        probabilities = self._ideal_probabilities(
            circuit,
            list(counts.keys()),
        )

        overlap = sum(
            (count / shots) * probabilities.get(bitstring, 0.0)
            for bitstring, count in counts.items()
        )

        return float((2.0**circuit.num_qubits) * overlap - 1.0)

    def _analyze(self) -> dict[str, Any]:
        """Analyze the measured Clifford XEB data and return summary metrics."""
        fidelities = []
        for circuit, counts in zip(
            self.experiment_data["circuit"],
            self.experiment_data["circuit_measurements"],
            strict=True,
        ):
            fidelities.append(self._cross_entropy_fidelity(circuit, counts))

        self.experiment_data["linear_xeb_fidelity"] = fidelities
        return {
            "mean_fidelity": float(np.mean(fidelities)),
            "fidelities": fidelities,
        }