"""Tests for the simultaneous randomized benchmarking crosstalk metric."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest
from qiskit.quantum_info import Operator

from qcmet.benchmarks.gate_execution_quality_metrics.simultaneous_rb import (
    SimultaneousRB,
)


@pytest.mark.parametrize(
    "subsets, message",
    [
        ([[0]], "At least two qubit subsets"),
        ([[0], []], "must not be empty"),
        ([[0, 1], [1, 2]], "must be disjoint"),
        ([[0], [-1]], "Invalid number of qubits"),
        ([[0], [1.5]], "indices must be int"),
    ],
)
def test_rejects_invalid_subsets(subsets, message):
    """Reject subset definitions that cannot represent disjoint registers."""
    with pytest.raises(ValueError, match=message):
        SimultaneousRB(sequence_lengths=[1], subsets=subsets)


def test_generation_has_one_simultaneous_and_one_isolated_circuit_per_subset():
    """Generate the documented circuit families and identifying metadata."""
    benchmark = SimultaneousRB(
        sequence_lengths=[3, 1],
        subsets=[[0], [2, 3], [5]],
        circuits_per_sequence_length=2,
        seed=7,
    )
    benchmark.generate_circuits()

    data = benchmark.experiment_data
    assert len(data) == 2 * 2 * (1 + 3)
    assert benchmark.config["sequence_lengths"] == [1, 3]
    assert benchmark.labels == ["q0", "q2_3", "q5"]
    assert benchmark.pairs == [(0, 1), (0, 2), (1, 2)]

    for (_, _), group in data.groupby(["sequence_length", "seq_id"]):
        assert len(group) == 4
        assert list(group["mode"]).count("simultaneous") == 1
        assert set(group.loc[group["mode"] == "isolated", "active"]) == {
            "q0",
            "q2_3",
            "q5",
        }


def test_generation_is_reproducible_for_a_fixed_seed():
    """Use the benchmark seed to reproduce every random Clifford circuit."""
    kwargs = {
        "sequence_lengths": [1, 3],
        "subsets": [[0], [1]],
        "circuits_per_sequence_length": 2,
        "seed": 123,
    }
    first = SimultaneousRB(**kwargs)
    second = SimultaneousRB(**kwargs)
    first.generate_circuits()
    second.generate_circuits()

    assert all(
        circuit_a == circuit_b
        for circuit_a, circuit_b in zip(
            first.circuits,
            second.circuits,
            strict=True,
        )
    )


def test_every_generated_sequence_is_inverted_to_identity():
    """Return all active subsets to the initial state in the absence of noise."""
    benchmark = SimultaneousRB(
        sequence_lengths=[1, 4],
        subsets=[[0], [1]],
        circuits_per_sequence_length=2,
        seed=17,
    )
    benchmark.generate_circuits()

    identity = Operator(np.eye(2**benchmark.num_qubits))
    for measured_circuit in benchmark.circuits:
        circuit = measured_circuit.remove_final_measurements(inplace=False)
        assert Operator(circuit).equiv(identity)


def test_noncontiguous_physical_qubits_are_routed_correctly():
    """Route the compact benchmark register only onto requested physical qubits."""
    benchmark = SimultaneousRB(
        sequence_lengths=[2],
        subsets=[[2], [5]],
        circuits_per_sequence_length=1,
        seed=19,
    )
    benchmark.generate_circuits()

    assert all(circuit.num_qubits == 6 for circuit in benchmark.circuits)
    for circuit in benchmark.circuits:
        touched = {
            circuit.find_bit(qubit).index
            for instruction in circuit.data
            if instruction.operation.name != "barrier"
            for qubit in instruction.qubits
        }
        assert touched <= {2, 5}


def test_survival_and_parity_estimators_marginalize_bitstrings():
    """Calculate subset survival and Pauli-Z parity from full-register data."""
    probabilities = {"00": 0.4, "01": 0.1, "10": 0.2, "11": 0.3}

    assert SimultaneousRB._survival(probabilities, [0]) == pytest.approx(0.5)
    assert SimultaneousRB._survival(probabilities, [1]) == pytest.approx(0.6)
    assert SimultaneousRB._survival(probabilities, [0, 1]) == pytest.approx(0.4)
    assert SimultaneousRB._parity(probabilities, [0]) == pytest.approx(0.0)
    assert SimultaneousRB._parity(probabilities, [1]) == pytest.approx(0.2)
    assert SimultaneousRB._parity(probabilities, [0, 1]) == pytest.approx(0.4)


@pytest.mark.parametrize(
    "alpha, width, expected_epc",
    [(0.98, 1, 0.01), (0.92, 2, 0.06)],
)
def test_epc_conversion_uses_subset_dimension(alpha, width, expected_epc):
    """Convert RB decay into EPC using d = 2**subset_width."""
    benchmark = SimultaneousRB(sequence_lengths=[1], subsets=[[0], [1]])
    assert benchmark._epc(alpha, width) == pytest.approx(expected_epc)


def _integer_counts(probabilities: dict[str, float], shots: int) -> dict[str, int]:
    """Round probabilities to integer counts while preserving the shot total."""
    outcomes = list(probabilities)
    expected = np.asarray([probabilities[outcome] for outcome in outcomes]) * shots
    counts = np.floor(expected).astype(int)
    remainder_order = np.argsort(expected - counts)[::-1]
    for index in remainder_order[: shots - int(counts.sum())]:
        counts[index] += 1
    return dict(zip(outcomes, counts.tolist(), strict=True))


def _counts_for_known_decays(
    benchmark: SimultaneousRB,
    isolated_alpha: tuple[float, float],
    simultaneous_alpha: tuple[float, float],
    joint_alpha: float,
    shots: int = 1_000_000,
) -> list[dict[str, int]]:
    """Construct physical two-qubit distributions with prescribed Z decays."""
    all_counts = []
    for row in benchmark.experiment_data.itertuples():
        sequence_length = row.sequence_length
        if row.mode == "isolated":
            subset_index = benchmark.labels.index(row.active)
            z_value = isolated_alpha[subset_index] ** sequence_length
            zero_probability = (1 + z_value) / 2
            excited = "10" if subset_index == 0 else "01"
            probabilities = {"00": zero_probability, excited: 1 - zero_probability}
        else:
            z_0 = simultaneous_alpha[0] ** sequence_length
            z_1 = simultaneous_alpha[1] ** sequence_length
            z_joint = joint_alpha**sequence_length
            probabilities = {
                "00": (1 + z_0 + z_1 + z_joint) / 4,
                "01": (1 + z_0 - z_1 - z_joint) / 4,
                "10": (1 - z_0 + z_1 - z_joint) / 4,
                "11": (1 - z_0 - z_1 + z_joint) / 4,
            }
            assert all(probability >= 0 for probability in probabilities.values())
        all_counts.append(_integer_counts(probabilities, shots))
    return all_counts


def _analyze_known_decays(
    isolated_alpha: tuple[float, float],
    simultaneous_alpha: tuple[float, float],
    joint_alpha: float,
) -> tuple[SimultaneousRB, dict]:
    """Run the public analysis path on exact synthetic decay data."""
    shots = 1_000_000
    benchmark = SimultaneousRB(
        sequence_lengths=[1, 2, 4, 8, 16, 32],
        subsets=[[0], [1]],
        circuits_per_sequence_length=1,
        seed=23,
    )
    benchmark.generate_circuits()
    benchmark.load_circuit_measurements(
        _counts_for_known_decays(
            benchmark,
            isolated_alpha,
            simultaneous_alpha,
            joint_alpha,
            shots,
        )
    )
    benchmark._runtime_params = {"num_shots": shots}
    return benchmark, benchmark.analyze()


def test_independent_noise_has_no_addressability_or_correlation_penalty():
    """Recover zero crosstalk when simultaneous noise factorizes by subset."""
    subset_alpha = (0.97, 0.95)
    benchmark, result = _analyze_known_decays(
        isolated_alpha=subset_alpha,
        simultaneous_alpha=subset_alpha,
        joint_alpha=subset_alpha[0] * subset_alpha[1],
    )

    assert result["AddressabilityError"]["q0"] == pytest.approx(0.0, abs=2e-5)
    assert result["AddressabilityError"]["q1"] == pytest.approx(0.0, abs=2e-5)
    assert result["CorrelationParameter"]["q0|q1"] == pytest.approx(
        0.0,
        abs=2e-5,
    )
    assert result["alpha"]["isolated"]["q0"] == pytest.approx(
        subset_alpha[0],
        abs=2e-5,
    )
    assert result["alpha"]["isolated"]["q1"] == pytest.approx(
        subset_alpha[1],
        abs=2e-5,
    )

    figure, axes = plt.subplots()
    benchmark.plot(axes)
    assert axes.get_xlabel() == r"$\text{Sequence Length}, m$"
    assert axes.get_ylabel() == "Survival Probability, $p_0$"
    plt.close(figure)


def test_correlated_crosstalk_is_recovered_from_known_physical_decays():
    """Detect simultaneous degradation and non-factorizing joint noise."""
    isolated_alpha = (0.98, 0.96)
    simultaneous_alpha = (0.92, 0.90)
    joint_alpha = 0.86
    _, result = _analyze_known_decays(
        isolated_alpha=isolated_alpha,
        simultaneous_alpha=simultaneous_alpha,
        joint_alpha=joint_alpha,
    )

    for index, label in enumerate(("q0", "q1")):
        expected_isolated_epc = (1 - isolated_alpha[index]) / 2
        expected_simultaneous_epc = (1 - simultaneous_alpha[index]) / 2
        assert result["IsolatedEPC"][label] == pytest.approx(
            expected_isolated_epc,
            abs=2e-5,
        )
        assert result["SimultaneousEPC"][label] == pytest.approx(
            expected_simultaneous_epc,
            abs=2e-5,
        )
        assert result["AddressabilityError"][label] > 0

    expected_correlation = joint_alpha - np.prod(simultaneous_alpha)
    assert result["CorrelationParameter"]["q0|q1"] == pytest.approx(
        expected_correlation,
        abs=2e-5,
    )
