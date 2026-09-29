"""test_sequential_benchmark.py.

Unit tests for the SequentialBenchmark in qcmet.benchmarks.sequential_benchmark.
"""
import pytest
from qiskit import QuantumCircuit

from qcmet.benchmarks import BaseBenchmark, SequentialBenchmark
from qcmet.devices import IdealSimulator


class DummyBenchmark(BaseBenchmark):
    """Concrete BaseBenchmark for testing SequentialBenchmark methods."""

    def __init__(self, qubits, parameter, **kwargs):
        """Initialize dummy with a parameter."""
        super().__init__("Dummy", qubits, **kwargs)
        self.parameter = parameter

    def _generate_circuits(self):
        c = QuantumCircuit(max(self.qubits) + 1)
        for i in range(max(self.qubits) + 1):
            c.x(i)
        c.measure_all()
        return [c]

    def _analyze(self):
        return {"max_qubit": max(self.qubits) + 1, "parameter": self.parameter}

class DummySequentialBenchmark(SequentialBenchmark):
    """Concrete SequentialBenchmark for testing SequentialBenchmark methods."""

    def __init__(self):
        """Initialize dummy sequential benchmark to run from 2 qubit to 4 qubits."""
        super().__init__(
            name="DummySequentialBenchmark",
            benchmark_class=DummyBenchmark,
            parameter_sequence=[{"qubits": qubits} for qubits in range(2, 5)],
            fixed_parameters={"parameter": "some value"},
        )
        self.fail_number = 3

    def should_stop(self, results):
        """Check stopping condition."""
        return results["max_qubit"] >= self.fail_number

    def set_stopping_num(self, fail_number):
        """Change the fail condition for testing purposes."""
        self.fail_number = fail_number

    def _analyze(self):
        return {"analysis": "some text"}

@pytest.fixture
def dummy_sb_instance():
    """Fixture to create a dummy benchmark instance with qubit number from 2 to 4."""
    dummy = DummySequentialBenchmark()
    return dummy


def test_generate_circuits(dummy_sb_instance):
    """Verify that SequentialBenchmark generates a dummy circuit for the initial run index."""
    dummy_sb_instance.generate_circuits()
    circuits = dummy_sb_instance.circuits
    assert len(circuits) == 1
    assert circuits[0].num_qubits == 2

    dummy_sb_instance.run_index = 1
    dummy_sb_instance.generate_circuits()
    circuits = dummy_sb_instance.circuits
    assert len(circuits) == 1
    assert circuits[0].num_qubits == 3


def test_should_stop(dummy_sb_instance):
    """Verify that SequentialBenchmark correctly decides whether it stops based on some results."""
    should_stop = dummy_sb_instance.should_stop({"max_qubit": 2})
    assert not should_stop
    should_stop = dummy_sb_instance.should_stop({"max_qubit": 3})
    assert should_stop
    should_stop = dummy_sb_instance.should_stop({"max_qubit": 4})
    assert should_stop


def test_run(dummy_sb_instance):
    """Verify that SequentialBenchmark runs sub-benchmarks sequentially and correctly stops."""
    device = IdealSimulator()
    dummy_sb_instance.generate_circuits()
    dummy_sb_instance.run(device)
    assert len(dummy_sb_instance.benchmarks) == 2
    assert len(dummy_sb_instance.run_records) == 2
    assert dummy_sb_instance.benchmarks[0].circuits[0].num_qubits == 2
    assert dummy_sb_instance.benchmarks[1].circuits[0].num_qubits == 3
    assert dummy_sb_instance.run_records[0]["result"]["max_qubit"] == 2
    assert dummy_sb_instance.run_records[1]["result"]["max_qubit"] == 3
    assert dummy_sb_instance.run_records[0]["result"]["parameter"] == "some value"
    assert dummy_sb_instance.run_records[1]["result"]["parameter"] == "some value"


def test_call_runs_sequence_without_parent_circuit_generation(
        dummy_sb_instance):
    """Verify that calling a sequential benchmark executes its child sequence."""
    result = dummy_sb_instance(IdealSimulator())

    assert result == {"analysis": "some text"}
    assert len(dummy_sb_instance.run_records) == 2
