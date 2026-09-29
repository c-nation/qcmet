"""sequential_benchmark.py.

This module provides a utility class for executing a benchmark multiple times
with different parameter values.

Sequential execution runs a benchmark for a user-defined ordered sequence of
parameter dictionaries.

Subclasses must implement ``_analyze`` to compute the aggregate result from
the sequence, and may override ``should_stop`` to terminate the sequence
based on the result of an individual benchmark run.
"""

from __future__ import annotations
from typing import Any

from pathlib import Path

from pandas import DataFrame
from qiskit import QuantumCircuit

from qcmet.benchmarks.base_benchmark import BaseBenchmark
from qcmet.core import FileManager
from qcmet.core.exceptions import MeasurementOutcomesExistError


class SequentialBenchmark(BaseBenchmark):
    """Run a child benchmark over an ordered sequence of parameters.

    ``parameter_sequence`` varies arbitrary constructor parameters between
    runs. For example::

        parameter_sequence = [
            {"qubits": 4, "size": 16},
            {"qubits": 4, "size": 24},
            {"qubits": 6, "size": 48},
        ]

    Parameters in ``parameter_sequence`` override parameters in
    ``fixed_parameters`` for the corresponding run.

    Subclasses must implement:

    - ``_analyze`` to calculate an aggregate result from the completed runs.

    Subclasses may override:

    - ``should_stop`` to define a benchmark-specific stopping condition.

    Attributes:
        benchmarks:
            Child benchmark instances, in execution order.
        run_records:
            Parameters, results, and stopping information for every completed
            run.
        run_index:
            Zero-based index of the next run in the configured sequence.
    """

    def __init__(
        self,
        name: str,
        benchmark_class: type[BaseBenchmark],
        parameter_sequence: list[dict[str, Any]],
        fixed_parameters: dict[str, Any] | None = None,
        save_path: str | Path | FileManager | None = None,
    ):
        """Initialise a sequential benchmark.

        Args:
            name:
                Name of the sequential benchmark.
            benchmark_class:
                ``BaseBenchmark`` subclass to instantiate for each run.
            fixed_parameters:
                Constructor parameters shared by every child benchmark.
            parameter_sequence:
                Ordered list of parameter dictionaries for child benchmarks.
            save_path:
                Optional path or ``FileManager`` for sequential benchmark
                outputs.

        Raises:
            TypeError:
                If ``benchmark_class`` is not a ``BaseBenchmark`` subclass.
            ValueError:
                If the parameter sequence is invalid.
        """
        super().__init__(
            name=name,
            qubits=[],
            save_path=save_path,
        )
        self.qubits = []

        self._validate_benchmark_class(benchmark_class)

        fixed_parameters = (
            {} if fixed_parameters is None else dict(fixed_parameters)
        )

        sequence = self._validate_parameter_sequence(parameter_sequence)

        self.benchmark_class = benchmark_class
        self.fixed_parameters = fixed_parameters
        self.parameter_sequence = sequence

        self.config["benchmark_class"] = benchmark_class.__name__
        self.config["fixed_parameters"] = fixed_parameters
        self.config["parameter_sequence"] = sequence

        self.run_index = 0
        self.benchmarks: list[BaseBenchmark] = []
        self.run_records: list[dict[str, Any]] = []
        self._prepared_stages: dict[int, BaseBenchmark] = {}

    @staticmethod
    def _validate_benchmark_class(
        benchmark_class: type[BaseBenchmark],
    ) -> None:
        """Validate the child benchmark class."""
        if (
            not isinstance(benchmark_class, type)
            or not issubclass(benchmark_class, BaseBenchmark)
        ):
            raise TypeError(
                "benchmark_class must be a subclass of BaseBenchmark"
            )

    @staticmethod
    def _validate_parameter_sequence(
        parameter_sequence: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Validate and copy a user-defined parameter sequence."""
        if not isinstance(parameter_sequence, list):
            raise TypeError(
                "parameter_sequence must be a list of dictionaries"
            )

        if not parameter_sequence:
            raise ValueError("parameter_sequence must not be empty")

        if not all(
            isinstance(parameters, dict)
            for parameters in parameter_sequence
        ):
            raise TypeError(
                "Every entry in parameter_sequence must be a dictionary"
            )

        return [dict(parameters) for parameters in parameter_sequence]

    def _get_parameters(
        self,
        run_index: int,
    ) -> dict[str, Any]:
        """Return the resolved child-benchmark parameters for one run."""
        if run_index < 0 or run_index >= len(self.parameter_sequence):
            raise IndexError(
                f"Run index {run_index} is outside the configured sequence"
            )

        varying_parameters = dict(self.parameter_sequence[run_index])

        # Sequence values intentionally take precedence over fixed_parameters.
        parameters = self.fixed_parameters | varying_parameters

        return parameters

    def _create_benchmark(
        self,
        run_index: int,
    ) -> tuple[BaseBenchmark, dict[str, Any]]:
        """Create the child benchmark for one sequence entry."""
        parameters = self._get_parameters(run_index)
        benchmark = self.benchmark_class(**parameters)
        return benchmark, parameters

    def prepare_stage(
        self,
        run_index: int,
        regenerate: bool = False,
    ) -> BaseBenchmark:
        """Prepare one child stage and cache it for later execution."""
        if run_index in self._prepared_stages:
            benchmark = self._prepared_stages[run_index]
            benchmark.generate_circuits(regenerate=regenerate)
            return benchmark

        benchmark, _ = self._create_benchmark(run_index)
        benchmark.generate_circuits()
        self._prepared_stages[run_index] = benchmark
        return benchmark

    def prepare_all_stages(
        self,
        regenerate: bool = False,
    ) -> list[BaseBenchmark]:
        """Prepare all configured stages without executing them."""
        return [
            self.prepare_stage(index, regenerate=regenerate)
            for index in range(len(self.parameter_sequence))
        ]

    def generate_circuits(self, regenerate: bool = False) -> None:
        """Prepare all child stages without flattening their experiments."""
        self.prepare_all_stages(regenerate=regenerate)

    def _generate_circuits(
        self,
    ) -> list[QuantumCircuit] | dict[str, Any]:
        """Generate raw circuits for the current child stage.

        Public callers should use ``prepare_stage`` or ``generate_circuits``
        so the generated circuits remain attached to the child benchmark that
        will execute them.
        """
        benchmark, _ = self._create_benchmark(self.run_index)
        return benchmark._generate_circuits()

    @property
    def circuits(self) -> list[QuantumCircuit]:
        """Return circuits from prepared stages in stage order."""
        if not self._prepared_stages:
            raise AttributeError(f"Circuits not generated for {self.name}!")
        circuits = []
        for index in sorted(self._prepared_stages):
            circuits.extend(self._prepared_stages[index].circuits)
        return circuits

    def should_stop(
        self,
        results: dict[str, Any],
    ) -> bool:
        """Return whether sequential execution should stop.

        The default implementation never stops. Subclasses can override this
        method to implement benchmark-specific stopping conditions.

        Args:
            results:
                Result dictionary returned by the completed child benchmark.

        Returns:
            ``True`` if no further runs should be executed.
        """
        return False

    def run(
        self,
        device=None,
        num_shots: int = 1024,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Execute and analyse child stages lazily in configured order."""
        if self.run_records:
            raise MeasurementOutcomesExistError()

        self._runtime_params = {
            "num_shots": num_shots,
            "device": device,
        } | kwargs
        self.run_index = 0
        self.benchmarks = []

        for run_index in range(len(self.parameter_sequence)):
            self.run_index = run_index
            parameters = self._get_parameters(run_index)

            benchmark = self._prepared_stages.get(run_index)
            if benchmark is None:
                benchmark, _ = self._create_benchmark(run_index)
                self._prepared_stages[run_index] = benchmark

            # Child run lazily generates circuits if this stage was not prepared.
            benchmark.run(
                device=device,
                num_shots=num_shots,
                **kwargs,
            )
            results = benchmark.analyze()

            if not isinstance(results, dict):
                raise TypeError(
                    f"{self.benchmark_class.__name__} returned "
                    f"{type(results).__name__}; sequential benchmark results "
                    "must be dictionaries"
                )

            stop = bool(self.should_stop(results))
            self.benchmarks.append(benchmark)
            self.run_records.append(
                {
                    "run_index": run_index,
                    "sequence_parameters": dict(
                        self.parameter_sequence[run_index]
                    ),
                    "resolved_parameters": parameters,
                    "result": results,
                    "stopping_condition_met": stop,
                }
            )

            if stop:
                print(
                    "The stopping condition was met on run "
                    f"{run_index} with parameters {parameters}."
                )
                break

        self.run_index = len(self.run_records)
        self._experiment_data = DataFrame(self.run_records)
        return [record["result"] for record in self.run_records]

    def reset(self, clear_prepared: bool = True) -> None:
        """Reset sequential state before a new execution."""
        if not clear_prepared and any(
            child._experiment_data is not None
            and "circuit_measurements" in child._experiment_data.columns
            for child in self._prepared_stages.values()
        ):
            raise MeasurementOutcomesExistError()

        self.run_index = 0
        self.benchmarks = []
        self.run_records = []
        self._experiment_data = None
        if clear_prepared:
            self._prepared_stages = {}
