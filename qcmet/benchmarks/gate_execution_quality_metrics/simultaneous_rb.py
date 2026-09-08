"""Simultaneous Randomised Benchmarking crosstalk (addressability) metric.

The metric quantifies how much the error rate of a set of qubits degrades
when neighbouring qubits are driven at the same time, and whether the
additional error is correlated between subsets.
It therefore measures crosstalk.

The protocol follows Phys. Rev. Lett. 109, 240504 (2012),
generalised from single qubits to arbitrary disjoint qubit subsets.

Three families of numbers are produced:

- the isolated error per Clifford of each subset, obtained from a standard
  Clifford RB decay measured while every other subset idles;
- the simultaneous error per Clifford of each subset, obtained from the same
  decay measured while every other subset is driven with its own independent
  random Clifford sequence;
- a correlation parameter for each pair of subsets, obtained from the decay of
  the joint Pauli-Z parity observable, which vanishes identically when the
  noise acting on the two subsets is a tensor product.
"""

from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING, Any, Dict, List, Sequence, Tuple

if TYPE_CHECKING:
    from pathlib import Path

    from qcmet.core import FileManager

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Clifford, random_clifford
from scipy.optimize import curve_fit

from qcmet.benchmarks import BaseBenchmark


class SimultaneousRB(BaseBenchmark):
    """Implements the simultaneous randomised benchmarking crosstalk metric.

    The register is partitioned into disjoint subsets. For every sequence
    length ``m`` and every random seed the class emits

    - one *simultaneous* circuit in which each subset executes its own
      independent length-``m`` random Clifford sequence in parallel, and
    - one *isolated* circuit per subset, in which only that subset executes its
      sequence and every other subset idles.

    The isolated and simultaneous circuits for a given seed reuse the *same*
    random Clifford sequences, so the only difference between them is the
    presence of gates on the spectator subsets. This common-random-number
    pairing removes the sampling noise that would otherwise dominate the
    difference of two independently estimated error rates.

    All circuits measure the whole register, so readout is identical in the
    isolated and simultaneous cases and the metric isolates gate-induced
    crosstalk rather than differences in readout conditions.

    Attributes:
        labels (List[str]): One label per subset, derived from the physical
            qubit indices, e.g. ``"q0"`` or ``"q2_3"``.

    """

    def __init__(
        self,
        m_list: List[int],
        subsets: Sequence[Sequence[int]],
        circs_per_m: int = 10,
        seed: int | None = None,
        save_path: str | Path | FileManager | None = None,
    ):
        """Initialize the simultaneous randomised benchmark.

        Args:
            m_list (List[int]): The list of sequence lengths to run the benchmark on.
            subsets (Sequence[Sequence[int]]): Disjoint groups of physical qubit
                indices that are benchmarked against each other, for example
                ``[[0], [1]]`` for two single qubits or ``[[0, 1], [2, 3]]`` for
                two qubit pairs. At least two subsets must be supplied.
            circs_per_m (int): The number of random seeds generated for a given
                sequence length m. Each seed produces one simultaneous circuit and
                one isolated circuit per subset.
            seed (int, optional): Seed for the random Clifford sampling, used to make
                circuit generation reproducible. Defaults to None.
            save_path (str | Path | FileManager | None, optional): Directory path to
                save results. Defaults to None.

        Raises:
            ValueError: If fewer than two subsets are given, if a subset is empty,
                if the subsets are not disjoint, or if a qubit index is not a
                non-negative int.

        """
        subsets = [list(subset) for subset in subsets]

        if len(subsets) < 2:
            raise ValueError(
                "At least two qubit subsets are required to measure crosstalk."
            )
        flat: List[int] = []
        for subset in subsets:
            if len(subset) == 0:
                raise ValueError("Qubit subsets must not be empty.")
            if not all(isinstance(qb, (int, np.integer)) for qb in subset):
                raise ValueError("Qubits indices must be int")
            if any(qb < 0 for qb in subset):
                raise ValueError("Invalid number of qubits specified")
            flat.extend(int(qb) for qb in subset)
        if len(set(flat)) != len(flat):
            raise ValueError("Qubit subsets must be disjoint.")

        super().__init__("SimultaneousRB", qubits=flat, save_path=save_path)

        self.config["m_list"] = sorted(int(m) for m in m_list)
        self.config["circs_per_m"] = circs_per_m
        self.config["subsets"] = subsets
        self.config["seed"] = seed

        self.labels: List[str] = ["q" + "_".join(str(qb) for qb in s) for s in subsets]
        self.config["subset_labels"] = self.labels

        # Position of each subset inside the compact register handed back to
        # BaseBenchmark.generate_circuits, which routes compact index i onto the
        # physical qubit self.qubits[i].
        self._positions: List[List[int]] = []
        offset = 0
        for subset in subsets:
            self._positions.append(list(range(offset, offset + len(subset))))
            offset += len(subset)

        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # circuit generation
    # ------------------------------------------------------------------

    @property
    def subsets(self) -> List[List[int]]:
        """Physical qubit indices of each benchmarked subset.

        Returns:
            List[List[int]]: The disjoint qubit subsets.

        """
        return self.config["subsets"]

    @property
    def pairs(self) -> List[Tuple[int, int]]:
        """Index pairs of subsets for which a correlation parameter is computed.

        Returns:
            List[Tuple[int, int]]: All unordered pairs of subset indices.

        """
        return list(combinations(range(len(self.subsets)), 2))

    def _random_sequence(self, num_qubits: int, m: int) -> Dict[str, Any]:
        """Draw a random Clifford sequence together with its inverting Clifford.

        Elements are sampled uniformly from the ``num_qubits``-qubit Clifford
        group, which is a unitary 2-design and therefore twirls the noise into a
        depolarizing channel, giving a single exponential decay.

        Args:
            num_qubits (int): Width of the subset the sequence acts on.
            m (int): Number of random Clifford elements in the sequence.

        Returns:
            Dict[str, Any]: Dictionary with key 'layers', the list of m Clifford
            element circuits, and key 'inverse', the circuit implementing the
            inverse of their product.

        """
        total = Clifford(QuantumCircuit(num_qubits))
        layers = []
        for _ in range(m):
            element = random_clifford(num_qubits, seed=self._rng)
            layers.append(element.to_circuit())
            total = total.compose(element)
        return {"layers": layers, "inverse": total.adjoint().to_circuit()}

    def _build_circuit(
        self, sequences: List[Dict[str, Any]], active: Sequence[int], m: int
    ) -> QuantumCircuit:
        """Assemble one benchmarking circuit on the compact register.

        Args:
            sequences (List[Dict[str, Any]]): One random sequence per subset, as
                returned by ``_random_sequence``.
            active (Sequence[int]): Indices of the subsets that are driven. Subsets
                that are not listed idle for the whole circuit.
            m (int): Sequence length.

        Returns:
            QuantumCircuit: Circuit of ``self.num_qubits`` qubits, layer separated
            by barriers, ending in a measurement of the whole register.

        """
        q_reg = QuantumRegister(self.num_qubits, name="q")
        circ = QuantumCircuit(q_reg)
        for layer in range(m):
            for k in active:
                circ.compose(
                    sequences[k]["layers"][layer],
                    qubits=self._positions[k],
                    inplace=True,
                )
            circ.barrier()
        for k in active:
            circ.compose(
                sequences[k]["inverse"], qubits=self._positions[k], inplace=True
            )
        circ.measure_all()
        return circ

    def _generate_circuits(self):
        """Generate the isolated and simultaneous randomised benchmarking circuits.

        For each sequence length m and each of ``circs_per_m`` seeds:
            1. Draw an independent random Clifford sequence of length m for every
               subset, together with its inverting Clifford.
            2. Build one circuit in which all subsets run their sequence in parallel.
            3. Build one circuit per subset in which only that subset runs its
               sequence, reusing the sequence drawn in step 1.

        Returns:
            List[Dict]: Each dict contains:
                'circuit' (QuantumCircuit): The benchmark circuit.
                'm' (int): The sequence length.
                'mode' (str): Either 'isolated' or 'simultaneous'.
                'active' (str): Subset label being driven, or 'all'.
                'seq_id' (int): Index of the random seed, shared by the isolated
                    and simultaneous circuits built from the same sequences.

        """
        data = []
        all_subsets = range(len(self.subsets))
        for m in self.config["m_list"]:
            for seq_id in range(self.config["circs_per_m"]):
                sequences = [
                    self._random_sequence(len(subset), m) for subset in self.subsets
                ]

                data.append(
                    self._circ_with_metadata_dict(
                        self._build_circuit(sequences, all_subsets, m),
                        m=m,
                        mode="simultaneous",
                        active="all",
                        seq_id=seq_id,
                    )
                )
                for k in all_subsets:
                    data.append(
                        self._circ_with_metadata_dict(
                            self._build_circuit(sequences, [k], m),
                            m=m,
                            mode="isolated",
                            active=self.labels[k],
                            seq_id=seq_id,
                        )
                    )
        return data

    # ------------------------------------------------------------------
    # estimators and fitting
    # ------------------------------------------------------------------

    @staticmethod
    def fit_func(m, alpha, a0, b0):
        """Exponential decay fit function.

        Args:
            m (series): sequence length.
            alpha (float): decay fitting parameter.
            a0 (float): amplitude fitting parameter.
            b0 (float): baseline fitting parameter.

        Returns:
            ndarray: fitting function datapoints.

        """
        return a0 * alpha**m + b0

    @staticmethod
    def _survival(probabilities: Dict[str, float], positions: Sequence[int]) -> float:
        """Marginal probability that the given qubits all return outcome zero.

        Args:
            probabilities (Dict[str, float]): Outcome probabilities keyed by
                bitstring, where character i corresponds to qubit i.
            positions (Sequence[int]): Compact indices of the subset.

        Returns:
            float: Marginal survival probability of the subset.

        """
        total = 0.0
        for bits, prob in probabilities.items():
            bits = bits.replace(" ", "")
            if all(bits[i] == "0" for i in positions):
                total += prob
        return total

    @staticmethod
    def _parity(probabilities: Dict[str, float], positions: Sequence[int]) -> float:
        """Estimate the expectation value of the Pauli-Z parity over the given qubits.

        This estimates the Pauli transfer matrix eigenvalue of the tensor
        product of Z operators on ``positions``. Under a tensor product noise
        channel the eigenvalue of a joint observable factorises into the product
        of the eigenvalues of its parts, which is what the correlation parameter
        tests for.

        Args:
            probabilities (Dict[str, float]): Outcome probabilities keyed by
                bitstring, where character i corresponds to qubit i.
            positions (Sequence[int]): Compact indices to take the parity over.

        Returns:
            float: Expectation value in [-1, 1].

        """
        total = 0.0
        for bits, prob in probabilities.items():
            bits = bits.replace(" ", "")
            weight = sum(int(bits[i]) for i in positions)
            total += prob * (-1.0) ** weight
        return total

    def _fit_decay(self, frame, column, floor: float):
        """Average an estimator over seeds and fit an exponential decay to it.

        Args:
            frame (pandas.DataFrame): Rows of experiment_data to fit.
            column (str): Column holding the per-circuit estimator.
            floor (float): Expected asymptotic value of the estimator, used as the
                initial guess for the baseline and to bound it from below.

        Returns:
            Dict[str, Any]: Dictionary with the sequence lengths, the averaged
            estimator, the fitted parameters, the covariance matrix, the decay
            parameter and its standard error.

        """
        averaged = frame.groupby("m")[column].mean().sort_index()
        m_values = np.asarray(averaged.index, dtype=float)
        y_values = np.asarray(averaged.to_list(), dtype=float)

        lower = (0.0, 0.0, min(floor, 0.0) - 0.1)
        upper = (1.0, 1.0 - floor + 1e-9, floor + 0.1)
        p0 = (0.99, max(float(y_values[0]) - floor, 1e-3), floor)
        fitted_parameters, parameter_covariance = curve_fit(
            self.fit_func,
            m_values,
            y_values,
            p0=np.clip(p0, lower, upper),
            bounds=(lower, upper),
            maxfev=20000,
        )
        errors = np.sqrt(np.abs(np.diag(parameter_covariance)))
        return {
            "m": m_values,
            "mean": y_values,
            "fitted_parameters": fitted_parameters,
            "parameter_covariance": parameter_covariance,
            "alpha": float(fitted_parameters[0]),
            "alpha_stderr": float(errors[0]),
        }

    def _epc(self, alpha: float, num_qubits: int) -> float:
        """Convert a decay parameter into an error per Clifford.

        Args:
            alpha (float): Fitted decay parameter.
            num_qubits (int): Width of the benchmarked subset.

        Returns:
            float: Error per Clifford, (d - 1)(1 - alpha) / d with d = 2**num_qubits.

        """
        d = 2**num_qubits
        return (d - 1) * (1 - alpha) / d

    # ------------------------------------------------------------------
    # analysis
    # ------------------------------------------------------------------

    def _add_estimator_columns(self) -> None:
        """Add per-subset survival and parity columns to experiment_data."""
        self.measurements_to_probabilities()
        probabilities = self._experiment_data["meas_prob"]

        for label, positions in zip(self.labels, self._positions, strict=True):
            self._experiment_data[f"p_surv_{label}"] = probabilities.apply(
                lambda x, pos=positions: self._survival(x, pos)
            )
            self._experiment_data[f"parity_{label}"] = probabilities.apply(
                lambda x, pos=positions: self._parity(x, pos)
            )
        for i, j in self.pairs:
            joint = self._positions[i] + self._positions[j]
            key = f"parity_{self.labels[i]}|{self.labels[j]}"
            self._experiment_data[key] = probabilities.apply(
                lambda x, pos=joint: self._parity(x, pos)
            )

    def _analyze(self):
        """Analyze measurements to obtain the crosstalk metrics.

        Survival probabilities are taken on each subset and fitted
        separately for the isolated and the simultaneous circuits, giving an
        error per Clifford for each case. The difference gives the
        addressability error of that subset. Pauli-Z parity decays measured on
        the simultaneous circuits give the correlation parameter of each pair of
        subsets.

        Returns:
            dict: {
              'qubits': List[int],
              'subsets': {label: List[int]},
              'IsolatedEPC': {label: float},
              'SimultaneousEPC': {label: float},
              'AddressabilityError': {label: float},
              'AddressabilityRatio': {label: float},
              'CorrelationParameter': {'labelA|labelB': float},
              'alpha': {'isolated': ..., 'simultaneous': ..., 'parity': ...},
              'uncertainties': {...},
              'fit_result': {...}
            }

        """
        self._add_estimator_columns()
        data = self._experiment_data

        self.fits: Dict[str, Dict[str, Any]] = {
            "isolated": {},
            "simultaneous": {},
            "parity": {},
        }

        isolated_epc: Dict[str, float] = {}
        simultaneous_epc: Dict[str, float] = {}
        addressability: Dict[str, float] = {}
        ratio: Dict[str, float] = {}
        alpha_iso: Dict[str, float] = {}
        alpha_sim: Dict[str, float] = {}
        alpha_parity: Dict[str, float] = {}
        errors: Dict[str, Dict[str, float]] = {
            "IsolatedEPC": {},
            "SimultaneousEPC": {},
            "AddressabilityError": {},
            "CorrelationParameter": {},
        }

        simultaneous_rows = data[data["mode"] == "simultaneous"]

        for label, subset in zip(self.labels, self.subsets, strict=True):
            width = len(subset)
            floor = 1.0 / 2**width

            iso_rows = data[(data["mode"] == "isolated") & (data["active"] == label)]
            iso_fit = self._fit_decay(iso_rows, f"p_surv_{label}", floor)
            sim_fit = self._fit_decay(simultaneous_rows, f"p_surv_{label}", floor)
            par_fit = self._fit_decay(simultaneous_rows, f"parity_{label}", 0.0)

            self.fits["isolated"][label] = iso_fit
            self.fits["simultaneous"][label] = sim_fit
            self.fits["parity"][label] = par_fit

            alpha_iso[label] = iso_fit["alpha"]
            alpha_sim[label] = sim_fit["alpha"]
            alpha_parity[label] = par_fit["alpha"]

            isolated_epc[label] = self._epc(iso_fit["alpha"], width)
            simultaneous_epc[label] = self._epc(sim_fit["alpha"], width)
            addressability[label] = simultaneous_epc[label] - isolated_epc[label]
            ratio[label] = (
                simultaneous_epc[label] / isolated_epc[label]
                if isolated_epc[label] > 0
                else float("nan")
            )

            scale = (2**width - 1) / 2**width
            errors["IsolatedEPC"][label] = scale * iso_fit["alpha_stderr"]
            errors["SimultaneousEPC"][label] = scale * sim_fit["alpha_stderr"]
            errors["AddressabilityError"][label] = scale * float(
                np.hypot(iso_fit["alpha_stderr"], sim_fit["alpha_stderr"])
            )

        correlation: Dict[str, float] = {}
        for i, j in self.pairs:
            label_i, label_j = self.labels[i], self.labels[j]
            key = f"{label_i}|{label_j}"
            joint_fit = self._fit_decay(simultaneous_rows, f"parity_{key}", 0.0)
            self.fits["parity"][key] = joint_fit

            a_i = alpha_parity[label_i]
            a_j = alpha_parity[label_j]
            correlation[key] = joint_fit["alpha"] - a_i * a_j
            alpha_parity[key] = joint_fit["alpha"]

            errors["CorrelationParameter"][key] = float(
                np.sqrt(
                    joint_fit["alpha_stderr"] ** 2
                    + (a_j * self.fits["parity"][label_i]["alpha_stderr"]) ** 2
                    + (a_i * self.fits["parity"][label_j]["alpha_stderr"]) ** 2
                )
            )

        self.run_id = self.file_manager.run_id if self.file_manager else None

        return {
            "qubits": list(self.qubits),
            "subsets": dict(zip(self.labels, self.subsets, strict=True)),
            "IsolatedEPC": isolated_epc,
            "SimultaneousEPC": simultaneous_epc,
            "AddressabilityError": addressability,
            "AddressabilityRatio": ratio,
            "CorrelationParameter": correlation,
            "alpha": {
                "isolated": alpha_iso,
                "simultaneous": alpha_sim,
                "parity": alpha_parity,
            },
            "uncertainties": errors,
            "fit_result": {
                context: {
                    label: {
                        "fitted_parameters": fit["fitted_parameters"],
                        "parameter_covariance": fit["parameter_covariance"],
                    }
                    for label, fit in fits.items()
                }
                for context, fits in self.fits.items()
            },
        }

    def _plot(self, axes):
        """Plot isolated and simultaneous survival decays for every subset.

        Isolated decays are drawn with open circles and dashed fits, simultaneous
        decays with crosses and solid fits. A visible gap between the two curves
        for a subset is the signature of crosstalk.

        Args:
            axes (matplotlib.axes.Axes): Axes to draw the plots on.

        Returns:
            matplotlib.legend.Legend: Legend for the plot.

        """
        m_max = max(self.config["m_list"])
        fit_xxs = np.linspace(0, m_max, 500)
        colours = [f"C{i}" for i in range(len(self.labels))]

        for label, colour in zip(self.labels, colours, strict=True):
            for context, marker, style in (
                ("isolated", "o", "--"),
                ("simultaneous", "x", "-"),
            ):
                fit = self.fits[context][label]
                axes.plot(
                    fit["m"],
                    fit["mean"],
                    linestyle="",
                    marker=marker,
                    markerfacecolor="none",
                    c=colour,
                    label=f"{label} {context}",
                )
                axes.plot(
                    fit_xxs,
                    self.fit_func(fit_xxs, *fit["fitted_parameters"]),
                    linestyle=style,
                    c=colour,
                )

        smallest = min(len(subset) for subset in self.subsets)
        axes.set_xlim((0, m_max))
        axes.set_ylim((1 / 2**smallest - 0.05, 1.02))
        axes.set_xlabel(r"$m$")
        axes.set_ylabel(r"$p_0$")

        return axes.legend()
