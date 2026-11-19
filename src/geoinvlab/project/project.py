"""
GeoInvLab3D project management.

This module provides the high-level Project object used by the GUI.

Supported physical methods:
    - Gravity
    - Magnetic
    - Magnetotelluric (MT)

The inversion interface is compatible with:
    optimizer.solve(problem)

MT data convention:
    Each row represents one station-frequency observation and contains
    eight scalar values:

        rho_xx, rho_xy, rho_yx, rho_yy,
        phase_xx, phase_xy, phase_yx, phase_yy

    Therefore, N MT observations correspond to 8N scalar data values.
    
     MIT License Copyright (c) 17/09/2026 [Boukar Abdelghafour]
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import inspect
import json
import time

import numpy as np

from geoinvlab.core.data.dataset import DataSet
from geoinvlab.core.data.mt_dataset import MTDataSet
from geoinvlab.core.mesh import StructuredMesh3D
from geoinvlab.core.model import Model3D


@dataclass
class Project:
    """High-level GeoInvLab3D project."""

    name: str = "Untitled"

    method: object | None = None
    dataset: DataSet | MTDataSet | None = None
    mesh: StructuredMesh3D | None = None
    model: Model3D | None = None
    operator: object | None = None
    inverse_problem: object | None = None
    result: Model3D | None = None

    def __post_init__(self) -> None:
        self.name = str(self.name)

        # Keep the public dataclass field contract stable.
        # The optional reference model is managed as a runtime attribute.
        self.reference_model: Model3D | None = None

        self.inversion_history: list[dict] = []
        self.inversion_statistics: dict = {}

        self.inversion_config: dict = {
            "method": "Default",
            "lambda": 0.0,
            "max_iterations": 20,
            "tolerance": 1e-6,
            "inclination": 0.0,
            "declination": 0.0,
            "field_intensity": 50000.0,
            "mt_parameterization": "resistivity",
            "mt_solver_method": "direct",
            "mt_solver_tolerance": 1e-10,
            "mt_solver_maxiter": 1000,
        }

    @staticmethod
    def _meshes_compatible(
        first: StructuredMesh3D,
        second: StructuredMesh3D,
    ) -> bool:
        """Return whether two structured meshes have identical geometry."""
        if not isinstance(first, StructuredMesh3D) or not isinstance(
            second, StructuredMesh3D
        ):
            return False

        integer_attributes = (
            "nx", "ny", "nz", "padding_x", "padding_y", "padding_z"
        )
        if any(
            getattr(first, key) != getattr(second, key)
            for key in integer_attributes
        ):
            return False

        float_attributes = ("dx", "dy", "dz", "padding_factor")
        if any(
            not np.isclose(
                float(getattr(first, key)),
                float(getattr(second, key)),
                rtol=1e-12,
                atol=1e-12,
            )
            for key in float_attributes
        ):
            return False

        return bool(
            np.allclose(
                np.asarray(first.origin, dtype=float),
                np.asarray(second.origin, dtype=float),
                rtol=1e-12,
                atol=1e-12,
            )
        )

    @staticmethod
    def _mesh_to_dict(mesh: StructuredMesh3D) -> dict:
        """Serialize all structured-mesh geometry parameters."""
        return {
            "nx": int(mesh.nx),
            "ny": int(mesh.ny),
            "nz": int(mesh.nz),
            "dx": float(mesh.dx),
            "dy": float(mesh.dy),
            "dz": float(mesh.dz),
            "origin": [float(value) for value in mesh.origin],
            "padding_x": [int(value) for value in mesh.padding_x],
            "padding_y": [int(value) for value in mesh.padding_y],
            "padding_z": [int(value) for value in mesh.padding_z],
            "padding_factor": float(mesh.padding_factor),
        }

    @staticmethod
    def _model_to_dict(model: Model3D | None) -> dict | None:
        """Serialize a model without duplicating its mesh geometry."""
        if not isinstance(model, Model3D):
            return None
        return {
            "name": model.name,
            "values": np.asarray(model.values, dtype=float).tolist(),
        }

    @staticmethod
    def _model_npz_metadata(model: Model3D) -> dict:
        """Return mesh metadata shared by NPZ model exports."""
        return {
            "nx": np.asarray(model.mesh.nx),
            "ny": np.asarray(model.mesh.ny),
            "nz": np.asarray(model.mesh.nz),
            "dx": np.asarray(model.mesh.dx),
            "dy": np.asarray(model.mesh.dy),
            "dz": np.asarray(model.mesh.dz),
            "origin": np.asarray(model.mesh.origin, dtype=float),
            "padding_x": np.asarray(model.mesh.padding_x, dtype=int),
            "padding_y": np.asarray(model.mesh.padding_y, dtype=int),
            "padding_z": np.asarray(model.mesh.padding_z, dtype=int),
            "padding_factor": np.asarray(model.mesh.padding_factor),
            "name": np.asarray(model.name),
        }

    # ==============================================================
    # DATA
    # ==============================================================

    def set_data(
        self,
        dataset: DataSet | MTDataSet,
    ) -> None:
        """Set a Gravity, Magnetic or MT dataset."""

        if not isinstance(dataset, (DataSet, MTDataSet)):
            raise TypeError(
                "dataset must be a DataSet or MTDataSet instance."
            )

        self.dataset = dataset
        self.result = None
        self.inverse_problem = None
        self.operator = None
        self.method = None

        self.inversion_history = []
        self.inversion_statistics = {}

    def clear_data(self) -> None:
        """Remove the project dataset."""

        self.dataset = None
        self.result = None
        self.operator = None
        self.method = None
        self.inverse_problem = None

        self.inversion_history = []
        self.inversion_statistics = {}

    # ==============================================================
    # MESH
    # ==============================================================

    def set_mesh(self, mesh: StructuredMesh3D) -> None:
        """Set the project mesh."""

        if not isinstance(mesh, StructuredMesh3D):
            raise TypeError(
                "mesh must be a StructuredMesh3D instance."
            )

        self.mesh = mesh

        self.model = None
        self.reference_model = None
        self.result = None
        self.operator = None
        self.method = None
        self.inverse_problem = None

        self.inversion_history = []
        self.inversion_statistics = {}

    def clear_mesh(self) -> None:
        """Remove the mesh and associated models."""

        self.mesh = None
        self.model = None
        self.reference_model = None
        self.result = None
        self.operator = None
        self.method = None
        self.inverse_problem = None

        self.inversion_history = []
        self.inversion_statistics = {}

    # ==============================================================
    # MODEL
    # ==============================================================

    def set_model(self, model: Model3D) -> None:
        """Set the initial model after checking mesh compatibility."""
        if not isinstance(model, Model3D):
            raise TypeError("model must be a Model3D instance.")
        if self.mesh is None:
            raise RuntimeError("Set the project mesh before setting a model.")
        if not self._meshes_compatible(model.mesh, self.mesh):
            raise ValueError(
                "Model mesh geometry is incompatible with the project's mesh."
            )

        # Rebind to the project's mesh instance so all project components
        # share one canonical mesh object.
        self.model = Model3D(
            mesh=self.mesh,
            values=np.asarray(model.values, dtype=float).copy(),
            name=model.name,
        )
        self.result = None
        self.operator = None
        self.method = None
        self.inverse_problem = None
        self.inversion_history = []
        self.inversion_statistics = {}

    def clear_model(self) -> None:
        """Remove the current initial model and associated inversion state."""
        self.model = None
        self.result = None
        self.operator = None
        self.method = None
        self.inverse_problem = None
        self.inversion_history = []
        self.inversion_statistics = {}

    def set_reference_model(self, model: Model3D) -> None:
        """Set a reference model after checking mesh compatibility."""
        if not isinstance(model, Model3D):
            raise TypeError("model must be a Model3D instance.")
        if self.mesh is None:
            raise RuntimeError(
                "Set the project mesh before setting a reference model."
            )
        if not self._meshes_compatible(model.mesh, self.mesh):
            raise ValueError(
                "Reference-model mesh geometry is incompatible with "
                "the project's mesh."
            )

        self.reference_model = Model3D(
            mesh=self.mesh,
            values=np.asarray(model.values, dtype=float).copy(),
            name=model.name,
        )
        self.result = None
        self.inverse_problem = None
        self.operator = None
        self.method = None
        self.inversion_history = []
        self.inversion_statistics = {}

    def clear_reference_model(self) -> None:
        """Remove the reference model."""
        self.reference_model = None
        self.result = None
        self.inverse_problem = None
        self.operator = None
        self.method = None
        self.inversion_history = []
        self.inversion_statistics = {}

    # ==============================================================
    # PROJECT STATE
    # ==============================================================

    @property
    def n_data(self) -> int:
        """
        Number of scalar data values.

        For MT, this is 8 times the number of station-frequency rows.
        """

        if self.dataset is None:
            return 0

        return int(self.dataset.n_data)

    @property
    def n_observations(self) -> int:
        """Number of observation rows."""

        if self.dataset is None:
            return 0

        if isinstance(self.dataset, MTDataSet):
            return int(self.dataset.n_observations)

        return int(self.dataset.n_data)

    @property
    def n_cells(self) -> int:
        """Number of mesh cells."""

        if self.mesh is None:
            return 0

        return int(self.mesh.n_cells)

    @property
    def n_model_parameters(self) -> int:
        """Number of model parameters."""

        if self.model is None:
            return 0

        return int(self.model.n_parameters)

    @property
    def is_mt(self) -> bool:
        """Return True when the project contains MT data."""

        return isinstance(self.dataset, MTDataSet)

    @property
    def is_ready_for_inversion(self) -> bool:
        """Check whether the project has all required components."""

        return (
            self.dataset is not None
            and self.mesh is not None
            and self.model is not None
        )

    # ==============================================================
    # INVERSION CONFIGURATION
    # ==============================================================

    def configure_inversion(
        self,
        method: str = "Default",
        regularization_parameter: float = 0.0,
        max_iterations: int = 20,
        tolerance: float = 1e-6,
        inclination: float = 0.0,
        declination: float = 0.0,
        field_intensity: float = 50000.0,
        mt_parameterization: str = "resistivity",
        mt_solver_method: str = "direct",
        mt_solver_tolerance: float = 1e-10,
        mt_solver_maxiter: int = 1000,
    ) -> None:
        """Configure Gravity, Magnetic or MT inversion."""

        method_identifier = str(method).lower().strip()

        supported_methods = {
            "",
            "default",
            "gravity",
            "grav",
            "magnetic",
            "mag",
            "magnetics",
            "mt",
            "magnetotelluric",
        }

        if method_identifier not in supported_methods:
            raise ValueError(
                f"Unsupported inversion method: '{method}'. "
                "Supported methods: Gravity, Magnetic and MT."
            )

        if regularization_parameter < 0.0:
            raise ValueError(
                "Regularization parameter must be non-negative."
            )

        if max_iterations < 1:
            raise ValueError(
                "max_iterations must be at least 1."
            )

        if tolerance <= 0.0 or not np.isfinite(tolerance):
            raise ValueError(
                "tolerance must be finite and strictly positive."
            )

        if not np.isfinite(inclination):
            raise ValueError("inclination must be finite.")

        if not np.isfinite(declination):
            raise ValueError("declination must be finite.")

        if not np.isfinite(field_intensity) or field_intensity < 0.0:
            raise ValueError(
                "field_intensity must be finite and non-negative."
            )

        mt_parameterization = str(mt_parameterization).lower().strip()

        if mt_parameterization not in {
            "resistivity",
            "conductivity",
        }:
            raise ValueError(
                "mt_parameterization must be 'resistivity' "
                "or 'conductivity'."
            )

        if mt_solver_tolerance <= 0.0:
            raise ValueError(
                "mt_solver_tolerance must be strictly positive."
            )

        if mt_solver_maxiter < 1:
            raise ValueError(
                "mt_solver_maxiter must be at least 1."
            )

        self.inversion_config = {
            "method": str(method),
            "lambda": float(regularization_parameter),
            "max_iterations": int(max_iterations),
            "tolerance": float(tolerance),
            "inclination": float(inclination),
            "declination": float(declination),
            "field_intensity": float(field_intensity),
            "mt_parameterization": mt_parameterization,
            "mt_solver_method": str(mt_solver_method),
            "mt_solver_tolerance": float(mt_solver_tolerance),
            "mt_solver_maxiter": int(mt_solver_maxiter),
        }

    # ==============================================================
    # PRINT HELPERS
    # ==============================================================

    @staticmethod
    def _print_separator(
        character: str = "-",
        length: int = 72,
    ) -> None:
        print(character * length)

    @staticmethod
    def _print_header(title: str) -> None:
        print()
        print("=" * 72)
        print(title)
        print("=" * 72)

    # ==============================================================
    # STATISTICS
    # ==============================================================

    @staticmethod
    def _model_statistics(model: Model3D) -> dict:
        values = np.asarray(model.values, dtype=float).reshape(-1)

        if values.size == 0:
            return {
                "minimum": 0.0,
                "maximum": 0.0,
                "mean": 0.0,
                "median": 0.0,
                "std": 0.0,
                "norm": 0.0,
                "l2_squared": 0.0,
            }

        return {
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "std": float(np.std(values)),
            "norm": float(np.linalg.norm(values)),
            "l2_squared": float(np.sum(values ** 2)),
        }

    def _data_statistics(self) -> dict:
        if self.dataset is None:
            return {}

        values = np.asarray(
            self.dataset.values,
            dtype=float,
        ).reshape(-1)

        if values.size == 0:
            return {
                "minimum": 0.0,
                "maximum": 0.0,
                "mean": 0.0,
                "std": 0.0,
                "norm": 0.0,
            }

        return {
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "norm": float(np.linalg.norm(values)),
        }

    # ==============================================================
    # FORWARD RESIDUALS
    # ==============================================================

    def _residual(
        self,
        predicted: np.ndarray,
        observed: np.ndarray,
        mt_method=None,
        weighted: bool = False,
    ) -> np.ndarray:
        """
        Calculate residuals.

        MT phase residuals are wrapped to [-180, 180) degrees.
        Other methods use ordinary subtraction.
        """

        predicted = np.asarray(
            predicted,
            dtype=float,
        ).reshape(-1)

        observed = np.asarray(
            observed,
            dtype=float,
        ).reshape(-1)

        if predicted.shape != observed.shape:
            raise ValueError(
                "Predicted and observed vectors must have identical sizes."
            )

        if mt_method is not None:
            return mt_method.residual(
                predicted,
                weighted=weighted,
            )

        residual = predicted - observed

        if weighted:
            weights = np.asarray(
                self.dataset.weights(),
                dtype=float,
            ).reshape(-1)

            if weights.shape != residual.shape:
                raise ValueError(
                    "Dataset weights have an invalid shape."
                )

            residual *= weights

        return residual

    # ==============================================================
    # INVERSION
    # ==============================================================

    def run_inversion(
        self,
        config: dict | None = None,
    ) -> Model3D:
        """Run the configured geophysical inversion."""

        if self.dataset is None:
            raise RuntimeError("No dataset is available.")

        if self.mesh is None:
            raise RuntimeError("No mesh is available.")

        if self.model is None:
            raise RuntimeError("No initial model is available.")

        # ----------------------------------------------------------
        # Configuration
        # ----------------------------------------------------------

        current_config = dict(self.inversion_config)

        if config is not None:
            current_config.update(dict(config))

        method_name = str(
            current_config.get("method", "Default")
        )

        method_identifier = method_name.lower().strip()

        gravity_identifiers = {
            "",
            "default",
            "gravity",
            "grav",
        }

        magnetic_identifiers = {
            "magnetic",
            "mag",
            "magnetics",
        }

        mt_identifiers = {
            "mt",
            "magnetotelluric",
        }

        if method_identifier in gravity_identifiers:
            physical_method = "gravity"

        elif method_identifier in magnetic_identifiers:
            physical_method = "magnetic"

        elif method_identifier in mt_identifiers:
            physical_method = "mt"

        else:
            raise ValueError(
                f"Unsupported inversion method: '{method_name}'. "
                "Supported methods: Gravity, Magnetic and MT."
            )

        if physical_method == "mt" and not isinstance(
            self.dataset,
            MTDataSet,
        ):
            raise TypeError(
                "MT inversion requires an MTDataSet. "
                "Load an MT dataset before selecting the MT method."
            )

        if physical_method != "mt" and isinstance(
            self.dataset,
            MTDataSet,
        ):
            raise TypeError(
                "The selected method is not MT, but the project "
                "contains an MTDataSet."
            )

        regularization_parameter = float(
            current_config.get("lambda", 0.0)
        )

        max_iterations = int(
            current_config.get("max_iterations", 20)
        )

        tolerance = float(
            current_config.get("tolerance", 1e-6)
        )

        inclination = float(
            current_config.get("inclination", 0.0)
        )

        declination = float(
            current_config.get("declination", 0.0)
        )

        field_intensity = float(
            current_config.get("field_intensity", 50000.0)
        )

        if (
            not np.isfinite(regularization_parameter)
            or regularization_parameter < 0.0
        ):
            raise ValueError(
                "Regularization parameter must be finite "
                "and non-negative."
            )

        if max_iterations < 1:
            raise ValueError("max_iterations must be at least 1.")

        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError(
                "tolerance must be finite and strictly positive."
            )

        if not np.isfinite(inclination):
            raise ValueError("inclination must be finite.")

        if not np.isfinite(declination):
            raise ValueError("declination must be finite.")

        if not np.isfinite(field_intensity) or field_intensity < 0.0:
            raise ValueError(
                "field_intensity must be finite and non-negative."
            )

        self.inversion_config = current_config
        self.inversion_history = []
        self.inversion_statistics = {}

        # ----------------------------------------------------------
        # Header
        # ----------------------------------------------------------

        self._print_header("GeoInvLab3D - INVERSION")

        print(f"Project                : {self.name}")
        print(f"Method                 : {physical_method.upper()}")

        if physical_method == "mt":
            print(
                f"Station-frequency rows : "
                f"{self.dataset.n_observations}"
            )
            print(
                f"Scalar MT data         : "
                f"{self.dataset.n_data}"
            )
            print(
                f"Unique frequencies     : "
                f"{self.dataset.n_frequencies}"
            )
            print(
                f"Unique stations        : "
                f"{len(np.unique(self.dataset.locations, axis=0))}"
            )
        else:
            print(f"Observations           : {self.dataset.n_data}")

        print(f"Model parameters       : {self.model.n_parameters}")
        print(
            f"Mesh                   : "
            f"{self.mesh.nx} x {self.mesh.ny} x {self.mesh.nz}"
        )
        print(
            f"Cell size              : "
            f"{self.mesh.dx} x {self.mesh.dy} x {self.mesh.dz}"
        )
        print(f"Regularization lambda  : {regularization_parameter:.6g}")
        print(f"Maximum iterations     : {max_iterations}")
        print(f"Tolerance              : {tolerance:.6g}")

        if physical_method == "magnetic":
            print(f"Magnetic inclination   : {inclination:.6g} deg")
            print(f"Magnetic declination   : {declination:.6g} deg")
            print(f"Magnetic field         : {field_intensity:.6g} nT")

        if physical_method == "mt":
            print(
                "MT parameterization    : "
                f"{current_config.get('mt_parameterization', 'resistivity')}"
            )

        # ----------------------------------------------------------
        # Statistics
        # ----------------------------------------------------------

        self._print_separator()
        print("OBSERVED DATA")

        data_statistics = self._data_statistics()

        for key, label in (
            ("minimum", "Minimum"),
            ("maximum", "Maximum"),
            ("mean", "Mean"),
            ("std", "Standard deviation"),
            ("norm", "L2 norm"),
        ):
            if key in data_statistics:
                print(f"{label:22s}: {data_statistics[key]:.6e}")

        initial_statistics = self._model_statistics(self.model)

        print()
        print("INITIAL MODEL")

        for key, label in (
            ("minimum", "Minimum"),
            ("maximum", "Maximum"),
            ("mean", "Mean"),
            ("std", "Standard deviation"),
            ("norm", "L2 norm"),
        ):
            print(f"{label:22s}: {initial_statistics[key]:.6e}")

        # ----------------------------------------------------------
        # Inversion components
        # ----------------------------------------------------------

        print()
        print("[1/5] Loading inversion components...")

        from geoinvlab.core.inversion.problem import InverseProblem
        from geoinvlab.core.inversion.optimizer import Optimizer
        from geoinvlab.core.inversion.regularization import L2Regularization

        print("      Inversion components loaded.")

        # ----------------------------------------------------------
        # Build forward operator
        # ----------------------------------------------------------

        print()
        print("[2/5] Building forward operator...")

        mt_method = None

        if physical_method == "gravity":
            from geoinvlab.methods.gravity.operator import GravityOperator

            operator = GravityOperator(
                observation_locations=self.dataset.locations
            )

            print("      Gravity operator initialized.")

        elif physical_method == "magnetic":
            from geoinvlab.methods.magnetic.operator import MagneticOperator

            operator = MagneticOperator(
                observation_locations=self.dataset.locations,
                inclination=inclination,
                declination=declination,
                field_intensity=field_intensity,
            )

            print("      Magnetic operator initialized.")

        else:
            from geoinvlab.methods.mt.method import MTMethod

            mt_method = MTMethod(
                dataset=self.dataset,
                mesh=self.mesh,
            )

            operator = mt_method.operator

            # MTOperator.predict(model) must return one response row
            # for each row in MTDataSet, in the original dataset order.
            #
            # MTMethod should initialize MTOperator.locations with
            # dataset.locations (not only unique_locations).
            operator.locations = np.asarray(
                self.dataset.locations,
                dtype=float,
            ).copy()

            operator.observation_frequencies = np.asarray(
                self.dataset.frequencies,
                dtype=float,
            ).copy()

            operator.frequencies = np.unique(
                np.asarray(self.dataset.frequencies, dtype=float)
            )

            print("      MT method initialized.")
            print("      MT forward operator initialized.")

        self.method = mt_method
        self.operator = operator

        # ----------------------------------------------------------
        # Regularization
        # ----------------------------------------------------------

        regularization = None

        if regularization_parameter > 0.0:
            regularization = L2Regularization()
            print()
            print("      L2 Tikhonov regularization enabled.")
            print(
                f"      lambda = {regularization_parameter:.6e}"
            )
        else:
            print()
            print("      Regularization disabled.")

        # ----------------------------------------------------------
        # Inverse problem
        # ----------------------------------------------------------

        print()
        print("[3/5] Building inverse problem...")

        problem = InverseProblem(
            model=self.model,
            data=self.dataset,
            operator=operator,
            regularization=regularization,
            regularization_parameter=regularization_parameter,
        )

        self.inverse_problem = problem

        print(f"      Number of scalar data : {problem.n_data}")
        print(f"      Number of parameters  : {problem.n_parameters}")
        print(f"      Regularized           : {problem.is_regularized}")

        # ----------------------------------------------------------
        # Initial prediction
        # ----------------------------------------------------------

        print()
        print("      Calculating initial prediction...")

        initial_prediction = np.asarray(
            operator.predict(self.model),
            dtype=float,
        ).reshape(-1)

        observed_values = np.asarray(
            self.dataset.values,
            dtype=float,
        ).reshape(-1)

        if initial_prediction.shape != observed_values.shape:
            raise ValueError(
                "Invalid initial prediction shape: "
                f"{initial_prediction.shape}; "
                f"expected {observed_values.shape}."
            )

        if not np.all(np.isfinite(initial_prediction)):
            raise ValueError(
                "Initial prediction contains NaN or infinite values."
            )

        initial_residual = self._residual(
            initial_prediction,
            observed_values,
            mt_method=mt_method,
            weighted=False,
        )

        weights = np.asarray(
            self.dataset.weights(),
            dtype=float,
        ).reshape(-1)

        if weights.shape != observed_values.shape:
            raise ValueError(
                "Dataset weights have an invalid shape: "
                f"{weights.shape}; expected {observed_values.shape}."
            )

        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError(
                "Dataset weights must be finite and non-negative."
            )

        initial_weighted_residual = initial_residual * weights

        initial_data_misfit = float(
            np.dot(
                initial_weighted_residual,
                initial_weighted_residual,
            )
        )

        initial_rms = float(
            np.sqrt(np.mean(initial_residual ** 2))
        )

        initial_weighted_rms = float(
            np.sqrt(np.mean(initial_weighted_residual ** 2))
        )

        try:
            initial_objective = float(problem.objective_value())
        except Exception:
            initial_objective = initial_data_misfit

        print(f"      Initial data misfit : {initial_data_misfit:.6e}")
        print(f"      Initial RMS         : {initial_rms:.6e}")
        print(
            f"      Initial weighted RMS: {initial_weighted_rms:.6e}"
        )
        print(f"      Initial objective   : {initial_objective:.6e}")

        self.inversion_history.append(
            {
                "iteration": 0,
                "objective": initial_objective,
                "data_misfit": initial_data_misfit,
                "rms": initial_rms,
                "weighted_rms": initial_weighted_rms,
            }
        )

        # ----------------------------------------------------------
        # Optimizer
        # ----------------------------------------------------------

        print()
        print("[4/5] Initializing optimizer...")

        try:
            signature = inspect.signature(Optimizer)
            parameters = signature.parameters

            optimizer_kwargs = {}

            if "max_iterations" in parameters:
                optimizer_kwargs["max_iterations"] = max_iterations

            if "tolerance" in parameters:
                optimizer_kwargs["tolerance"] = tolerance

            optimizer = Optimizer(**optimizer_kwargs)

        except Exception as exc:
            raise RuntimeError(
                "Could not initialize the inversion optimizer."
            ) from exc

        print("      Optimizer initialized.")

        # ----------------------------------------------------------
        # Solve
        # ----------------------------------------------------------

        print()
        print("[5/5] RUNNING INVERSION")

        self._print_separator("=", 72)

        start_time = time.perf_counter()

        try:
            result = optimizer.solve(problem)

        except Exception as exc:
            elapsed = time.perf_counter() - start_time

            print()
            print("INVERSION FAILED")
            print(f"Elapsed time          : {elapsed:.3f} s")
            print(f"{type(exc).__name__}: {exc}")
            self._print_separator("=", 72)

            raise

        elapsed = time.perf_counter() - start_time

        # ----------------------------------------------------------
        # Validate recovered model
        # ----------------------------------------------------------

        if not isinstance(result, Model3D):
            raise TypeError(
                "The optimizer must return a Model3D instance."
            )

        result_values = np.asarray(
            result.values,
            dtype=float,
        ).reshape(-1)

        if result_values.size != self.model.n_parameters:
            raise ValueError(
                "Invalid recovered model size: "
                f"{result_values.size}; "
                f"{self.model.n_parameters} expected."
            )

        if not np.all(np.isfinite(result_values)):
            raise ValueError(
                "Recovered model contains NaN or infinite values."
            )

        # ----------------------------------------------------------
        # Recovered model prediction
        # ----------------------------------------------------------

        recovered_prediction = np.asarray(
            operator.predict(result),
            dtype=float,
        ).reshape(-1)

        if recovered_prediction.shape != observed_values.shape:
            raise ValueError(
                "Invalid recovered prediction shape: "
                f"{recovered_prediction.shape}; "
                f"{observed_values.shape} expected."
            )

        if not np.all(np.isfinite(recovered_prediction)):
            raise ValueError(
                "Recovered prediction contains NaN or infinite values."
            )

        recovered_residual = self._residual(
            recovered_prediction,
            observed_values,
            mt_method=mt_method,
            weighted=False,
        )

        recovered_weighted_residual = recovered_residual * weights

        final_data_misfit = float(
            np.dot(
                recovered_weighted_residual,
                recovered_weighted_residual,
            )
        )

        final_rms = float(
            np.sqrt(np.mean(recovered_residual ** 2))
        )

        final_weighted_rms = float(
            np.sqrt(np.mean(recovered_weighted_residual ** 2))
        )

        try:
            final_objective = float(
                problem.objective_value_for(result)
            )
        except Exception:
            final_objective = final_data_misfit

        self.inversion_history.append(
            {
                "iteration": -1,
                "objective": final_objective,
                "data_misfit": final_data_misfit,
                "rms": final_rms,
                "weighted_rms": final_weighted_rms,
            }
        )

        # ----------------------------------------------------------
        # Statistics
        # ----------------------------------------------------------

        recovered_statistics = self._model_statistics(result)

        objective_improvement = (
            initial_objective - final_objective
        )

        if initial_data_misfit != 0.0:
            misfit_reduction_percent = (
                100.0
                * (initial_data_misfit - final_data_misfit)
                / initial_data_misfit
            )
        else:
            misfit_reduction_percent = 0.0

        self.inversion_statistics = {
            "elapsed_seconds": float(elapsed),
            "initial_objective": float(initial_objective),
            "final_objective": float(final_objective),
            "objective_improvement": float(objective_improvement),
            "initial_data_misfit": float(initial_data_misfit),
            "final_data_misfit": float(final_data_misfit),
            "misfit_reduction_percent": float(
                misfit_reduction_percent
            ),
            "initial_rms": float(initial_rms),
            "final_rms": float(final_rms),
            "initial_weighted_rms": float(initial_weighted_rms),
            "final_weighted_rms": float(final_weighted_rms),
            "n_data": int(self.dataset.n_data),
            "n_observations": int(self.n_observations),
            "n_parameters": int(self.model.n_parameters),
            "lambda": float(regularization_parameter),
            "method": physical_method,
            "model": recovered_statistics,
        }

        self.result = result

        # ----------------------------------------------------------
        # Print results
        # ----------------------------------------------------------

        print()
        self._print_separator("=", 72)

        print("INVERSION COMPLETED")

        self._print_separator()

        print(f"Elapsed time          : {elapsed:.3f} s")

        print()
        print("OBJECTIVE")
        print(f"Initial               : {initial_objective:.6e}")
        print(f"Final                 : {final_objective:.6e}")
        print(f"Improvement           : {objective_improvement:.6e}")

        print()
        print("DATA MISFIT")
        print(f"Initial               : {initial_data_misfit:.6e}")
        print(f"Final                 : {final_data_misfit:.6e}")
        print(
            f"Reduction             : "
            f"{misfit_reduction_percent:.3f} %"
        )

        print()
        print("RMS")
        print(f"Initial               : {initial_rms:.6e}")
        print(f"Final                 : {final_rms:.6e}")

        print()
        print("REGULARIZATION")
        print(f"Lambda                : {regularization_parameter:.6e}")
        print(f"Active                : {problem.is_regularized}")

        print()
        print("RECOVERED MODEL")

        for key, label in (
            ("minimum", "Minimum"),
            ("maximum", "Maximum"),
            ("mean", "Mean"),
            ("median", "Median"),
            ("std", "Standard deviation"),
            ("norm", "L2 norm"),
        ):
            print(f"{label:22s}: {recovered_statistics[key]:.6e}")

        print(f"Number of parameters  : {result.n_parameters}")

        self._print_separator("=", 72)

        return result

    # ==============================================================
    # SAVE RECOVERED MODEL
    # ==============================================================

    def save_recovered_model(
        self,
        filename: str,
        save_npy: bool = True,
        save_npz: bool = True,
    ) -> dict:
        """Save the recovered model in NPY and/or NPZ format."""

        if not isinstance(self.result, Model3D):
            raise RuntimeError(
                "No recovered model is available. Run the inversion first."
            )

        model = self.result
        values = np.asarray(model.values, dtype=float)

        if not np.all(np.isfinite(values)):
            raise ValueError(
                "Recovered model contains NaN or infinite values."
            )

        path = Path(filename)

        if path.exists() and path.is_dir():
            base = path / "recovered_model"
        elif path.suffix.lower() in {".npy", ".npz"}:
            base = path.with_suffix("")
        else:
            base = path

        base.parent.mkdir(parents=True, exist_ok=True)

        saved_files = {}

        if save_npy:
            npy_path = base.with_suffix(".npy")
            np.save(npy_path, values)
            saved_files["npy"] = str(npy_path)

        if save_npz:
            npz_path = base.with_suffix(".npz")

            np.savez(
                npz_path,
                values=values,
                **self._model_npz_metadata(model),
            )

            saved_files["npz"] = str(npz_path)

        return saved_files

    # ==============================================================
    # SAVE ALL MODELS
    # ==============================================================

    def save_all_models(self, directory: str) -> dict:
        """Save initial and recovered models."""

        output_directory = Path(directory)
        output_directory.mkdir(parents=True, exist_ok=True)

        saved = {}

        if isinstance(self.model, Model3D):
            values = np.asarray(self.model.values, dtype=float)

            initial_npy = output_directory / "initial_model.npy"
            initial_npz = output_directory / "initial_model.npz"

            np.save(initial_npy, values)

            np.savez(
                initial_npz,
                values=values,
                **self._model_npz_metadata(self.model),
            )

            saved["initial_npy"] = str(initial_npy)
            saved["initial_npz"] = str(initial_npz)

        if isinstance(self.reference_model, Model3D):
            reference_values = np.asarray(
                self.reference_model.values, dtype=float
            )
            reference_npy = output_directory / "reference_model.npy"
            reference_npz = output_directory / "reference_model.npz"
            np.save(reference_npy, reference_values)
            np.savez(
                reference_npz,
                values=reference_values,
                **self._model_npz_metadata(self.reference_model),
            )
            saved["reference_npy"] = str(reference_npy)
            saved["reference_npz"] = str(reference_npz)

        if isinstance(self.result, Model3D):
            recovered = self.save_recovered_model(
                str(output_directory)
            )

            saved.update({
                "recovered_" + key: value
                for key, value in recovered.items()
            })

        return saved

    # ==============================================================
    # SAVE INVERSION REPORT
    # ==============================================================

    def save_inversion_report(self, filename: str) -> str:
        """Save inversion diagnostics as JSON."""

        path = Path(filename)

        if path.suffix.lower() != ".json":
            path = path.with_suffix(".json")

        path.parent.mkdir(parents=True, exist_ok=True)

        report = {
            "project": self.name,
            "configuration": dict(self.inversion_config),
            "statistics": dict(self.inversion_statistics),
            "history": list(self.inversion_history),
        }

        with path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)

        return str(path)

    # ==============================================================
    # DATASET SERIALIZATION
    # ==============================================================

    def _serialize_dataset(self) -> dict | None:
        """Serialize DataSet or MTDataSet without losing its structure."""

        if self.dataset is None:
            return None

        if isinstance(self.dataset, MTDataSet):
            return {
                "type": "mt",
                "name": self.dataset.name,
                "locations": self.dataset.locations.tolist(),
                "frequencies": self.dataset.frequencies.tolist(),
                "rho_xx": self.dataset.rho_xx.tolist(),
                "rho_xy": self.dataset.rho_xy.tolist(),
                "rho_yx": self.dataset.rho_yx.tolist(),
                "rho_yy": self.dataset.rho_yy.tolist(),
                "phase_xx": self.dataset.phase_xx.tolist(),
                "phase_xy": self.dataset.phase_xy.tolist(),
                "phase_yx": self.dataset.phase_yx.tolist(),
                "phase_yy": self.dataset.phase_yy.tolist(),
                "uncertainties": (
                    None
                    if self.dataset.uncertainties is None
                    else self.dataset.uncertainties.tolist()
                ),
            }

        return {
            "type": "standard",
            "name": self.dataset.name,
            "units": self.dataset.units,
            "values": self.dataset.values.tolist(),
            "locations": self.dataset.locations.tolist(),
            "uncertainties": (
                None
                if self.dataset.uncertainties is None
                else self.dataset.uncertainties.tolist()
            ),
        }

    @staticmethod
    def _deserialize_dataset(dataset_data: dict):
        """Reconstruct a standard or MT dataset."""

        dataset_type = dataset_data.get("type", "standard")

        if dataset_type == "mt":
            return MTDataSet(
                locations=np.asarray(
                    dataset_data["locations"],
                    dtype=float,
                ),
                frequencies=np.asarray(
                    dataset_data["frequencies"],
                    dtype=float,
                ),
                rho_xx=np.asarray(dataset_data["rho_xx"], dtype=float),
                rho_xy=np.asarray(dataset_data["rho_xy"], dtype=float),
                rho_yx=np.asarray(dataset_data["rho_yx"], dtype=float),
                rho_yy=np.asarray(dataset_data["rho_yy"], dtype=float),
                phase_xx=np.asarray(dataset_data["phase_xx"], dtype=float),
                phase_xy=np.asarray(dataset_data["phase_xy"], dtype=float),
                phase_yx=np.asarray(dataset_data["phase_yx"], dtype=float),
                phase_yy=np.asarray(dataset_data["phase_yy"], dtype=float),
                uncertainties=(
                    None
                    if dataset_data.get("uncertainties") is None
                    else np.asarray(
                        dataset_data["uncertainties"],
                        dtype=float,
                    )
                ),
                name=dataset_data.get("name", "MT data"),
            )

        # Backward compatibility with older project files.
        return DataSet(
            values=np.asarray(dataset_data["values"], dtype=float),
            locations=np.asarray(dataset_data["locations"], dtype=float),
            uncertainties=(
                None
                if dataset_data.get("uncertainties") is None
                else np.asarray(
                    dataset_data["uncertainties"],
                    dtype=float,
                )
            ),
            name=dataset_data.get("name", "data"),
            units=dataset_data.get("units", ""),
        )

    # ==============================================================
    # SAVE PROJECT
    # ==============================================================

    def save(self, filename: str) -> str:
        """Save the complete project to a JSON file."""

        path = Path(filename)

        if path.suffix.lower() != ".json":
            path = path.with_suffix(".json")

        data = {
            "name": self.name,
            "inversion_config": dict(self.inversion_config),
            "inversion_statistics": dict(self.inversion_statistics),
            "inversion_history": list(self.inversion_history),
            "dataset": self._serialize_dataset(),
        }

        data["mesh"] = (
            self._mesh_to_dict(self.mesh)
            if self.mesh is not None
            else None
        )
        data["model"] = self._model_to_dict(self.model)
        data["reference_model"] = self._model_to_dict(self.reference_model)
        data["result"] = self._model_to_dict(self.result)

        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)

        return str(path)

    # ==============================================================
    # LOAD PROJECT
    # ==============================================================

    @classmethod
    def load(cls, filename: str) -> "Project":
        """Load a project from a JSON file."""

        path = Path(filename)

        if not path.exists():
            raise FileNotFoundError(
                f"Project file does not exist: {path}"
            )

        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        project = cls(
            name=data.get("name", "Untitled")
        )

        # Configuration
        config = data.get("inversion_config")

        if isinstance(config, dict):
            defaults = dict(project.inversion_config)
            defaults.update(config)
            project.inversion_config = defaults

        # Statistics and history
        statistics = data.get("inversion_statistics")
        if isinstance(statistics, dict):
            project.inversion_statistics = statistics

        history = data.get("inversion_history")
        if isinstance(history, list):
            project.inversion_history = history

        # Dataset
        dataset_data = data.get("dataset")

        if dataset_data is not None:
            project.dataset = cls._deserialize_dataset(dataset_data)

        # Mesh
        mesh_data = data.get("mesh")

        if mesh_data is not None:
            project.mesh = StructuredMesh3D(
                nx=int(mesh_data["nx"]),
                ny=int(mesh_data["ny"]),
                nz=int(mesh_data["nz"]),
                dx=float(mesh_data["dx"]),
                dy=float(mesh_data["dy"]),
                dz=float(mesh_data["dz"]),
                origin=tuple(
                    mesh_data.get("origin", (0.0, 0.0, 0.0))
                ),
                padding_x=tuple(mesh_data.get("padding_x", (0, 0))),
                padding_y=tuple(mesh_data.get("padding_y", (0, 0))),
                padding_z=tuple(mesh_data.get("padding_z", (0, 0))),
                padding_factor=float(mesh_data.get("padding_factor", 1.0)),
            )

        # Initial, reference and recovered models all use the same mesh.
        for key, attribute, default_name in (
            ("model", "model", "model"),
            ("reference_model", "reference_model", "reference model"),
            ("result", "result", "result"),
        ):
            model_data = data.get(key)
            if model_data is not None and project.mesh is not None:
                setattr(
                    project,
                    attribute,
                    Model3D(
                        mesh=project.mesh,
                        values=np.asarray(model_data["values"], dtype=float),
                        name=model_data.get("name", default_name),
                    ),
                )

        return project


__all__ = ["Project"]
