"""
Inverse problem definition for GeoInvLab3D.

MIT License Copyright (c) 01/10/2026 [Boukar Abdelghafour]

An inverse problem combines:
    - a physical model,
    - observed data,
    - a forward operator,
    - an optional regularization term.

The data and operator are validated through their public interfaces,
allowing specialized containers such as MTDataSet.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from geoinvlab.core.data import DataSet
from geoinvlab.core.model import Model3D
from geoinvlab.core.operators import ForwardOperator
from geoinvlab.core.inversion.objective import LeastSquaresObjective
from geoinvlab.core.inversion.regularization import Regularization


@dataclass
class InverseProblem:
    """
    Definition of a numerical inverse problem.

    Parameters
    ----------
    model:
        Initial or reference physical model.

    data:
        Observed data container. Must provide n_data, values,
        and weights().

    operator:
        Forward operator providing predict(model) and jacobian(model).

    regularization:
        Optional regularization object.

    regularization_parameter:
        Non-negative regularization weight.

    Notes
    -----
    n_data always denotes the number of scalar values used by
    the inversion, not necessarily the number of physical observations.

    For MTDataSet:

        n_observations = N
        n_data = n_values = 8 * N
    """

    model: Model3D
    data: DataSet
    operator: ForwardOperator
    regularization: Regularization | None = None
    regularization_parameter: float = 0.0

    objective: LeastSquaresObjective = field(init=False)

    def __post_init__(self) -> None:
        """Validate the inverse problem and construct its objective."""
        self._validate_regularization_parameter()
        self._validate_model()
        self._validate_data_interface()
        self._validate_operator_interface()
        self._validate_operator_data_compatibility()

        self.objective = LeastSquaresObjective(
            operator=self.operator,
            data=self.data,
            regularization=self.regularization,
            regularization_parameter=self.regularization_parameter,
        )

    # ==================================================================
    # Scalar data count
    # ==================================================================

    @property
    def n_data(self) -> int:
        """Number of scalar data values used by the inversion."""
        return int(self.data.n_data)

    @property
    def n_parameters(self) -> int:
        """Number of model parameters."""
        return int(self.model.n_parameters)

    @property
    def is_regularized(self) -> bool:
        """Whether regularization is active."""
        return (
            self.regularization is not None
            and self.regularization_parameter > 0.0
        )

    # ==================================================================
    # Validation
    # ==================================================================

    def _validate_regularization_parameter(self) -> None:
        """Validate the regularization parameter."""
        try:
            value = float(self.regularization_parameter)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "Regularization parameter must be a real number."
            ) from exc

        if not np.isfinite(value):
            raise ValueError(
                "Regularization parameter must be finite."
            )

        if value < 0.0:
            raise ValueError(
                "Regularization parameter must be non-negative."
            )

        self.regularization_parameter = value

    def _validate_model(self) -> None:
        """Validate the model parameter vector."""
        if not isinstance(self.model, Model3D):
            raise TypeError(
                "model must be a Model3D instance."
            )

        try:
            n_parameters = int(self.model.n_parameters)
            values = np.asarray(self.model.values, dtype=float)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "The model must expose numerical parameters."
            ) from exc

        if n_parameters <= 0:
            raise ValueError(
                "The model must contain at least one parameter."
            )

        if values.ndim != 1:
            raise ValueError(
                "model.values must be one-dimensional."
            )

        if values.size != n_parameters:
            raise ValueError(
                "Model parameter count is inconsistent: "
                f"n_parameters={n_parameters}, "
                f"values.size={values.size}."
            )

        if not np.all(np.isfinite(values)):
            raise ValueError(
                "model.values must contain only finite values."
            )

    def _validate_data_interface(self) -> None:
        """
        Validate the generic data interface.

        The data count is the size of the flattened scalar vector.
        For MTDataSet, each physical observation contributes eight
        scalar values.
        """
        if not hasattr(self.data, "n_data"):
            raise TypeError(
                "data must provide an 'n_data' attribute."
            )

        if not hasattr(self.data, "values"):
            raise TypeError(
                "data must provide a 'values' attribute."
            )

        weights_method = getattr(
            self.data,
            "weights",
            None,
        )

        if not callable(weights_method):
            raise TypeError(
                "data must provide a callable weights() method."
            )

        try:
            n_data = int(self.data.n_data)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "data.n_data must be an integer."
            ) from exc

        if n_data <= 0:
            raise ValueError(
                "The data set must contain at least one scalar datum."
            )

        try:
            values = np.asarray(
                self.data.values,
                dtype=float,
            )
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "data.values must be a numerical array."
            ) from exc

        values = values.reshape(-1)

        if values.size != n_data:
            raise ValueError(
                "Data vector dimension mismatch: "
                f"data.n_data={n_data}, "
                f"data.values.size={values.size}."
            )

        if not np.all(np.isfinite(values)):
            raise ValueError(
                "data.values must contain only finite values."
            )

        try:
            weights = np.asarray(
                weights_method(),
                dtype=float,
            ).reshape(-1)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "data.weights() must return numerical weights."
            ) from exc

        if weights.size != n_data:
            raise ValueError(
                "Weight vector dimension mismatch: "
                f"expected {n_data} weights, "
                f"got {weights.size}."
            )

        if not np.all(np.isfinite(weights)):
            raise ValueError(
                "data.weights() must contain only finite values."
            )

        if np.any(weights < 0.0):
            raise ValueError(
                "data.weights() must be non-negative."
            )

    def _validate_operator_interface(self) -> None:
        """Validate the forward operator interface."""
        predict = getattr(
            self.operator,
            "predict",
            None,
        )

        if not callable(predict):
            raise TypeError(
                "operator must provide predict(model)."
            )

        jacobian = getattr(
            self.operator,
            "jacobian",
            None,
        )

        if not callable(jacobian):
            raise TypeError(
                "operator must provide jacobian(model)."
            )

    def _validate_operator_data_compatibility(self) -> None:
        """
        Validate operator/data compatibility.

        Specialized operators can validate the model/data pair with
        check_prediction(model, data). Generic operators can expose
        n_data instead.
        """
        check_prediction = getattr(
            self.operator,
            "check_prediction",
            None,
        )

        if callable(check_prediction):
            check_prediction(
                self.model,
                self.data,
            )
            return

        operator_n_data = getattr(
            self.operator,
            "n_data",
            None,
        )

        if operator_n_data is not None:
            try:
                operator_n_data = int(operator_n_data)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    "operator.n_data must be an integer."
                ) from exc

            if operator_n_data != self.n_data:
                raise ValueError(
                    "Operator/data size mismatch: "
                    f"operator.n_data={operator_n_data}, "
                    f"data.n_data={self.n_data}."
                )

        # Validate the actual prediction when possible.
        # This also catches operators that return a wrong-sized vector
        # without exposing an n_data attribute.
        predicted = np.asarray(
            self.operator.predict(self.model),
            dtype=float,
        ).reshape(-1)

        if predicted.size != self.n_data:
            raise ValueError(
                "Prediction/data size mismatch: "
                f"operator returned {predicted.size} scalar values, "
                f"but the data set contains {self.n_data}."
            )

        if not np.all(np.isfinite(predicted)):
            raise ValueError(
                "The forward operator returned non-finite predictions."
            )

    # ==================================================================
    # Objective evaluation
    # ==================================================================

    def objective_value(self) -> float:
        """Evaluate the total objective for the current model."""
        return float(self.objective.value(self.model))

    def data_misfit(self) -> float:
        """Evaluate the data-misfit contribution."""
        return float(self.objective.data_misfit(self.model))

    def regularization_value(self) -> float:
        """Evaluate regularization before scaling."""
        return float(self.objective.regularization_value(self.model))

    # ==================================================================
    # Residuals
    # ==================================================================

    def residual(self) -> np.ndarray:
        """Return the unweighted residual."""
        return self.objective.residual(self.model)

    def weighted_residual(self) -> np.ndarray:
        """Return the weighted residual."""
        return self.objective.weighted_residual(self.model)

    # ==================================================================
    # Evaluation for arbitrary models
    # ==================================================================

    def objective_value_for(
        self,
        model: Model3D,
    ) -> float:
        """Evaluate the total objective for another model."""
        return float(self.objective.value(model))

    def data_misfit_for(
        self,
        model: Model3D,
    ) -> float:
        """Evaluate data misfit for another model."""
        return float(self.objective.data_misfit(model))

    def regularization_value_for(
        self,
        model: Model3D,
    ) -> float:
        """Evaluate regularization for another model."""
        return float(self.objective.regularization_value(model))

    def residual_for(
        self,
        model: Model3D,
    ) -> np.ndarray:
        """Return the unweighted residual for another model."""
        return self.objective.residual(model)

    def weighted_residual_for(
        self,
        model: Model3D,
    ) -> np.ndarray:
        """Return the weighted residual for another model."""
        return self.objective.weighted_residual(model)
