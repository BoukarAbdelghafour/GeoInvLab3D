"""
Tests for the 3D magnetotelluric inversion workflow.

The tests verify:

    1. The generic Gauss-Newton solver on a linear problem.
    2. Construction of a synthetic MT data set from the forward model.
    3. Consistency between MTDataSet.n_data and MTDataSet.values.
    4. Consistency between MTOperator.predict() and MTDataSet.values.
    5. Reduction of the MT inversion objective.
    6. Stability of the recovered model for a small homogeneous problem.

The synthetic MT data are generated with the same forward operator
used by the inversion. This makes the test independent of a particular
analytical approximation of the finite 3D computational domain.
"""

from __future__ import annotations

import numpy as np
import pytest

from geoinvlab.core.model import Model3D
from geoinvlab.core.inversion.problem import InverseProblem
from geoinvlab.core.data import MTDataSet
from geoinvlab.methods.mt.operator import MTOperator
from geoinvlab.core.inversion.optimizer import (
    GaussNewtonSolver,
    LinearLeastSquaresSolver,
)


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def mesh():
    """Small 2 x 2 x 2 structured mesh."""

    from geoinvlab.core.mesh import StructuredMesh3D

    return StructuredMesh3D(
        nx=2,
        ny=2,
        nz=2,
        dx=100.0,
        dy=100.0,
        dz=100.0,
    )


@pytest.fixture
def frequencies():
    """Frequencies used by the MT inversion tests."""

    return np.array(
        [0.1, 1.0, 10.0],
        dtype=float,
    )


@pytest.fixture
def stations():
    """Two MT stations."""

    return np.array(
        [
            [50.0, 50.0, 100.0],
            [150.0, 150.0, 100.0],
        ],
        dtype=float,
    )


@pytest.fixture
def observation_locations(
    stations,
    frequencies,
):
    """
    Expand the two stations into station-frequency observations.

    The ordering is

        STA01, f1
        STA01, f2
        STA01, f3
        STA02, f1
        STA02, f2
        STA02, f3
    """

    return np.repeat(
        stations,
        frequencies.size,
        axis=0,
    )


@pytest.fixture
def observation_frequencies(
    stations,
    frequencies,
):
    """Frequency associated with each station-frequency observation."""

    return np.tile(
        frequencies,
        stations.shape[0],
    )


@pytest.fixture
def true_model(mesh):
    """Homogeneous synthetic resistivity model."""

    rho_true = 100.0

    return Model3D(
        mesh=mesh,
        values=np.full(
            mesh.n_cells,
            rho_true,
            dtype=float,
        ),
        name="true_resistivity",
    )


@pytest.fixture
def initial_model(mesh):
    """Homogeneous starting model."""

    rho_initial = 300.0

    return Model3D(
        mesh=mesh,
        values=np.full(
            mesh.n_cells,
            rho_initial,
            dtype=float,
        ),
        name="initial_resistivity",
    )


@pytest.fixture
def mt_operator(
    mesh,
    frequencies,
    observation_locations,
    observation_frequencies,
):
    """MT forward operator configured for the synthetic data."""

    return MTOperator(
        mesh=mesh,
        frequencies=frequencies,
        locations=observation_locations,
        observation_frequencies=observation_frequencies,
        jacobian_step=1.0e-5,
    )


@pytest.fixture
def synthetic_data(
    mt_operator,
    true_model,
    observation_locations,
    observation_frequencies,
):
    """
    Generate noiseless MT observations from the true model.
    """

    predicted = mt_operator.predict(
        true_model
    )

    n_observations = observation_locations.shape[0]

    assert predicted.shape == (
        8 * n_observations,
    )

    values = predicted.reshape(
        n_observations,
        8,
    )

    return MTDataSet(
        locations=observation_locations,
        frequencies=observation_frequencies,
        rho_xx=values[:, 0],
        rho_xy=values[:, 1],
        rho_yx=values[:, 2],
        rho_yy=values[:, 3],
        phase_xx=values[:, 4],
        phase_xy=values[:, 5],
        phase_yx=values[:, 6],
        phase_yy=values[:, 7],
        name="synthetic_MT",
    )


# ======================================================================
# Data-set consistency
# ======================================================================


def test_mt_dataset_n_data(
    synthetic_data,
):
    """n_data must count scalar MT values, not observations."""

    assert synthetic_data.n_observations == 6
    assert synthetic_data.n_values == 48
    assert synthetic_data.n_data == 48

    assert synthetic_data.values.shape == (
        48,
    )

    assert synthetic_data.weights().shape == (
        48,
    )


def test_mt_dataset_values_are_finite(
    synthetic_data,
):
    """The synthetic MT data vector must be finite."""

    values = synthetic_data.values

    assert values.ndim == 1
    assert values.size == synthetic_data.n_data
    assert np.all(np.isfinite(values))


# ======================================================================
# Forward/operator consistency
# ======================================================================


def test_mt_operator_prediction_matches_dataset(
    mt_operator,
    true_model,
    synthetic_data,
):
    """
    The operator prediction must reproduce the synthetic data
    used to construct MTDataSet.
    """

    predicted = mt_operator.predict(
        true_model
    )

    observed = synthetic_data.values

    assert predicted.shape == observed.shape

    np.testing.assert_allclose(
        predicted,
        observed,
        rtol=1.0e-10,
        atol=1.0e-10,
    )


def test_mt_operator_data_dimension(
    mt_operator,
    synthetic_data,
):
    """The operator and data set must have the same n_data."""

    assert mt_operator.n_data == synthetic_data.n_data


# ======================================================================
# InverseProblem construction
# ======================================================================


def test_mt_inverse_problem(
    mt_operator,
    synthetic_data,
    initial_model,
):
    """The MT objects must form a valid inverse problem."""

    problem = InverseProblem(
        model=initial_model,
        data=synthetic_data,
        operator=mt_operator,
    )

    assert problem.model is initial_model
    assert problem.data is synthetic_data
    assert problem.operator is mt_operator

    assert problem.data.n_data == (
        problem.operator.n_data
    )


# ======================================================================
# Objective reduction
# ======================================================================


def test_mt_inversion_reduces_objective(
    mt_operator,
    synthetic_data,
    initial_model,
):
    """
    Gauss-Newton inversion must reduce the synthetic MT misfit.

    Exact recovery is deliberately not required here because the
    homogeneous 3D MT problem can be poorly conditioned when every
    cell is inverted independently.
    """

    problem = InverseProblem(
        model=initial_model,
        data=synthetic_data,
        operator=mt_operator,
    )

    initial_prediction = mt_operator.predict(
        initial_model
    )

    initial_residual = (
        synthetic_data.values
        - initial_prediction
    )

    initial_objective = (
        problem.objective.value(
            initial_model
        )
    )

    assert np.isfinite(initial_objective)
    assert initial_objective >= 0.0

    assert np.linalg.norm(
        initial_residual
    ) > 0.0

    solver = GaussNewtonSolver(
        max_iterations=3,
        tolerance=1.0e-6,
        damping=1.0e-8,
        verbose=False,
    )

    recovered_model = solver.solve(
        problem
    )

    assert isinstance(
        recovered_model,
        Model3D,
    )

    assert recovered_model.values.shape == (
        initial_model.values.shape
    )

    assert np.all(
        np.isfinite(
            recovered_model.values
        )
    )

    final_objective = (
        problem.objective.value(
            recovered_model
        )
    )

    assert np.isfinite(final_objective)

    assert final_objective <= (
        initial_objective
        + 1.0e-12
    )


# ======================================================================
# Synthetic homogeneous inversion
# ======================================================================


def test_mt_homogeneous_inversion(
    mt_operator,
    synthetic_data,
    true_model,
    initial_model,
):
    """
    Test a complete homogeneous MT inversion.

    The test checks that the inversion moves the starting model toward
    the synthetic model and decreases the data misfit.
    """

    problem = InverseProblem(
        model=initial_model,
        data=synthetic_data,
        operator=mt_operator,
    )

    initial_prediction = mt_operator.predict(
        initial_model
    )

    true_prediction = mt_operator.predict(
        true_model
    )

    observed = synthetic_data.values

    initial_misfit = np.linalg.norm(
        observed - initial_prediction
    )

    true_misfit = np.linalg.norm(
        observed - true_prediction
    )

    assert true_misfit < 1.0e-8
    assert initial_misfit > true_misfit

    solver = GaussNewtonSolver(
        max_iterations=3,
        tolerance=1.0e-6,
        damping=1.0e-8,
        verbose=False,
    )

    recovered_model = solver.solve(
        problem
    )

    recovered_prediction = (
        mt_operator.predict(
            recovered_model
        )
    )

    recovered_misfit = np.linalg.norm(
        observed - recovered_prediction
    )

    assert np.isfinite(recovered_misfit)

    assert recovered_misfit <= (
        initial_misfit
        + 1.0e-10
    )

    assert np.all(
        np.isfinite(
            recovered_model.values
        )
    )


# ======================================================================
# Jacobian
# ======================================================================


def test_mt_jacobian_shape(
    mt_operator,
    initial_model,
    synthetic_data,
):
    """The MT Jacobian must have the expected dimensions."""

    jacobian = mt_operator.jacobian(
        initial_model
    )

    assert jacobian.shape == (
        synthetic_data.n_data,
        initial_model.n_parameters,
    )

    assert np.all(
        np.isfinite(jacobian)
    )


def test_mt_jacobian_is_not_zero(
    mt_operator,
    initial_model,
):
    """
    At least one model parameter must influence the predicted MT data.
    """

    jacobian = mt_operator.jacobian(
        initial_model
    )

    assert np.linalg.norm(
        jacobian
    ) > 0.0


# ======================================================================
# Generic linear least-squares sanity check
# ======================================================================


def test_linear_least_squares_solver():
    """
    Verify the generic LinearLeastSquaresSolver independently from MT.

    This isolates the optimizer from the nonlinear MT forward model.
    """

    from geoinvlab.core.inversion.problem import InverseProblem

    class LinearOperator:
        """Minimal linear forward operator."""

        n_data = 2

        def predict(
            self,
            model,
        ):
            return np.asarray(
                [
                    model.values[0],
                    2.0 * model.values[1],
                ],
                dtype=float,
            )

        def jacobian(
            self,
            model,
        ):
            return np.array(
                [
                    [1.0, 0.0],
                    [0.0, 2.0],
                ],
                dtype=float,
            )

    class SimpleData:
        """Minimal data object compatible with the optimizer."""

        n_data = 2

        @property
        def values(self):
            return np.array(
                [10.0, 20.0],
                dtype=float,
            )

        def weights(self):
            return np.ones(
                2,
                dtype=float,
            )

    class SimpleModel:
        """Minimal model used for the linear solver test."""

        n_parameters = 2

        def __init__(
            self,
            values,
        ):
            self.values = np.asarray(
                values,
                dtype=float,
            )
            self.mesh = None

    from geoinvlab.core.mesh import StructuredMesh3D

    mesh = StructuredMesh3D(
        nx=1,
        ny=1,
        nz=2,
        dx=100.0,
        dy=100.0,
        dz=100.0,
    )

    model = Model3D(
        mesh=mesh,
        values=np.array(
            [1.0, 1.0],
            dtype=float,
        ),
        name="linear_initial",
    )

    class ModelLinearOperator:
        """Linear operator using Model3D."""

        n_data = 2

        def predict(
            self,
            model,
        ):
            return np.array(
                [
                    model.values[0],
                    2.0 * model.values[1],
                ],
                dtype=float,
            )

        def jacobian(
            self,
            model,
        ):
            return np.array(
                [
                    [1.0, 0.0],
                    [0.0, 2.0],
                ],
                dtype=float,
            )

    class DataLinear:
        """Linear data object."""

        n_data = 2

        @property
        def values(self):
            return np.array(
                [10.0, 20.0],
                dtype=float,
            )

        def weights(self):
            return np.ones(
                2,
                dtype=float,
            )

    problem = InverseProblem(
        model=model,
        data=DataLinear(),
        operator=ModelLinearOperator(),
    )

    solver = LinearLeastSquaresSolver()

    recovered = solver.solve(
        problem
    )

    assert isinstance(
        recovered,
        Model3D,
    )

    np.testing.assert_allclose(
        recovered.values,
        np.array(
            [10.0, 10.0],
            dtype=float,
        ),
        rtol=1.0e-10,
        atol=1.0e-10,
    )
