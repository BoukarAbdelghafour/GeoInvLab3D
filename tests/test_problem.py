import numpy as np
import pytest

from geoinvlab.core.data import DataSet
from geoinvlab.core.inversion import (
    InverseProblem,
    L2Regularization,
)
from geoinvlab.core.mesh import StructuredMesh3D
from geoinvlab.core.model import Model3D
from geoinvlab.core.operators import LinearOperator


def create_mesh():
    """Create a small mesh for testing."""
    return StructuredMesh3D(
        nx=3,
        ny=1,
        nz=1,
        dx=10.0,
        dy=10.0,
        dz=10.0,
    )


def create_initial_model():
    """Create an initial zero-valued model."""
    return Model3D(
        mesh=create_mesh(),
        values=np.zeros(3),
    )


def create_data():
    """Create synthetic observation data."""
    return DataSet(
        values=np.array([
            2.0,
            8.0,
            18.0,
        ]),
        locations=np.array([
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
        ]),
    )


def create_operator():
    """Create a simple diagonal linear operator."""
    matrix = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 3.0],
    ])

    return LinearOperator(matrix)


def test_problem_dimensions():
    """Test dimensions exposed by the inverse problem."""

    problem = InverseProblem(
        model=create_initial_model(),
        data=create_data(),
        operator=create_operator(),
    )

    assert problem.model.n_parameters == 3
    assert problem.data.n_data == 3


def test_problem_is_not_regularized_by_default():
    """Test that regularization is disabled by default."""

    problem = InverseProblem(
        model=create_initial_model(),
        data=create_data(),
        operator=create_operator(),
    )

    assert problem.regularization is None
    assert problem.regularization_parameter == 0.0


def test_problem_is_regularized():
    """Test creation of a regularized inverse problem."""

    regularization = L2Regularization()

    problem = InverseProblem(
        model=create_initial_model(),
        data=create_data(),
        operator=create_operator(),
        regularization=regularization,
        regularization_parameter=1.0,
    )

    assert problem.regularization is regularization
    assert problem.regularization_parameter == 1.0


def test_zero_regularization_is_not_active():
    """Test that zero regularization has no contribution."""

    problem = InverseProblem(
        model=create_initial_model(),
        data=create_data(),
        operator=create_operator(),
        regularization=L2Regularization(),
        regularization_parameter=0.0,
    )

    assert problem.regularization_parameter == 0.0

    assert problem.objective_value() == problem.objective.data_misfit(
        problem.model
    )


def test_negative_regularization_parameter():
    """Test rejection of a negative regularization parameter."""

    with pytest.raises(ValueError):
        InverseProblem(
            model=create_initial_model(),
            data=create_data(),
            operator=create_operator(),
            regularization_parameter=-1.0,
        )


def test_problem_data_misfit():
    """Test the data misfit exposed by the inverse problem."""

    problem = InverseProblem(
        model=create_initial_model(),
        data=create_data(),
        operator=create_operator(),
    )

    expected = np.sum(
        np.square(
            -problem.data.values
        )
    )

    assert problem.objective.data_misfit(
        problem.model
    ) == expected


def test_problem_regularization_value():
    """Test the regularization contribution."""

    problem = InverseProblem(
        model=create_initial_model(),
        data=create_data(),
        operator=create_operator(),
        regularization=L2Regularization(),
        regularization_parameter=2.0,
    )

    expected = (
        2.0
        * problem.regularization.value(
            problem.model
        )
    )

    assert (
        problem.objective_value()
        == problem.objective.data_misfit(
            problem.model
        ) + expected
    )


def test_problem_residual():
    """Test the residual returned by the inverse problem."""

    problem = InverseProblem(
        model=create_initial_model(),
        data=create_data(),
        operator=create_operator(),
    )

    expected = -problem.data.values

    np.testing.assert_allclose(
        problem.residual(),
        expected,
    )


def test_problem_weighted_residual():
    """Test the weighted residual returned by the inverse problem."""

    problem = InverseProblem(
        model=create_initial_model(),
        data=create_data(),
        operator=create_operator(),
    )

    expected = -problem.data.values

    np.testing.assert_allclose(
        problem.weighted_residual(),
        expected,
    )

