import numpy as np

from geoinvlab.core.data import DataSet
from geoinvlab.core.mesh import StructuredMesh3D
from geoinvlab.core.model import Model3D
from geoinvlab.core.operators import LinearOperator
from geoinvlab.core.inversion.objective import LeastSquaresObjective
from geoinvlab.core.inversion import L2Regularization
from conftest import (
    create_test_model,
    create_test_data,
    create_test_operator,
    )
def create_model():

    mesh = StructuredMesh3D(
        nx=3,
        ny=1,
        nz=1,
        dx=10.0,
        dy=10.0,
        dz=10.0,
    )

    return Model3D(
        mesh=mesh,
        values=np.array([
            1.0,
            2.0,
            3.0,
        ]),
    )


def create_data():

    return DataSet(
        values=np.array([
            1.0,
            4.0,
            8.0,
        ]),
        locations=np.array([
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
        ]),
    )


def create_operator():

    matrix = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 3.0],
    ])

    return LinearOperator(matrix)


def test_residual():

    model = create_model()
    data = create_data()
    operator = create_operator()

    objective = LeastSquaresObjective(
        operator=operator,
        data=data,
    )

    residual = objective.residual(model)

    expected = np.array([
        0.0,
        0.0,
        1.0,
    ])

    np.testing.assert_allclose(
        residual,
        expected,
    )


def test_weighted_residual():

    model = create_model()

    data = DataSet(
        values=np.array([
            1.0,
            4.0,
            8.0,
        ]),
        locations=np.array([
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
        ]),
        uncertainties=np.array([
            1.0,
            2.0,
            0.5,
        ]),
    )

    operator = create_operator()

    objective = LeastSquaresObjective(
        operator=operator,
        data=data,
    )

    weighted = objective.weighted_residual(model)

    expected = np.array([
        0.0,
        0.0,
        2.0,
    ])

    np.testing.assert_allclose(
        weighted,
        expected,
    )


def test_objective_value():

    model = create_model()
    data = create_data()
    operator = create_operator()

    objective = LeastSquaresObjective(
        operator=operator,
        data=data,
    )

    value = objective.value(model)

    assert value == 1.0
def test_regularized_objective():
    model = create_test_model()

    data = create_test_data()
    operator = create_test_operator()

    regularization = L2Regularization()

    objective = LeastSquaresObjective(
        operator=operator,
        data=data,
        regularization=regularization,
        regularization_parameter=0.5,
    )

    value = objective.value(model)

    expected = (
        objective.data_misfit(model)
        + 0.5 * regularization.value(model)
    )

    assert value == expected

