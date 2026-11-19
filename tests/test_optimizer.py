import numpy as np

from geoinvlab.core.data import DataSet
from geoinvlab.core.inversion import (
    InverseProblem,
    LinearLeastSquaresSolver,
)
from geoinvlab.core.mesh import StructuredMesh3D
from geoinvlab.core.model import Model3D
from geoinvlab.core.operators import LinearOperator


def create_mesh():

    return StructuredMesh3D(
        nx=3,
        ny=1,
        nz=1,
        dx=10.0,
        dy=10.0,
        dz=10.0,
    )


def create_true_model():

    return Model3D(
        mesh=create_mesh(),
        values=np.array([
            2.0,
            4.0,
            6.0,
        ]),
    )


def create_operator():

    matrix = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 3.0],
    ])

    return LinearOperator(matrix)


def create_data():

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


def create_initial_model():

    return Model3D(
        mesh=create_mesh(),
        values=np.zeros(3),
    )


def test_linear_least_squares_solver():

    problem = InverseProblem(
        model=create_initial_model(),
        data=create_data(),
        operator=create_operator(),
    )

    solver = LinearLeastSquaresSolver()

    estimated_model = solver.solve(problem)

    expected = np.array([
        2.0,
        4.0,
        6.0,
    ])

    np.testing.assert_allclose(
        estimated_model.values,
        expected,
    )


def test_solution_fits_data():

    problem = InverseProblem(
        model=create_initial_model(),
        data=create_data(),
        operator=create_operator(),
    )

    solver = LinearLeastSquaresSolver()

    estimated_model = solver.solve(problem)

    predicted = problem.operator.predict(
        estimated_model
    )

    np.testing.assert_allclose(
        predicted,
        problem.data.values,
    )
def test_underdetermined_problem():

    mesh = StructuredMesh3D(
        nx=3,
        ny=1,
        nz=1,
        dx=10.0,
        dy=10.0,
        dz=10.0,
    )

    model = Model3D(
        mesh=mesh,
        values=np.zeros(3),
    )

    matrix = np.array([
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
    ])

    operator = LinearOperator(matrix)

    data = DataSet(
        values=np.array([
            3.0,
            4.0,
        ]),
        locations=np.array([
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ]),
    )

    problem = InverseProblem(
        model=model,
        data=data,
        operator=operator,
    )

    solver = LinearLeastSquaresSolver()

    estimated_model = solver.solve(problem)

    predicted = operator.predict(
        estimated_model
    )

    np.testing.assert_allclose(
        predicted,
        data.values,
    )
def test_regularized_linear_least_squares_solver():
    import numpy as np

    from geoinvlab.core.data import DataSet
    from geoinvlab.core.inversion import (
        InverseProblem,
        L2Regularization,
        LinearLeastSquaresSolver,
    )
    from geoinvlab.core.mesh import StructuredMesh3D
    from geoinvlab.core.model import Model3D
    from geoinvlab.core.operators import LinearOperator

    mesh = StructuredMesh3D(
        nx=2,
        ny=1,
        nz=1,
        dx=1.0,
        dy=1.0,
        dz=1.0,
    )

    model = Model3D(
        mesh=mesh,
        values=np.zeros(2),
    )

    operator = LinearOperator(
        np.eye(2),
    )

    data = DataSet(
        values=np.array([
            2.0,
            4.0,
        ]),
        locations=np.array([
            [0.0, 0.0, 10.0],
            [1.0, 0.0, 10.0],
        ]),
    )

    problem = InverseProblem(
        model=model,
        data=data,
        operator=operator,
        regularization=L2Regularization(),
        regularization_parameter=1.0,
    )

    solver = LinearLeastSquaresSolver()

    estimated_model = solver.solve(problem)

    expected = np.array([
        1.0,
        2.0,
    ])

    np.testing.assert_allclose(
        estimated_model.values,
        expected,
    )
