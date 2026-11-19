import numpy as np

from geoinvlab.core.inversion import L2Regularization
from geoinvlab.core.mesh import StructuredMesh3D
from geoinvlab.core.model import Model3D


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


def test_l2_regularization_value():

    regularization = L2Regularization()
    model = create_model()

    value = regularization.value(model)

    assert value == 14.0


def test_l2_regularization_gradient():

    regularization = L2Regularization()
    model = create_model()

    gradient = regularization.gradient(model)

    expected = np.array([
        2.0,
        4.0,
        6.0,
    ])

    np.testing.assert_allclose(
        gradient,
        expected,
    )
def test_l2_regularization_matrix():
    import numpy as np

    from geoinvlab.core.inversion import L2Regularization
    from geoinvlab.core.mesh import StructuredMesh3D
    from geoinvlab.core.model import Model3D

    mesh = StructuredMesh3D(
        nx=3,
        ny=1,
        nz=1,
        dx=1.0,
        dy=1.0,
        dz=1.0,
    )

    model = Model3D(
        mesh=mesh,
        values=np.array([
            1.0,
            2.0,
            3.0,
        ]),
    )

    regularization = L2Regularization()

    matrix = regularization.matrix(model)

    np.testing.assert_allclose(
        matrix,
        np.eye(3),
    )


def test_l2_gradient_matches_matrix():
    import numpy as np

    from geoinvlab.core.inversion import L2Regularization
    from geoinvlab.core.mesh import StructuredMesh3D
    from geoinvlab.core.model import Model3D

    mesh = StructuredMesh3D(
        nx=3,
        ny=1,
        nz=1,
        dx=1.0,
        dy=1.0,
        dz=1.0,
    )

    values = np.array([
        1.0,
        2.0,
        3.0,
    ])

    model = Model3D(
        mesh=mesh,
        values=values,
    )

    regularization = L2Regularization()

    matrix = regularization.matrix(model)

    expected = (
        2.0
        * matrix.T
        @ matrix
        @ values
    )

    np.testing.assert_allclose(
        regularization.gradient(model),
        expected,
    )
