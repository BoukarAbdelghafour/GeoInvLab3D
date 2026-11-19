import numpy as np
import pytest

from geoinvlab.core.data import DataSet
from geoinvlab.core.mesh import StructuredMesh3D
from geoinvlab.core.model import Model3D
from geoinvlab.core.operators import ForwardOperator
from geoinvlab.core.operators.linear import LinearOperator
from conftest import create_test_model, create_test_data

class TestOperator(ForwardOperator):

    @property
    def n_data(self) -> int:
        """Return the number of test observations."""

        return 3

    def predict(self, model):

        return np.array([
            1.0,
            2.0,
            3.0,
        ])

    def jacobian(self, model):

        return np.eye(3)


def test_forward_operator_prediction():

    operator = TestOperator()
    model = create_test_model()

    predicted = operator.predict(model)

    assert predicted.shape == (3,)


def test_forward_operator_jacobian():

    operator = TestOperator()
    model = create_test_model()

    jacobian = operator.jacobian(model)

    assert jacobian.shape == (3, 3)


def test_prediction_compatibility():

    operator = TestOperator()

    model = create_test_model()
    data = create_test_data()

    operator.check_prediction(
        model,
        data,
    )


def test_abstract_operator():

    with pytest.raises(TypeError):
        ForwardOperator()
def test_linear_operator():

    matrix = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 3.0],
    ])

    operator = LinearOperator(matrix)

    model = create_test_model()
    model.values[:] = [1.0, 2.0, 3.0]

    predicted = operator.predict(model)

    expected = np.array([
        1.0,
        4.0,
        9.0,
    ])

    np.testing.assert_allclose(
        predicted,
        expected,
    )
def test_linear_operator_jacobian():

    matrix = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ])

    operator = LinearOperator(matrix)

    model = create_test_model()

    jacobian = operator.jacobian(model)

    np.testing.assert_allclose(
        jacobian,
        matrix,
    )

