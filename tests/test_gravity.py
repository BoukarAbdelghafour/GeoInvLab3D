"""
MIT License Copyright (c) 17/09/2026 [Boukar Abdelghafour]
"""
import numpy as np
import pytest

from geoinvlab.methods.gravity import (
    GravityMethod,
    GravityOperator,
)


def test_gravity_method_identity():
    method = GravityMethod()

    assert method.name == "Gravity"
    assert method.identifier == "gravity"


def test_gravity_method_creates_operator():
    locations = np.array(
        [
            [0.0, 0.0, 10.0],
            [10.0, 0.0, 10.0],
        ]
    )

    method = GravityMethod()
    operator = method.create_operator(locations)

    assert isinstance(operator, GravityOperator)
    assert operator.n_data == 2


def test_gravity_operator_invalid_locations():
    with pytest.raises(
        ValueError,
        match="shape \\(n_data, 3\\)",
    ):
        GravityOperator(
            observation_locations=np.zeros((5, 2))
        )
def test_gravity_jacobian_shape():
    import numpy as np

    from geoinvlab.core.mesh import StructuredMesh3D
    from geoinvlab.core.model import Model3D

    mesh = StructuredMesh3D(
        nx=2,
        ny=2,
        nz=2,
        dx=1.0,
        dy=1.0,
        dz=1.0,
    )

    model = Model3D(
        mesh=mesh,
        values=np.ones(mesh.n_cells),
    )

    locations = np.array(
        [
            [0.5, 0.5, 10.0],
            [1.5, 1.5, 10.0],
            [0.0, 0.0, 12.0],
        ]
    )

    operator = GravityOperator(locations)

    jacobian = operator.jacobian(model)

    assert jacobian.shape == (
        3,
        mesh.n_cells,
    )


def test_gravity_prediction_matches_jacobian():
    import numpy as np

    from geoinvlab.core.mesh import StructuredMesh3D
    from geoinvlab.core.model import Model3D

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
        values=np.array([100.0, 200.0]),
    )

    locations = np.array(
        [
            [0.5, 0.5, 10.0],
            [1.5, 0.5, 10.0],
        ]
    )

    operator = GravityOperator(locations)

    jacobian = operator.jacobian(model)
    prediction = operator.predict(model)

    np.testing.assert_allclose(
        prediction,
        jacobian @ model.values,
    )
def test_gravity_jacobian_is_cached():
    import numpy as np

    from geoinvlab.core.mesh import StructuredMesh3D
    from geoinvlab.core.model import Model3D
    from geoinvlab.methods.gravity import GravityOperator

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
        values=np.ones(mesh.n_cells),
    )

    locations = np.array([
        [0.5, 0.5, 10.0],
        [1.5, 0.5, 10.0],
    ])

    operator = GravityOperator(locations)

    first = operator.jacobian(model)
    second = operator.jacobian(model)

    assert first is second


def test_gravity_prediction_reuses_jacobian():
    import numpy as np

    from geoinvlab.core.mesh import StructuredMesh3D
    from geoinvlab.core.model import Model3D
    from geoinvlab.methods.gravity import GravityOperator

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
        values=np.array([
            100.0,
            200.0,
        ]),
    )

    locations = np.array([
        [0.5, 0.5, 10.0],
        [1.5, 0.5, 10.0],
    ])

    operator = GravityOperator(locations)

    jacobian = operator.jacobian(model)
    predicted = operator.predict(model)

    expected = jacobian @ model.values

    np.testing.assert_allclose(
        predicted,
        expected,
    )

    assert operator._jacobian is jacobian


def test_gravity_clear_cache():
    import numpy as np

    from geoinvlab.core.mesh import StructuredMesh3D
    from geoinvlab.core.model import Model3D
    from geoinvlab.methods.gravity import GravityOperator

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
        values=np.ones(mesh.n_cells),
    )

    locations = np.array([
        [0.5, 0.5, 10.0],
    ])

    operator = GravityOperator(locations)

    first = operator.jacobian(model)

    operator.clear_cache()

    assert operator._jacobian is None
    assert operator._cached_mesh_signature is None

    second = operator.jacobian(model)

    assert second is not first

    np.testing.assert_allclose(
        second,
        first,
    )
def test_gravity_jacobian_recomputed_for_different_mesh():
    import numpy as np

    from geoinvlab.core.mesh import StructuredMesh3D
    from geoinvlab.core.model import Model3D
    from geoinvlab.methods.gravity import GravityOperator

    mesh1 = StructuredMesh3D(
        nx=2,
        ny=1,
        nz=1,
        dx=1.0,
        dy=1.0,
        dz=1.0,
    )

    mesh2 = StructuredMesh3D(
        nx=3,
        ny=1,
        nz=1,
        dx=1.0,
        dy=1.0,
        dz=1.0,
    )

    model1 = Model3D(
        mesh=mesh1,
        values=np.ones(mesh1.n_cells),
    )

    model2 = Model3D(
        mesh=mesh2,
        values=np.ones(mesh2.n_cells),
    )

    locations = np.array([
        [0.5, 0.5, 10.0],
    ])

    operator = GravityOperator(locations)

    first = operator.jacobian(model1)
    second = operator.jacobian(model2)

    assert first is not second

    assert first.shape == (1, 2)
    assert second.shape == (1, 3)

