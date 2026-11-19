import numpy as np

from geoinvlab.core.mesh.structured import StructuredMesh3D
from geoinvlab.core.model import Model3D


def create_test_mesh():
    return StructuredMesh3D(
        nx=4,
        ny=3,
        nz=2,
        dx=10.0,
        dy=10.0,
        dz=10.0,
    )


def test_model_creation():

    mesh = create_test_mesh()

    values = np.zeros(mesh.n_cells)

    model = Model3D(
        mesh=mesh,
        values=values,
        name="density_contrast",
    )

    assert model.n_parameters == 24
    assert model.name == "density_contrast"


def test_model_reshape():

    mesh = create_test_mesh()

    values = np.arange(mesh.n_cells)

    model = Model3D(
        mesh=mesh,
        values=values,
    )

    assert model.reshape().shape == (4, 3, 2)


def test_model_statistics():

    mesh = create_test_mesh()

    values = np.arange(mesh.n_cells, dtype=float)

    model = Model3D(
        mesh=mesh,
        values=values,
    )

    assert model.min() == 0.0
    assert model.max() == 23.0
    assert model.mean() == 11.5


def test_model_copy():

    mesh = create_test_mesh()

    values = np.ones(mesh.n_cells)

    model = Model3D(
        mesh=mesh,
        values=values,
    )

    copied = model.copy()

    assert copied is not model
    assert np.array_equal(copied.values, model.values)

