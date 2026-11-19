from geoinvlab.core.mesh.structured import StructuredMesh3D


def test_number_of_cells():
    mesh = StructuredMesh3D(
        nx=40,
        ny=30,
        nz=20,
        dx=10.0,
        dy=10.0,
        dz=10.0,
    )

    assert mesh.n_cells == 24000


def test_mesh_shape():
    mesh = StructuredMesh3D(
        nx=40,
        ny=30,
        nz=20,
        dx=10.0,
        dy=10.0,
        dz=10.0,
    )

    assert mesh.shape == (40, 30, 20)


def test_cell_volume():
    mesh = StructuredMesh3D(
        nx=40,
        ny=30,
        nz=20,
        dx=10.0,
        dy=10.0,
        dz=10.0,
    )

    assert mesh.cell_volume == 1000.0


def test_mesh_dimensions():
    mesh = StructuredMesh3D(
        nx=40,
        ny=30,
        nz=20,
        dx=10.0,
        dy=10.0,
        dz=10.0,
    )

    assert mesh.dimensions == (400.0, 300.0, 200.0)


def test_cell_centers():
    mesh = StructuredMesh3D(
        nx=40,
        ny=30,
        nz=20,
        dx=10.0,
        dy=10.0,
        dz=10.0,
    )

    centers = mesh.cell_centers()

    assert centers.shape == (24000, 3)

