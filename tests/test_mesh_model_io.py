"""Round-trip tests for structured mesh and cell-based model I/O."""

import numpy as np
import pytest

from geoinvlab.core.mesh import StructuredMesh3D, load_mesh, save_mesh
from geoinvlab.core.model import Model3D, load_model, save_model


def make_mesh(**overrides):
    """Create a small padded structured mesh for I/O tests."""
    parameters = {
        "nx": 3,
        "ny": 2,
        "nz": 4,
        "dx": 100.0,
        "dy": 150.0,
        "dz": 50.0,
        "origin": (-300.0, -200.0, 0.0),
        "padding_x": (1, 2),
        "padding_y": (2, 1),
        "padding_z": (1, 1),
        "padding_factor": 1.4,
    }
    parameters.update(overrides)
    return StructuredMesh3D(**parameters)


def test_mesh_ascii_round_trip_preserves_geometry(tmp_path):
    original = make_mesh()
    path = tmp_path / "mesh.dat"

    save_mesh(original, path)
    restored = load_mesh(path)

    assert restored.nx == original.nx
    assert restored.ny == original.ny
    assert restored.nz == original.nz
    assert restored.dx == pytest.approx(original.dx)
    assert restored.dy == pytest.approx(original.dy)
    assert restored.dz == pytest.approx(original.dz)
    assert restored.origin == pytest.approx(original.origin)
    assert restored.padding_x == original.padding_x
    assert restored.padding_y == original.padding_y
    assert restored.padding_z == original.padding_z
    assert restored.padding_factor == pytest.approx(original.padding_factor)
    assert restored.n_cells == original.n_cells
    np.testing.assert_allclose(
        restored.cell_centers(),
        original.cell_centers(),
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        restored.cell_volumes(),
        original.cell_volumes(),
        rtol=0.0,
        atol=1e-12,
    )
    assert path.read_bytes().isascii()


def test_model_ascii_round_trip_preserves_values_and_metadata(tmp_path):
    mesh = make_mesh()
    values = np.linspace(10.0, 1000.0, mesh.n_cells)
    original = Model3D(mesh=mesh, values=values, name="resistivity")
    path = tmp_path / "model.dat"

    save_model(original, path, unit="ohm.m")
    restored, metadata = load_model(
        path,
        mesh,
        return_metadata=True,
    )

    assert restored.name == original.name
    assert restored.mesh is mesh
    np.testing.assert_array_equal(restored.values, original.values)
    assert metadata["name"] == "resistivity"
    assert metadata["unit"] == "ohm.m"
    assert metadata["value_count"] == mesh.n_cells
    assert metadata["ordering"] == "x-y-z (z fastest)"
    assert path.read_bytes().isascii()


def test_model_loading_rejects_mesh_with_same_cell_count_but_different_origin(
    tmp_path,
):
    mesh = make_mesh()
    model = Model3D(
        mesh=mesh,
        values=np.arange(1, mesh.n_cells + 1, dtype=float),
        name="density",
    )
    path = tmp_path / "model.dat"
    save_model(model, path, unit="kg/m3")

    incompatible_mesh = make_mesh(origin=(-250.0, -200.0, 0.0))
    assert incompatible_mesh.n_cells == mesh.n_cells

    with pytest.raises(ValueError, match="origin"):
        load_model(path, incompatible_mesh)


def test_model_loading_rejects_different_padding(tmp_path):
    mesh = make_mesh()
    model = Model3D(
        mesh=mesh,
        values=np.ones(mesh.n_cells),
        name="susceptibility",
    )
    path = tmp_path / "model.dat"
    save_model(model, path, unit="SI")

    incompatible_mesh = make_mesh(padding_x=(0, 3))
    assert incompatible_mesh.n_cells == mesh.n_cells

    with pytest.raises(ValueError, match="padding_x"):
        load_model(path, incompatible_mesh)


def test_loading_missing_mesh_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_mesh(tmp_path / "missing_mesh.dat")


def test_loading_missing_model_file_raises_file_not_found(tmp_path):
    mesh = make_mesh()
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "missing_model.dat", mesh)


def test_model_loading_rejects_truncated_values_section(tmp_path):
    mesh = make_mesh()
    model = Model3D(
        mesh=mesh,
        values=np.ones(mesh.n_cells),
        name="density",
    )
    path = tmp_path / "model.dat"
    save_model(model, path, unit="kg/m3")

    lines = path.read_text(encoding="ascii").splitlines()
    values_index = lines.index("values")
    path.write_text(
        "\n".join(lines[:values_index + 2]) + "\n",
        encoding="ascii",
    )

    with pytest.raises(ValueError, match="contains"):
        load_model(path, mesh)
