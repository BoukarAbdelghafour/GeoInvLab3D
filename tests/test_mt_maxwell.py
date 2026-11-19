"""
Tests for the discrete 3D Maxwell system.
"""

import numpy as np
import pytest

from geoinvlab.core.mesh.structured import StructuredMesh3D
from geoinvlab.core.model.model import Model3D
from geoinvlab.methods.mt.discretization.maxwell import (
    MU0,
    _edge_dual_volumes,
    _face_dual_volumes,
    build_maxwell_system,
)


def make_mesh():
    return StructuredMesh3D(
        nx=2,
        ny=2,
        nz=2,
        dx=10.0,
        dy=20.0,
        dz=30.0,
    )


def test_build_maxwell_system():
    mesh = make_mesh()

    sigma = np.full(mesh.n_cells, 0.01)

    system = build_maxwell_system(
        mesh,
        sigma,
    )

    assert system.mesh is mesh
    assert system.n_cells == mesh.n_cells
    assert system.n_edges == system.curl.n_edges
    assert system.n_faces == system.curl.n_faces

    assert system.conductivity.shape == (mesh.n_cells,)
    assert system.mu_r.shape == (mesh.n_cells,)


def test_default_relative_permeability():
    mesh = make_mesh()
    sigma = np.full(mesh.n_cells, 0.01)

    system = build_maxwell_system(
        mesh,
        sigma,
    )

    np.testing.assert_allclose(
        system.mu_r,
        1.0,
    )


def test_absolute_permeability():
    mesh = make_mesh()
    sigma = np.full(mesh.n_cells, 0.01)

    system = build_maxwell_system(
        mesh,
        sigma,
        mu_r=2.0,
    )

    expected = 2.0 * MU0

    np.testing.assert_allclose(
        system.permeability(),
        expected,
    )


def test_inverse_permeability():
    mesh = make_mesh()
    sigma = np.full(mesh.n_cells, 0.01)

    system = build_maxwell_system(
        mesh,
        sigma,
        mu_r=2.0,
    )

    expected = 1.0 / (2.0 * MU0)

    np.testing.assert_allclose(
        system.inverse_permeability(),
        expected,
    )


def test_conductivity_mass_shape():
    mesh = make_mesh()
    sigma = np.full(mesh.n_cells, 0.01)

    system = build_maxwell_system(
        mesh,
        sigma,
    )

    mass = system.conductivity_mass()

    assert mass.shape == (
        system.n_edges,
        system.n_edges,
    )


def test_conductivity_mass_is_diagonal():
    mesh = make_mesh()
    sigma = np.full(mesh.n_cells, 0.01)

    system = build_maxwell_system(
        mesh,
        sigma,
    )

    mass = system.conductivity_mass()

    assert mass.nnz == system.n_edges

    rows, cols = mass.nonzero()

    assert np.all(rows == cols)


def test_uniform_conductivity_is_preserved():
    """
    For homogeneous conductivity:

        M_sigma[e,e] = sigma * V*_e

    where V*_e is the edge dual volume.
    """
    mesh = make_mesh()

    sigma_value = 0.02

    sigma = np.full(
        mesh.n_cells,
        sigma_value,
    )

    system = build_maxwell_system(
        mesh,
        sigma,
    )

    mass = system.conductivity_mass()

    diagonal = mass.diagonal()

    expected = (
        sigma_value
        * _edge_dual_volumes(mesh)
    )

    np.testing.assert_allclose(
        diagonal,
        expected,
    )


def test_permeability_mass_shape():
    mesh = make_mesh()
    sigma = np.full(mesh.n_cells, 0.01)

    system = build_maxwell_system(
        mesh,
        sigma,
    )

    mass = system.permeability_mass_inverse()

    assert mass.shape == (
        system.n_faces,
        system.n_faces,
    )


def test_permeability_mass_is_diagonal():
    mesh = make_mesh()
    sigma = np.full(mesh.n_cells, 0.01)

    system = build_maxwell_system(
        mesh,
        sigma,
    )

    mass = system.permeability_mass_inverse()

    assert mass.nnz == system.n_faces

    rows, cols = mass.nonzero()

    assert np.all(rows == cols)


def test_uniform_inverse_permeability_is_preserved():
    """
    For homogeneous relative permeability:

        M_mu_inv[f,f] = V*_f / mu

    where V*_f is the face dual volume.
    """
    mesh = make_mesh()

    sigma = np.full(
        mesh.n_cells,
        0.01,
    )

    system = build_maxwell_system(
        mesh,
        sigma,
        mu_r=2.0,
    )

    mass = system.permeability_mass_inverse()

    mu_inv = 1.0 / (2.0 * MU0)

    expected = (
        _face_dual_volumes(mesh)
        * mu_inv
    )

    np.testing.assert_allclose(
        mass.diagonal(),
        expected,
    )


def test_stiffness_matrix_shape():
    mesh = make_mesh()
    sigma = np.full(mesh.n_cells, 0.01)

    system = build_maxwell_system(
        mesh,
        sigma,
    )

    stiffness = system.stiffness_matrix()

    assert stiffness.shape == (
        system.n_edges,
        system.n_edges,
    )


def test_stiffness_matrix_is_symmetric():
    mesh = make_mesh()
    sigma = np.full(mesh.n_cells, 0.01)

    system = build_maxwell_system(
        mesh,
        sigma,
    )

    stiffness = system.stiffness_matrix()

    difference = stiffness - stiffness.T

    assert difference.nnz == 0


def test_stiffness_matrix_is_real():
    mesh = make_mesh()
    sigma = np.full(mesh.n_cells, 0.01)

    system = build_maxwell_system(
        mesh,
        sigma,
    )

    stiffness = system.stiffness_matrix()

    assert not np.iscomplexobj(stiffness.data)


def test_system_matrix_shape():
    mesh = make_mesh()
    sigma = np.full(mesh.n_cells, 0.01)

    system = build_maxwell_system(
        mesh,
        sigma,
    )

    omega = 2.0 * np.pi

    matrix = system.system_matrix(omega)

    assert matrix.shape == (
        system.n_edges,
        system.n_edges,
    )


def test_system_matrix_is_complex():
    mesh = make_mesh()
    sigma = np.full(mesh.n_cells, 0.01)

    system = build_maxwell_system(
        mesh,
        sigma,
    )

    matrix = system.system_matrix(
        2.0 * np.pi,
    )

    assert np.iscomplexobj(matrix.data)


def test_system_matrix_has_expected_real_part():
    mesh = make_mesh()

    sigma = np.full(
        mesh.n_cells,
        0.01,
    )

    system = build_maxwell_system(
        mesh,
        sigma,
    )

    omega = 2.0 * np.pi

    stiffness = system.stiffness_matrix()
    matrix = system.system_matrix(omega)

    np.testing.assert_allclose(
        matrix.real.toarray(),
        stiffness.toarray(),
    )


def test_system_matrix_imaginary_part_is_conductivity_term():
    mesh = make_mesh()

    sigma = np.full(
        mesh.n_cells,
        0.01,
    )

    system = build_maxwell_system(
        mesh,
        sigma,
    )

    omega = 2.0 * np.pi

    conductivity_mass = system.conductivity_mass()
    matrix = system.system_matrix(omega)

    expected = omega * conductivity_mass.toarray()

    np.testing.assert_allclose(
        matrix.imag.toarray(),
        expected,
    )


def test_system_matrix_changes_with_frequency():
    mesh = make_mesh()

    sigma = np.full(
        mesh.n_cells,
        0.01,
    )

    system = build_maxwell_system(
        mesh,
        sigma,
    )

    omega_1 = 1.0
    omega_2 = 10.0

    a1 = system.system_matrix(omega_1)
    a2 = system.system_matrix(omega_2)

    difference = a2 - a1

    assert difference.nnz > 0

    expected_difference = (
        (omega_2 - omega_1)
        * system.conductivity_mass()
    )

    np.testing.assert_allclose(
        difference.imag.toarray(),
        expected_difference.toarray(),
    )


def test_model3d_conductivity_is_accepted():
    mesh = make_mesh()

    sigma = np.full(
        mesh.n_cells,
        0.01,
    )

    model = Model3D(
        mesh=mesh,
        values=sigma,
        name="conductivity",
    )

    system = build_maxwell_system(
        mesh,
        model,
    )

    np.testing.assert_allclose(
        system.conductivity,
        sigma,
    )


def test_variable_conductivity_is_accepted():
    mesh = make_mesh()

    sigma = np.linspace(
        0.001,
        0.1,
        mesh.n_cells,
    )

    system = build_maxwell_system(
        mesh,
        sigma,
    )

    np.testing.assert_allclose(
        system.conductivity,
        sigma,
    )


def test_variable_mu_r_is_accepted():
    mesh = make_mesh()

    sigma = np.full(
        mesh.n_cells,
        0.01,
    )

    mu_r = np.linspace(
        1.0,
        2.0,
        mesh.n_cells,
    )

    system = build_maxwell_system(
        mesh,
        sigma,
        mu_r=mu_r,
    )

    np.testing.assert_allclose(
        system.mu_r,
        mu_r,
    )


def test_invalid_frequency_zero():
    mesh = make_mesh()
    sigma = np.full(mesh.n_cells, 0.01)

    system = build_maxwell_system(
        mesh,
        sigma,
    )

    with pytest.raises(ValueError):
        system.system_matrix(0.0)


def test_invalid_frequency_negative():
    mesh = make_mesh()
    sigma = np.full(mesh.n_cells, 0.01)

    system = build_maxwell_system(
        mesh,
        sigma,
    )

    with pytest.raises(ValueError):
        system.system_matrix(-1.0)


def test_invalid_frequency_nan():
    mesh = make_mesh()
    sigma = np.full(mesh.n_cells, 0.01)

    system = build_maxwell_system(
        mesh,
        sigma,
    )

    with pytest.raises(ValueError):
        system.system_matrix(np.nan)


def test_invalid_conductivity_size():
    mesh = make_mesh()

    sigma = np.ones(mesh.n_cells + 1)

    with pytest.raises(ValueError):
        build_maxwell_system(
            mesh,
            sigma,
        )


def test_invalid_conductivity_dimension():
    mesh = make_mesh()

    sigma = np.ones(
        (mesh.n_cells, 1),
    )

    with pytest.raises(ValueError):
        build_maxwell_system(
            mesh,
            sigma,
        )


def test_invalid_conductivity_nan():
    mesh = make_mesh()

    sigma = np.full(
        mesh.n_cells,
        0.01,
    )

    sigma[0] = np.nan

    with pytest.raises(ValueError):
        build_maxwell_system(
            mesh,
            sigma,
        )


def test_invalid_mu_r_zero():
    mesh = make_mesh()
    sigma = np.full(mesh.n_cells, 0.01)

    with pytest.raises(ValueError):
        build_maxwell_system(
            mesh,
            sigma,
            mu_r=0.0,
        )


def test_invalid_mu_r_negative():
    mesh = make_mesh()
    sigma = np.full(mesh.n_cells, 0.01)

    with pytest.raises(ValueError):
        build_maxwell_system(
            mesh,
            sigma,
            mu_r=-1.0,
        )


def test_invalid_mu_r_size():
    mesh = make_mesh()
    sigma = np.full(mesh.n_cells, 0.01)

    mu_r = np.ones(
        mesh.n_cells + 1,
    )

    with pytest.raises(ValueError):
        build_maxwell_system(
            mesh,
            sigma,
            mu_r=mu_r,
        )


def test_invalid_mesh():
    sigma = np.ones(8)

    with pytest.raises(TypeError):
        build_maxwell_system(
            object(),
            sigma,
        )
