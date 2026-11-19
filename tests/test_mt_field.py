import numpy as np
import pytest

from geoinvlab.core.mesh.structured import StructuredMesh3D
from geoinvlab.methods.mt.discretization.curl import build_curl
from geoinvlab.methods.mt.discretization.field import (
    EdgeElectricField,
    build_edge_electric_field,
    electric_curl,
    magnetic_field,
)
from geoinvlab.methods.mt.discretization.maxwell import MU0


def make_mesh():
    return StructuredMesh3D(
        nx=2,
        ny=2,
        nz=2,
        dx=100.0,
        dy=100.0,
        dz=100.0,
    )


def n_edge_counts(mesh):
    n_ex = mesh.nx * (mesh.ny + 1) * (mesh.nz + 1)
    n_ey = (mesh.nx + 1) * mesh.ny * (mesh.nz + 1)
    n_ez = (mesh.nx + 1) * (mesh.ny + 1) * mesh.nz

    return n_ex, n_ey, n_ez


# ============================================================================
# EdgeElectricField
# ============================================================================


def test_edge_electric_field_splits_components():
    mesh = make_mesh()

    n_ex, n_ey, n_ez = n_edge_counts(mesh)

    values = np.arange(
        n_ex + n_ey + n_ez,
        dtype=float,
    ).astype(complex)

    field = build_edge_electric_field(mesh, values)

    assert isinstance(field, EdgeElectricField)

    assert field.n_ex == n_ex
    assert field.n_ey == n_ey
    assert field.n_ez == n_ez

    assert field.n_edges == n_ex + n_ey + n_ez

    np.testing.assert_allclose(
        field.ex,
        values[:n_ex],
    )

    np.testing.assert_allclose(
        field.ey,
        values[n_ex:n_ex + n_ey],
    )

    np.testing.assert_allclose(
        field.ez,
        values[n_ex + n_ey:],
    )


def test_edge_electric_field_as_vector_returns_copy():
    mesh = make_mesh()

    n_ex, n_ey, n_ez = n_edge_counts(mesh)

    values = np.ones(
        n_ex + n_ey + n_ez,
        dtype=complex,
    )

    field = build_edge_electric_field(mesh, values)

    result = field.as_vector()

    np.testing.assert_allclose(result, values)

    result[0] = 99.0

    # as_vector() doit retourner une copie.
    assert field.values[0] == 1.0


def test_edge_electric_field_accepts_complex_values():
    mesh = make_mesh()

    n_ex, n_ey, n_ez = n_edge_counts(mesh)

    values = (
        np.ones(n_ex + n_ey + n_ez)
        + 2.0j
    )

    field = build_edge_electric_field(mesh, values)

    np.testing.assert_allclose(
        field.values,
        values,
    )


def test_edge_electric_field_rejects_wrong_dimension():
    mesh = make_mesh()

    with pytest.raises(
        ValueError,
        match="one-dimensional",
    ):
        build_edge_electric_field(
            mesh,
            np.zeros((2, 2), dtype=complex),
        )


def test_edge_electric_field_rejects_wrong_size():
    mesh = make_mesh()

    n_ex, n_ey, n_ez = n_edge_counts(mesh)

    values = np.zeros(
        n_ex + n_ey + n_ez - 1,
        dtype=complex,
    )

    with pytest.raises(
        ValueError,
        match="incompatible size",
    ):
        build_edge_electric_field(mesh, values)


def test_edge_electric_field_rejects_nonfinite_values():
    mesh = make_mesh()

    n_ex, n_ey, n_ez = n_edge_counts(mesh)

    values = np.zeros(
        n_ex + n_ey + n_ez,
        dtype=complex,
    )

    values[0] = np.nan

    with pytest.raises(
        ValueError,
        match="non-finite",
    ):
        build_edge_electric_field(mesh, values)


def test_edge_electric_field_rejects_wrong_mesh_type():
    values = np.zeros(10, dtype=complex)

    with pytest.raises(
        TypeError,
        match="StructuredMesh3D",
    ):
        build_edge_electric_field(
            object(),
            values,
        )


# ============================================================================
# electric_curl
# ============================================================================


def test_electric_curl_has_face_size():
    mesh = make_mesh()

    n_ex, n_ey, n_ez = n_edge_counts(mesh)

    values = np.zeros(
        n_ex + n_ey + n_ez,
        dtype=complex,
    )

    field = build_edge_electric_field(mesh, values)

    curl = electric_curl(mesh, field)

    curl_operator = build_curl(mesh)

    assert curl.ndim == 1
    assert curl.size == curl_operator.n_faces


def test_electric_curl_accepts_raw_vector():
    mesh = make_mesh()

    n_ex, n_ey, n_ez = n_edge_counts(mesh)

    values = np.ones(
        n_ex + n_ey + n_ez,
        dtype=complex,
    )

    result = electric_curl(mesh, values)

    expected = build_curl(mesh).matrix @ values

    np.testing.assert_allclose(
        result,
        expected,
    )


def test_electric_curl_accepts_edge_field():
    mesh = make_mesh()

    n_ex, n_ey, n_ez = n_edge_counts(mesh)

    values = (
        np.arange(
            n_ex + n_ey + n_ez,
            dtype=float,
        )
        + 1.0j
    )

    field = build_edge_electric_field(mesh, values)

    result = electric_curl(mesh, field)

    expected = build_curl(mesh).matrix @ values

    np.testing.assert_allclose(
        result,
        expected,
    )


def test_electric_curl_of_zero_field_is_zero():
    mesh = make_mesh()

    n_ex, n_ey, n_ez = n_edge_counts(mesh)

    values = np.zeros(
        n_ex + n_ey + n_ez,
        dtype=complex,
    )

    result = electric_curl(mesh, values)

    np.testing.assert_allclose(
        result,
        0.0,
    )


def test_electric_curl_rejects_wrong_size():
    mesh = make_mesh()

    n_ex, n_ey, n_ez = n_edge_counts(mesh)

    values = np.zeros(
        n_ex + n_ey + n_ez - 1,
        dtype=complex,
    )

    with pytest.raises(
        ValueError,
        match="incompatible size",
    ):
        electric_curl(mesh, values)


def test_electric_curl_rejects_wrong_dimension():
    mesh = make_mesh()

    values = np.zeros(
        (2, 2),
        dtype=complex,
    )

    with pytest.raises(
        ValueError,
        match="one-dimensional",
    ):
        electric_curl(mesh, values)


def test_electric_curl_rejects_wrong_mesh_type():
    values = np.zeros(10, dtype=complex)

    with pytest.raises(
        TypeError,
        match="StructuredMesh3D",
    ):
        electric_curl(
            object(),
            values,
        )


# ============================================================================
# magnetic_field
# ============================================================================


def test_magnetic_field_has_face_size():
    mesh = make_mesh()

    n_ex, n_ey, n_ez = n_edge_counts(mesh)

    values = np.zeros(
        n_ex + n_ey + n_ez,
        dtype=complex,
    )

    result = magnetic_field(
        mesh,
        values,
        omega=2.0 * np.pi,
    )

    curl_operator = build_curl(mesh)

    assert result.ndim == 1
    assert result.size == curl_operator.n_faces


def test_magnetic_field_of_zero_electric_field_is_zero():
    mesh = make_mesh()

    n_ex, n_ey, n_ez = n_edge_counts(mesh)

    values = np.zeros(
        n_ex + n_ey + n_ez,
        dtype=complex,
    )

    result = magnetic_field(
        mesh,
        values,
        omega=2.0 * np.pi,
    )

    np.testing.assert_allclose(
        result,
        0.0,
    )


def test_magnetic_field_matches_faraday_law():
    mesh = make_mesh()

    n_ex, n_ey, n_ez = n_edge_counts(mesh)

    values = (
        np.arange(
            n_ex + n_ey + n_ez,
            dtype=float,
        )
        + 2.0j
    )

    omega = 2.0 * np.pi
    mu_r = 1.0
    mu = MU0 * mu_r

    curl_e = electric_curl(
        mesh,
        values,
    )

    h = magnetic_field(
        mesh,
        values,
        omega=omega,
        mu_r=mu_r,
    )

    # Avec exp(+i omega t) :
    #
    #     curl(E) = -i omega mu H
    #
    # donc :
    #
    #     -i omega mu H = curl(E)
    #
    lhs = -1j * omega * mu * h

    np.testing.assert_allclose(
        lhs,
        curl_e,
        rtol=1e-12,
        atol=1e-12,
    )


def test_magnetic_field_accepts_edge_field():
    mesh = make_mesh()

    n_ex, n_ey, n_ez = n_edge_counts(mesh)

    values = (
        np.arange(
            n_ex + n_ey + n_ez,
            dtype=float,
        )
        + 1.0j
    )

    field = build_edge_electric_field(
        mesh,
        values,
    )

    h_from_field = magnetic_field(
        mesh,
        field,
        omega=2.0 * np.pi,
    )

    h_from_vector = magnetic_field(
        mesh,
        values,
        omega=2.0 * np.pi,
    )

    np.testing.assert_allclose(
        h_from_field,
        h_from_vector,
    )


def test_magnetic_field_scales_with_inverse_omega():
    mesh = make_mesh()

    n_ex, n_ey, n_ez = n_edge_counts(mesh)

    values = (
        np.ones(
            n_ex + n_ey + n_ez,
            dtype=complex,
        )
        * (1.0 + 2.0j)
    )

    omega_1 = 2.0 * np.pi
    omega_2 = 4.0 * np.pi

    h1 = magnetic_field(
        mesh,
        values,
        omega=omega_1,
    )

    h2 = magnetic_field(
        mesh,
        values,
        omega=omega_2,
    )

    np.testing.assert_allclose(
        h2,
        h1 / 2.0,
    )


def test_magnetic_field_scales_with_inverse_mu_r():
    mesh = make_mesh()

    n_ex, n_ey, n_ez = n_edge_counts(mesh)

    values = (
        np.arange(
            n_ex + n_ey + n_ez,
            dtype=float,
        )
        + 1.0j
    )

    omega = 2.0 * np.pi

    h1 = magnetic_field(
        mesh,
        values,
        omega=omega,
        mu_r=1.0,
    )

    h2 = magnetic_field(
        mesh,
        values,
        omega=omega,
        mu_r=2.0,
    )

    np.testing.assert_allclose(
        h2,
        h1 / 2.0,
    )


def test_magnetic_field_accepts_cellwise_mu_r():
    mesh = make_mesh()

    n_ex, n_ey, n_ez = n_edge_counts(mesh)

    values = (
        np.arange(
            n_ex + n_ey + n_ez,
            dtype=float,
        )
        + 1.0j
    )

    mu_r = np.ones(
        mesh.n_cells,
        dtype=float,
    )

    result = magnetic_field(
        mesh,
        values,
        omega=2.0 * np.pi,
        mu_r=mu_r,
    )

    reference = magnetic_field(
        mesh,
        values,
        omega=2.0 * np.pi,
        mu_r=1.0,
    )

    np.testing.assert_allclose(
        result,
        reference,
    )


# ============================================================================
# Validation
# ============================================================================


def test_magnetic_field_rejects_zero_frequency():
    mesh = make_mesh()

    n_ex, n_ey, n_ez = n_edge_counts(mesh)

    values = np.zeros(
        n_ex + n_ey + n_ez,
        dtype=complex,
    )

    with pytest.raises(
        ValueError,
        match="strictly positive",
    ):
        magnetic_field(
            mesh,
            values,
            omega=0.0,
        )


def test_magnetic_field_rejects_negative_frequency():
    mesh = make_mesh()

    n_ex, n_ey, n_ez = n_edge_counts(mesh)

    values = np.zeros(
        n_ex + n_ey + n_ez,
        dtype=complex,
    )

    with pytest.raises(
        ValueError,
        match="strictly positive",
    ):
        magnetic_field(
            mesh,
            values,
            omega=-1.0,
        )


def test_magnetic_field_rejects_nonfinite_frequency():
    mesh = make_mesh()

    n_ex, n_ey, n_ez = n_edge_counts(mesh)

    values = np.zeros(
        n_ex + n_ey + n_ez,
        dtype=complex,
    )

    with pytest.raises(
        ValueError,
        match="finite",
    ):
        magnetic_field(
            mesh,
            values,
            omega=np.nan,
        )


def test_magnetic_field_rejects_zero_mu_r():
    mesh = make_mesh()

    n_ex, n_ey, n_ez = n_edge_counts(mesh)

    values = np.zeros(
        n_ex + n_ey + n_ez,
        dtype=complex,
    )

    with pytest.raises(
        ValueError,
        match="strictly positive",
    ):
        magnetic_field(
            mesh,
            values,
            omega=2.0 * np.pi,
            mu_r=0.0,
        )


def test_magnetic_field_rejects_negative_mu_r():
    mesh = make_mesh()

    n_ex, n_ey, n_ez = n_edge_counts(mesh)

    values = np.zeros(
        n_ex + n_ey + n_ez,
        dtype=complex,
    )

    with pytest.raises(
        ValueError,
        match="strictly positive",
    ):
        magnetic_field(
            mesh,
            values,
            omega=2.0 * np.pi,
            mu_r=-1.0,
        )


def test_magnetic_field_rejects_wrong_mu_r_size():
    mesh = make_mesh()

    n_ex, n_ey, n_ez = n_edge_counts(mesh)

    values = np.zeros(
        n_ex + n_ey + n_ez,
        dtype=complex,
    )

    mu_r = np.ones(
        mesh.n_cells + 1,
        dtype=float,
    )

    with pytest.raises(
        ValueError,
        match="incompatible size",
    ):
        magnetic_field(
            mesh,
            values,
            omega=2.0 * np.pi,
            mu_r=mu_r,
        )


def test_magnetic_field_rejects_wrong_mu_r_dimension():
    mesh = make_mesh()

    n_ex, n_ey, n_ez = n_edge_counts(mesh)

    values = np.zeros(
        n_ex + n_ey + n_ez,
        dtype=complex,
    )

    mu_r = np.ones(
        (mesh.n_cells, 1),
        dtype=float,
    )

    with pytest.raises(
        ValueError,
        match="scalar or a one-dimensional",
    ):
        magnetic_field(
            mesh,
            values,
            omega=2.0 * np.pi,
            mu_r=mu_r,
        )


def test_magnetic_field_rejects_nonfinite_mu_r():
    mesh = make_mesh()

    n_ex, n_ey, n_ez = n_edge_counts(mesh)

    values = np.zeros(
        n_ex + n_ey + n_ez,
        dtype=complex,
    )

    mu_r = np.ones(
        mesh.n_cells,
        dtype=float,
    )

    mu_r[0] = np.nan

    with pytest.raises(
        ValueError,
        match="non-finite",
    ):
        magnetic_field(
            mesh,
            values,
            omega=2.0 * np.pi,
            mu_r=mu_r,
        )


def test_magnetic_field_rejects_wrong_mesh_type():
    values = np.zeros(
        10,
        dtype=complex,
    )

    with pytest.raises(
        TypeError,
        match="StructuredMesh3D",
    ):
        magnetic_field(
            object(),
            values,
            omega=2.0 * np.pi,
        )
