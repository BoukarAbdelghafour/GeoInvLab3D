"""
Tests for the discrete 3D Maxwell operator used by MT.

These tests validate the algebraic/geometrical layer before testing
boundary conditions, sources, or the complete MT forward problem.

The geometry-aware formulation uses:

    C_h = D_face^{-1} C D_edge

for the physical curl, and diagonal dual-volume Hodge matrices for
the conductivity and inverse-permeability terms.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from geoinvlab.core.mesh.structured import StructuredMesh3D
from geoinvlab.methods.mt.discretization.curl import build_curl
from geoinvlab.methods.mt.discretization.field import (
    build_edge_electric_field,
    electric_curl,
    magnetic_field,
)
from geoinvlab.methods.mt.discretization.maxwell import (
    MU0,
    MaxwellSystem,
    _cell_to_edge_conductivity,
    _cell_to_face_property,
    build_maxwell_system,
)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def make_mesh(
    cell_widths_x: list[float],
    cell_widths_y: list[float],
    cell_widths_z: list[float],
) -> StructuredMesh3D:
    """
    Build a StructuredMesh3D from uniform cell-width specifications.

    StructuredMesh3D currently accepts one scalar cell size per axis,
    so every list must contain identical values.
    """

    widths_x = np.asarray(cell_widths_x, dtype=float)
    widths_y = np.asarray(cell_widths_y, dtype=float)
    widths_z = np.asarray(cell_widths_z, dtype=float)

    if widths_x.ndim != 1 or widths_x.size == 0:
        raise ValueError(
            "cell_widths_x must be a non-empty one-dimensional array."
        )

    if widths_y.ndim != 1 or widths_y.size == 0:
        raise ValueError(
            "cell_widths_y must be a non-empty one-dimensional array."
        )

    if widths_z.ndim != 1 or widths_z.size == 0:
        raise ValueError(
            "cell_widths_z must be a non-empty one-dimensional array."
        )

    if not np.allclose(widths_x, widths_x[0]):
        raise ValueError(
            "StructuredMesh3D currently requires uniform X cell widths."
        )

    if not np.allclose(widths_y, widths_y[0]):
        raise ValueError(
            "StructuredMesh3D currently requires uniform Y cell widths."
        )

    if not np.allclose(widths_z, widths_z[0]):
        raise ValueError(
            "StructuredMesh3D currently requires uniform Z cell widths."
        )

    return StructuredMesh3D(
        nx=int(widths_x.size),
        ny=int(widths_y.size),
        nz=int(widths_z.size),
        dx=float(widths_x[0]),
        dy=float(widths_y[0]),
        dz=float(widths_z[0]),
    )


def _edge_dual_volumes(
    mesh: StructuredMesh3D,
) -> np.ndarray:
    """
    Return the dual volume associated with every global edge.

    Each primal cell contributes one quarter of its volume to each
    of its 12 edges.

    Global ordering:

        [Ex, Ey, Ez]
    """

    nx, ny, nz = mesh.shape
    cell_volumes = mesh.cell_volumes().reshape(nx, ny, nz)

    n_ex = nx * (ny + 1) * (nz + 1)
    n_ey = (nx + 1) * ny * (nz + 1)
    n_ez = (nx + 1) * (ny + 1) * nz

    dual_ex = np.zeros(n_ex, dtype=float)
    dual_ey = np.zeros(n_ey, dtype=float)
    dual_ez = np.zeros(n_ez, dtype=float)

    # --------------------------------------------------------------
    # Ex edges
    #
    # An Ex edge is shared by cells in:
    #   y = iy-1, iy
    #   z = iz-1, iz
    # --------------------------------------------------------------

    def ex_index(
        ix: int,
        iy: int,
        iz: int,
    ) -> int:
        return (
            (ix * (ny + 1) + iy) * (nz + 1)
            + iz
        )

    for ix in range(nx):
        for iy in range(ny + 1):
            for iz in range(nz + 1):
                value = 0.0

                for dy in (-1, 0):
                    for dz in (-1, 0):
                        cy = iy + dy
                        cz = iz + dz

                        if (
                            0 <= cy < ny
                            and 0 <= cz < nz
                        ):
                            value += 0.25 * cell_volumes[
                                ix,
                                cy,
                                cz,
                            ]

                dual_ex[
                    ex_index(ix, iy, iz)
                ] = value

    # --------------------------------------------------------------
    # Ey edges
    # --------------------------------------------------------------

    offset_ey = n_ex

    def ey_index(
        ix: int,
        iy: int,
        iz: int,
    ) -> int:
        return offset_ey + (
            (ix * ny + iy) * (nz + 1)
            + iz
        )

    for ix in range(nx + 1):
        for iy in range(ny):
            for iz in range(nz + 1):
                value = 0.0

                for dx in (-1, 0):
                    for dz in (-1, 0):
                        cx = ix + dx
                        cz = iz + dz

                        if (
                            0 <= cx < nx
                            and 0 <= cz < nz
                        ):
                            value += 0.25 * cell_volumes[
                                cx,
                                iy,
                                cz,
                            ]

                dual_ey[
                    ey_index(ix, iy, iz) - offset_ey
                ] = value

    # --------------------------------------------------------------
    # Ez edges
    # --------------------------------------------------------------

    offset_ez = n_ex + n_ey

    def ez_index(
        ix: int,
        iy: int,
        iz: int,
    ) -> int:
        return offset_ez + (
            (ix * (ny + 1) + iy) * nz
            + iz
        )

    for ix in range(nx + 1):
        for iy in range(ny + 1):
            for iz in range(nz):
                value = 0.0

                for dx in (-1, 0):
                    for dy in (-1, 0):
                        cx = ix + dx
                        cy = iy + dy

                        if (
                            0 <= cx < nx
                            and 0 <= cy < ny
                        ):
                            value += 0.25 * cell_volumes[
                                cx,
                                cy,
                                iz,
                            ]

                dual_ez[
                    ez_index(ix, iy, iz) - offset_ez
                ] = value

    return np.concatenate(
        (
            dual_ex,
            dual_ey,
            dual_ez,
        )
    )


def _face_dual_volumes(
    mesh: StructuredMesh3D,
) -> np.ndarray:
    """
    Return the dual volume associated with every global face.

    Each adjacent primal cell contributes one half of its volume
    to the corresponding face.

    Global ordering:

        [Fx, Fy, Fz]
    """

    nx, ny, nz = mesh.shape
    cell_volumes = mesh.cell_volumes().reshape(nx, ny, nz)

    n_fx = (nx + 1) * ny * nz
    n_fy = nx * (ny + 1) * nz

    dual_fx = np.zeros(n_fx, dtype=float)
    dual_fy = np.zeros(n_fy, dtype=float)
    dual_fz = np.zeros(nx * ny * (nz + 1), dtype=float)

    # --------------------------------------------------------------
    # Fx
    # --------------------------------------------------------------

    def fx_index(
        ix: int,
        iy: int,
        iz: int,
    ) -> int:
        return (
            (ix * ny + iy) * nz
            + iz
        )

    for ix in range(nx + 1):
        for iy in range(ny):
            for iz in range(nz):
                value = 0.0

                if ix > 0:
                    value += 0.5 * cell_volumes[
                        ix - 1,
                        iy,
                        iz,
                    ]

                if ix < nx:
                    value += 0.5 * cell_volumes[
                        ix,
                        iy,
                        iz,
                    ]

                dual_fx[
                    fx_index(ix, iy, iz)
                ] = value

    # --------------------------------------------------------------
    # Fy
    # --------------------------------------------------------------

    offset_fy = n_fx

    def fy_index(
        ix: int,
        iy: int,
        iz: int,
    ) -> int:
        return offset_fy + (
            (ix * (ny + 1) + iy) * nz
            + iz
        )

    for ix in range(nx):
        for iy in range(ny + 1):
            for iz in range(nz):
                value = 0.0

                if iy > 0:
                    value += 0.5 * cell_volumes[
                        ix,
                        iy - 1,
                        iz,
                    ]

                if iy < ny:
                    value += 0.5 * cell_volumes[
                        ix,
                        iy,
                        iz,
                    ]

                dual_fy[
                    fy_index(ix, iy, iz) - offset_fy
                ] = value

    # --------------------------------------------------------------
    # Fz
    # --------------------------------------------------------------

    offset_fz = n_fx + n_fy

    def fz_index(
        ix: int,
        iy: int,
        iz: int,
    ) -> int:
        return offset_fz + (
            (ix * ny + iy) * (nz + 1)
            + iz
        )

    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz + 1):
                value = 0.0

                if iz > 0:
                    value += 0.5 * cell_volumes[
                        ix,
                        iy,
                        iz - 1,
                    ]

                if iz < nz:
                    value += 0.5 * cell_volumes[
                        ix,
                        iy,
                        iz,
                    ]

                dual_fz[
                    fz_index(ix, iy, iz) - offset_fz
                ] = value

    return np.concatenate(
        (
            dual_fx,
            dual_fy,
            dual_fz,
        )
    )


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


@pytest.fixture
def mesh() -> StructuredMesh3D:
    """
    Small unpadded mesh used for deterministic discretization tests.
    """
    return StructuredMesh3D(
        nx=2,
        ny=2,
        nz=2,
        dx=100.0,
        dy=100.0,
        dz=100.0,
    )


@pytest.fixture
def conductivity(
    mesh: StructuredMesh3D,
) -> np.ndarray:
    """
    Homogeneous conductivity in S/m.
    """
    return np.full(
        mesh.n_cells,
        1.0e-2,
        dtype=float,
    )


@pytest.fixture
def system(
    mesh: StructuredMesh3D,
    conductivity: np.ndarray,
) -> MaxwellSystem:
    """
    Homogeneous isotropic Maxwell system.
    """
    return build_maxwell_system(
        mesh,
        conductivity,
        mu_r=1.0,
    )


# ----------------------------------------------------------------------
# Topology
# ----------------------------------------------------------------------


def test_curl_dimensions(
    mesh: StructuredMesh3D,
) -> None:
    """
    The curl operator must map edge DOFs to face DOFs.
    """
    curl = build_curl(mesh)

    expected_n_edges = (
        mesh.nx * (mesh.ny + 1) * (mesh.nz + 1)
        + (mesh.nx + 1) * mesh.ny * (mesh.nz + 1)
        + (mesh.nx + 1) * (mesh.ny + 1) * mesh.nz
    )

    expected_n_faces = (
        (mesh.nx + 1) * mesh.ny * mesh.nz
        + mesh.nx * (mesh.ny + 1) * mesh.nz
        + mesh.nx * mesh.ny * (mesh.nz + 1)
    )

    assert curl.n_edges == expected_n_edges
    assert curl.n_faces == expected_n_faces

    assert curl.matrix.shape == (
        expected_n_faces,
        expected_n_edges,
    )


def test_curl_incidence_entries_are_signed(
    mesh: StructuredMesh3D,
) -> None:
    """
    The topological curl incidence matrix contains only -1, 0 and +1.
    """
    curl = build_curl(mesh)

    values = np.unique(
        np.asarray(
            curl.matrix.data,
            dtype=float,
        )
    )

    assert np.all(
        np.isin(
            values,
            np.array([-1.0, 1.0]),
        )
    )


# ----------------------------------------------------------------------
# Material averaging
# ----------------------------------------------------------------------


def test_homogeneous_conductivity_is_preserved(
    mesh: StructuredMesh3D,
) -> None:
    """
    A homogeneous cell conductivity must remain homogeneous on edges.
    """
    sigma0 = 2.5e-2

    conductivity = np.full(
        mesh.n_cells,
        sigma0,
        dtype=float,
    )

    sigma_edge = _cell_to_edge_conductivity(
        mesh,
        conductivity,
    )

    assert sigma_edge.shape == (
        build_curl(mesh).n_edges,
    )

    assert np.allclose(
        sigma_edge,
        sigma0,
    )


def test_homogeneous_face_property_is_preserved(
    mesh: StructuredMesh3D,
) -> None:
    """
    A homogeneous cell property must remain homogeneous on faces.
    """
    value0 = 7.5

    values = np.full(
        mesh.n_cells,
        value0,
        dtype=float,
    )

    face_values = _cell_to_face_property(
        mesh,
        values,
    )

    assert face_values.shape == (
        build_curl(mesh).n_faces,
    )

    assert np.allclose(
        face_values,
        value0,
    )


def test_face_property_averages_internal_faces(
    mesh: StructuredMesh3D,
) -> None:
    """
    Internal faces must receive the arithmetic mean of adjacent cells.
    """
    values = np.arange(
        mesh.n_cells,
        dtype=float,
    )

    faces = _cell_to_face_property(
        mesh,
        values,
    )

    nx, ny, nz = mesh.shape
    field = values.reshape(nx, ny, nz)

    ix = 1
    iy = 0
    iz = 0

    index = (
        (ix * ny + iy) * nz
        + iz
    )

    expected = 0.5 * (
        field[ix - 1, iy, iz]
        + field[ix, iy, iz]
    )

    assert faces[index] == pytest.approx(expected)


# ----------------------------------------------------------------------
# Maxwell material matrices
# ----------------------------------------------------------------------


def test_conductivity_mass_is_diagonal(
    system: MaxwellSystem,
) -> None:
    """
    The conductivity matrix is diagonal in the edge basis.
    """
    mass = system.conductivity_mass()

    assert mass.shape == (
        system.n_edges,
        system.n_edges,
    )

    assert mass.format == "csr"

    assert np.allclose(
        mass.toarray(),
        np.diag(mass.diagonal()),
    )

    assert np.all(
        mass.diagonal() > 0.0,
    )


def test_conductivity_mass_contains_edge_dual_volume(
    mesh: StructuredMesh3D,
) -> None:
    """
    For homogeneous conductivity:

        M_sigma[e,e] = sigma * V*_e.
    """
    sigma = 2.0

    system = build_maxwell_system(
        mesh,
        conductivity=np.full(
            mesh.n_cells,
            sigma,
        ),
    )

    expected = (
        sigma
        * _edge_dual_volumes(mesh)
    )

    assert np.allclose(
        system.conductivity_mass().diagonal(),
        expected,
    )


def test_inverse_permeability_mass_is_diagonal(
    system: MaxwellSystem,
) -> None:
    """
    The inverse-permeability face matrix is diagonal.
    """
    mass = system.permeability_mass_inverse()

    assert mass.shape == (
        system.n_faces,
        system.n_faces,
    )

    assert mass.format == "csr"

    assert np.allclose(
        mass.toarray(),
        np.diag(mass.diagonal()),
    )

    assert np.all(
        mass.diagonal() > 0.0,
    )


def test_inverse_permeability_mass_contains_face_dual_volume(
    mesh: StructuredMesh3D,
) -> None:
    """
    For homogeneous relative permeability:

        M_mu_inv[f,f] = V*_f / mu.
    """
    system = build_maxwell_system(
        mesh,
        conductivity=np.ones(mesh.n_cells),
        mu_r=1.0,
    )

    mu_inv = 1.0 / MU0

    expected = (
        _face_dual_volumes(mesh)
        * mu_inv
    )

    assert np.allclose(
        system.permeability_mass_inverse().diagonal(),
        expected,
    )


def test_stiffness_matrix_is_real_symmetric(
    system: MaxwellSystem,
) -> None:
    """
    K = C_h.T M_mu_inv C_h must be real symmetric.
    """
    stiffness = system.stiffness_matrix()

    matrix = stiffness.toarray()

    assert matrix.shape == (
        system.n_edges,
        system.n_edges,
    )

    assert np.isrealobj(matrix)

    assert np.allclose(
        matrix,
        matrix.T,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_stiffness_matrix_is_positive_semidefinite(
    system: MaxwellSystem,
) -> None:
    """
    K = C_h.T M_mu_inv C_h must satisfy

        x.T K x >= 0

    for real vectors x.
    """
    stiffness = system.stiffness_matrix()

    rng = np.random.default_rng(12345)

    for _ in range(10):
        x = rng.normal(
            size=system.n_edges,
        )

        quadratic_form = float(
            x @ (stiffness @ x)
        )

        assert quadratic_form >= -1.0e-10


# ----------------------------------------------------------------------
# Geometry-aware curl
# ----------------------------------------------------------------------


def test_physical_curl_contains_edge_and_face_geometry(
    mesh: StructuredMesh3D,
) -> None:
    """
    Verify

        C_h = D_A^{-1} C D_l.
    """
    system = build_maxwell_system(
        mesh,
        conductivity=np.ones(mesh.n_cells),
    )

    C = system.curl.matrix
    C_geom = system.physical_curl_matrix()

    expected = (
        sparse.diags(
            1.0 / system.metric.faces.areas,
        )
        @ C
        @ sparse.diags(
            system.metric.edges.lengths,
        )
    )

    assert np.allclose(
        C_geom.toarray(),
        expected.toarray(),
    )


def test_geometry_changes_physical_curl() -> None:
    """
    Changing the physical mesh dimensions must change C_h.
    """
    mesh_1 = make_mesh(
        cell_widths_x=[1.0],
        cell_widths_y=[1.0],
        cell_widths_z=[1.0],
    )

    mesh_2 = make_mesh(
        cell_widths_x=[2.0],
        cell_widths_y=[1.0],
        cell_widths_z=[1.0],
    )

    system_1 = build_maxwell_system(
        mesh_1,
        conductivity=np.ones(mesh_1.n_cells),
    )

    system_2 = build_maxwell_system(
        mesh_2,
        conductivity=np.ones(mesh_2.n_cells),
    )

    assert not np.allclose(
        system_1.physical_curl_matrix().toarray(),
        system_2.physical_curl_matrix().toarray(),
    )


# ----------------------------------------------------------------------
# Frequency-domain Maxwell matrix
# ----------------------------------------------------------------------


def test_system_matrix_has_correct_shape(
    system: MaxwellSystem,
) -> None:
    """
    A(omega) must act on the complete edge electric field.
    """
    omega = 2.0 * np.pi

    matrix = system.system_matrix(omega)

    assert matrix.shape == (
        system.n_edges,
        system.n_edges,
    )

    assert matrix.format == "csr"
    assert np.iscomplexobj(matrix.data)


def test_system_matrix_has_expected_real_and_imaginary_parts(
    system: MaxwellSystem,
) -> None:
    """
    Verify

        A = K + i omega M_sigma.
    """
    omega = 2.0 * np.pi * 10.0

    stiffness = system.stiffness_matrix()
    conductivity_mass = system.conductivity_mass()
    matrix = system.system_matrix(omega)

    assert np.allclose(
        matrix.toarray().real,
        stiffness.toarray(),
        rtol=1.0e-12,
        atol=1.0e-12,
    )

    assert np.allclose(
        matrix.toarray().imag,
        omega * conductivity_mass.toarray(),
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_system_matrix_scales_linearly_with_frequency(
    system: MaxwellSystem,
) -> None:
    """
    The imaginary material contribution must scale linearly with omega.
    """
    omega1 = 2.0 * np.pi * 1.0
    omega2 = 2.0 * np.pi * 10.0

    a1 = system.system_matrix(omega1).toarray()
    a2 = system.system_matrix(omega2).toarray()

    assert np.allclose(
        a1.real,
        a2.real,
        rtol=1.0e-12,
        atol=1.0e-12,
    )

    assert np.allclose(
        a2.imag,
        10.0 * a1.imag,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


# ----------------------------------------------------------------------
# Field utilities
# ----------------------------------------------------------------------


def test_electric_curl_has_face_dofs(
    mesh: StructuredMesh3D,
) -> None:
    """
    curl(E) must contain one value per face DOF.
    """
    curl = build_curl(mesh)

    rng = np.random.default_rng(123)

    values = (
        rng.normal(size=curl.n_edges)
        + 1j * rng.normal(size=curl.n_edges)
    )

    field = build_edge_electric_field(
        mesh,
        values,
    )

    curl_e = electric_curl(
        mesh,
        field,
    )

    assert curl_e.shape == (
        curl.n_faces,
    )

    assert np.all(
        np.isfinite(curl_e.real),
    )

    assert np.all(
        np.isfinite(curl_e.imag),
    )


def test_magnetic_field_satisfies_discrete_faraday_relation(
    mesh: StructuredMesh3D,
) -> None:
    """
    Verify the implemented convention

        curl(E) = -i omega mu H.

    Equivalently:

        H = i/(omega mu) curl(E).
    """
    curl = build_curl(mesh)

    rng = np.random.default_rng(54321)

    values = (
        rng.normal(size=curl.n_edges)
        + 1j * rng.normal(size=curl.n_edges)
    )

    field = build_edge_electric_field(
        mesh,
        values,
    )

    omega = 2.0 * np.pi * 5.0
    mu_r = 1.0

    curl_e = electric_curl(
        mesh,
        field,
    )

    magnetic = magnetic_field(
        mesh,
        field,
        omega,
        mu_r=mu_r,
    )

    reconstructed_curl = (
        -1j
        * omega
        * MU0
        * mu_r
        * magnetic
    )

    assert np.allclose(
        reconstructed_curl,
        curl_e,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
 

