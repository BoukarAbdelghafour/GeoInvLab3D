"""
Tests for the high-level MT inversion method.

The tests in this module focus on the integration layer provided by
MTMethod:

    dataset
        -> station/frequency mapping
        -> 3D MT forward solver
        -> impedance
        -> apparent resistivity / phase
        -> flattened observation vector
        -> residuals and weights

The 3D physical solver itself is tested separately in
test_mt_forward.py and test_mt_3d_physics.py.

Important
---------
The homogeneous dataset used below is an analytical 1D half-space
reference, whereas MTMethod.forward() uses a finite 3D mesh.

Consequently, exact equality between the two is not expected. The
comparison tolerance accounts for the finite-domain / finite-mesh
numerical approximation.
"""

import numpy as np
import pytest

from geoinvlab.core.data.mt_dataset import MTDataSet
from geoinvlab.core.mesh.structured import StructuredMesh3D
from geoinvlab.core.model.model import Model3D
from geoinvlab.methods.mt.method import MTMethod


def _make_mesh():
    """
    Build the small 3D mesh used by the method tests.
    """
    return StructuredMesh3D(
        nx=3,
        ny=3,
        nz=20,
        dx=100.0,
        dy=100.0,
        dz=100.0,
    )


def _make_model(
    mesh,
    resistivity: float = 100.0,
):
    """
    Build a homogeneous resistivity model.
    """
    return Model3D(
        mesh=mesh,
        values=np.full(
            mesh.n_cells,
            resistivity,
            dtype=float,
        ),
        name="resistivity",
    )


def _make_dataset():
    """
    Build a homogeneous 1D half-space reference dataset.

    There is one station and three frequencies:

        0.01 Hz
        0.1 Hz
        1.0 Hz

    For a 100 ohm.m homogeneous half-space, the ideal response is:

        rho_xy = rho_yx = 100 ohm.m
        phase_xy = +45 deg
        phase_yx = -135 deg

    The finite 3D solver is expected to reproduce this response
    approximately.
    """
    frequencies = np.array(
        [
            0.01,
            0.1,
            1.0,
        ],
        dtype=float,
    )

    locations = np.array(
        [
            [150.0, 150.0, 0.0],
            [150.0, 150.0, 0.0],
            [150.0, 150.0, 0.0],
        ],
        dtype=float,
    )

    n = frequencies.size

    rho_xx = np.zeros(
        n,
        dtype=float,
    )

    rho_xy = np.full(
        n,
        100.0,
        dtype=float,
    )

    rho_yx = np.full(
        n,
        100.0,
        dtype=float,
    )

    rho_yy = np.zeros(
        n,
        dtype=float,
    )

    phase_xx = np.zeros(
        n,
        dtype=float,
    )

    phase_xy = np.full(
        n,
        45.0,
        dtype=float,
    )

    phase_yx = np.full(
        n,
        -135.0,
        dtype=float,
    )

    phase_yy = np.zeros(
        n,
        dtype=float,
    )

    return MTDataSet(
        locations=locations,
        frequencies=frequencies,
        rho_xx=rho_xx,
        rho_xy=rho_xy,
        rho_yx=rho_yx,
        rho_yy=rho_yy,
        phase_xx=phase_xx,
        phase_xy=phase_xy,
        phase_yx=phase_yx,
        phase_yy=phase_yy,
    )


def test_mt_method_initialization():
    """
    MTMethod initializes correctly and exposes the expected
    dataset geometry.
    """
    mesh = _make_mesh()
    dataset = _make_dataset()

    method = MTMethod(
        dataset=dataset,
        mesh=mesh,
    )

    assert method.dataset is dataset
    assert method.mesh is mesh

    assert method.frequencies.shape == (3,)
    np.testing.assert_allclose(
        method.frequencies,
        np.array(
            [
                0.01,
                0.1,
                1.0,
            ],
            dtype=float,
        ),
    )

    assert method.locations.shape == (1, 3)

    np.testing.assert_allclose(
        method.locations[0],
        np.array(
            [
                150.0,
                150.0,
                0.0,
            ],
            dtype=float,
        ),
    )


def test_mt_method_forward_shape():
    """
    MTMethod.forward() returns one flattened value vector with
    eight values per observation.
    """
    mesh = _make_mesh()

    model = _make_model(
        mesh,
        resistivity=100.0,
    )

    dataset = _make_dataset()

    method = MTMethod(
        dataset=dataset,
        mesh=mesh,
    )

    predicted = method.forward(
        model
    )

    assert isinstance(
        predicted,
        np.ndarray,
    )

    assert predicted.shape == (
        dataset.n_values,
    )

    assert predicted.shape == (
        24,
    )

    assert np.all(
        np.isfinite(predicted)
    )


def test_mt_method_forward_matches_dataset_for_homogeneous_model():
    """
    A homogeneous 100 ohm.m model reproduces the homogeneous
    analytical MT response within a physically meaningful numerical
    tolerance.

    The dataset is a 1D half-space reference, while the forward
    calculation is performed on a finite 3D mesh. Exact equality is
    therefore not expected.

    The present finite-mesh calculation differs from the analytical
    reference by about 2 percent at the highest frequency, which is
    acceptable for this integration-level test.
    """
    mesh = _make_mesh()

    model = _make_model(
        mesh,
        resistivity=100.0,
    )

    dataset = _make_dataset()

    method = MTMethod(
        dataset=dataset,
        mesh=mesh,
    )

    predicted = method.forward(
        model
    )

    observed = dataset.values

    np.testing.assert_allclose(
        predicted,
        observed,
        rtol=3.0e-2,
        atol=1.0e-6,
    )


def test_mt_method_forward_impedance_shape():
    """
    The impedance tensor has shape

        (n_stations, n_frequencies, 2, 2).
    """
    mesh = _make_mesh()

    model = _make_model(
        mesh,
        resistivity=100.0,
    )

    dataset = _make_dataset()

    method = MTMethod(
        dataset=dataset,
        mesh=mesh,
    )

    impedance = method.forward_impedance(
        model
    )

    assert isinstance(
        impedance,
        np.ndarray,
    )

    assert impedance.shape == (
        1,
        3,
        2,
        2,
    )

    assert np.all(
        np.isfinite(impedance.real)
    )

    assert np.all(
        np.isfinite(impedance.imag)
    )


def test_mt_method_impedance_reciprocity():
    """
    The homogeneous response must satisfy

        Zyx = -Zxy

    for the chosen source-column convention.
    """
    mesh = _make_mesh()

    model = _make_model(
        mesh,
        resistivity=100.0,
    )

    dataset = _make_dataset()

    method = MTMethod(
        dataset=dataset,
        mesh=mesh,
    )

    impedance = method.forward_impedance(
        model
    )

    zxy = impedance[
        0,
        :,
        0,
        1,
    ]

    zyx = impedance[
        0,
        :,
        1,
        0,
    ]

    np.testing.assert_allclose(
        zyx,
        -zxy,
        rtol=1.0e-10,
        atol=1.0e-12,
    )


def test_mt_method_residual_is_small_for_matching_homogeneous_model():
    """
    The residual remains small when a homogeneous 100 ohm.m model is
    compared with the corresponding homogeneous analytical reference.

    Because the forward response is computed numerically on a finite
    3D mesh while the dataset is an analytical half-space reference,
    the residual is not expected to be exactly zero.

    The largest observed discrepancy for this mesh is approximately
    2.01 ohm.m, so a 3 ohm.m bound provides a meaningful regression
    test without incorrectly requiring analytical equality from the
    finite 3D solver.
    """
    mesh = _make_mesh()

    model = _make_model(
        mesh,
        resistivity=100.0,
    )

    dataset = _make_dataset()

    method = MTMethod(
        dataset=dataset,
        mesh=mesh,
    )

    predicted = method.forward(
        model
    )

    residual = method.residual(
        predicted,
        weighted=False,
    )

    assert residual.shape == (
        dataset.n_values,
    )

    assert np.all(
        np.isfinite(residual)
    )

    assert np.max(
        np.abs(residual)
    ) < 3.0


def test_mt_method_residual_uses_weights():
    """
    Explicit uncertainties must be applied component-wise.

    The second component of the first observation is rho_xy.
    Its uncertainty is set to 2, so the residual must be divided
    by 2.
    """
    mesh = _make_mesh()

    model = _make_model(
        mesh,
        resistivity=100.0,
    )

    dataset = _make_dataset()

    uncertainties = np.ones(
        (
            dataset.n_data,
            8,
        ),
        dtype=float,
    )

    uncertainties[0, 1] = 2.0

    dataset.uncertainties = uncertainties

    dataset._validate()

    method = MTMethod(
        dataset=dataset,
        mesh=mesh,
    )

    predicted = method.forward(
        model
    )

    residual = method.residual(
        predicted,
        weighted=True,
    )

    expected = (
        predicted[1]
        - dataset.values[1]
    ) / 2.0

    assert residual[1] == pytest.approx(
        expected,
        rel=1.0e-12,
        abs=1.0e-12,
    )

    # The next component has unit uncertainty and must therefore
    # remain unscaled.
    expected_next = (
        predicted[2]
        - dataset.values[2]
    )

    assert residual[2] == pytest.approx(
        expected_next,
        rel=1.0e-12,
        abs=1.0e-12,
    )


def test_mt_method_rejects_wrong_prediction_size():
    """
    A prediction vector with the wrong number of values is rejected.
    """
    mesh = _make_mesh()
    dataset = _make_dataset()

    method = MTMethod(
        dataset=dataset,
        mesh=mesh,
    )

    wrong_prediction = np.zeros(
        dataset.n_values - 1,
        dtype=float,
    )

    with pytest.raises(
        ValueError
    ):
        method.residual(
            wrong_prediction
        )


def test_mt_method_rejects_invalid_model_type():
    """
    MTMethod.forward() requires a Model3D instance.
    """
    mesh = _make_mesh()
    dataset = _make_dataset()

    method = MTMethod(
        dataset=dataset,
        mesh=mesh,
    )

    with pytest.raises(
        TypeError
    ):
        method.forward(
            np.ones(
                mesh.n_cells,
                dtype=float,
            )
        )


def test_mt_method_rejects_wrong_model_cell_count():
    """
    A model defined on a different mesh must be rejected.
    """
    mesh = _make_mesh()

    other_mesh = StructuredMesh3D(
        nx=2,
        ny=2,
        nz=2,
        dx=100.0,
        dy=100.0,
        dz=100.0,
    )

    dataset = _make_dataset()

    method = MTMethod(
        dataset=dataset,
        mesh=mesh,
    )

    model = _make_model(
        other_mesh,
        resistivity=100.0,
    )

    with pytest.raises(
        ValueError
    ):
        method.forward(
            model
        )
 

