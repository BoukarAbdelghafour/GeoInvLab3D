"""
Tests for synthetic gravity inversion.
"""

import numpy as np

from geoinvlab.core.data import DataSet
from geoinvlab.core.inversion import (
    InverseProblem,
    InversionEngine,
)
from geoinvlab.core.mesh import StructuredMesh3D
from geoinvlab.core.model import Model3D
from geoinvlab.methods.gravity import GravityOperator


def test_gravity_inversion_synthetic():
    """
    Test a complete synthetic gravity inversion.

    The synthetic observations are generated from a known
    density model and the inversion is then performed from
    an initial zero model.
    """

    # --------------------------------------------------------------
    # 1. Create a small 3D mesh.
    # --------------------------------------------------------------

    mesh = StructuredMesh3D(
        nx=3,
        ny=3,
        nz=2,
        dx=100.0,
        dy=100.0,
        dz=100.0,
        origin=(0.0, 0.0, -200.0),
    )

    # --------------------------------------------------------------
    # 2. Define observation locations.
    #
    # Observations are located above the model.
    # --------------------------------------------------------------

    observations = np.array(
        [
            [50.0, 50.0, 100.0],
            [150.0, 50.0, 100.0],
            [250.0, 50.0, 100.0],
            [50.0, 150.0, 100.0],
            [150.0, 150.0, 100.0],
            [250.0, 150.0, 100.0],
            [50.0, 250.0, 100.0],
            [150.0, 250.0, 100.0],
            [250.0, 250.0, 100.0],
        ],
        dtype=float,
    )

    # --------------------------------------------------------------
    # 3. Create the gravity operator.
    # --------------------------------------------------------------

    operator = GravityOperator(
        observation_locations=observations,
    )

    # --------------------------------------------------------------
    # 4. Define the true density model.
    #
    # One cell contains a positive density contrast.
    # --------------------------------------------------------------

    true_values = np.zeros(
        mesh.n_cells,
        dtype=float,
    )

    true_values[4] = 500.0

    true_model = Model3D(
        mesh=mesh,
        values=true_values,
        name="true_density",
    )

    # --------------------------------------------------------------
    # 5. Generate synthetic gravity data.
    # --------------------------------------------------------------

    synthetic_values = operator.predict(
        true_model,
    )

    data = DataSet(
        values=synthetic_values,
        locations=observations,
        name="synthetic_gravity",
        units="m/s^2",
    )

    # --------------------------------------------------------------
    # 6. Initial model.
    #
    # Start from zero density contrast.
    # --------------------------------------------------------------

    initial_model = Model3D(
        mesh=mesh,
        values=np.zeros(
            mesh.n_cells,
            dtype=float,
        ),
        name="initial_density",
    )

    # --------------------------------------------------------------
    # 7. Define the inverse problem.
    # --------------------------------------------------------------

    problem = InverseProblem(
        model=initial_model,
        data=data,
        operator=operator,
    )

    # --------------------------------------------------------------
    # 8. Run the inversion.
    # --------------------------------------------------------------

    engine = InversionEngine()

    result = engine.solve(problem)

    recovered_model = result.model

    # --------------------------------------------------------------
    # 9. Verify dimensions.
    # --------------------------------------------------------------

    assert recovered_model.n_parameters == mesh.n_cells

    assert recovered_model.values.shape == (
        mesh.n_cells,
    )

    # --------------------------------------------------------------
    # 10. Verify that the inversion reproduces the data.
    # --------------------------------------------------------------

    predicted = operator.predict(
        recovered_model,
    )

    assert predicted.shape == synthetic_values.shape

    np.testing.assert_allclose(
        predicted,
        synthetic_values,
        rtol=1e-8,
        atol=1e-15,
    )

    # --------------------------------------------------------------
    # 11. Verify that the final data misfit is small.
    # --------------------------------------------------------------

    assert result.data_misfit < 1e-20

    # --------------------------------------------------------------
    # 12. Verify that the objective is consistent.
    # --------------------------------------------------------------

    assert np.isfinite(
        result.objective_value
    )

    assert np.isfinite(
        result.regularization_value
    )

