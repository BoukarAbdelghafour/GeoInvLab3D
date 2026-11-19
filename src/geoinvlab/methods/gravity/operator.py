"""
Gravity forward operator.

GeoInvLab3D uses z positive downward, whereas the analytical
gravity kernel uses z positive upward. Coordinates are converted
at the operator/kernel boundary; mesh and station files remain
in the GeoInvLab3D convention.

MIT License Copyright (c) 01/10/2026 [Boukar Abdelghafour]
"""

from dataclasses import dataclass, field

import numpy as np

from geoinvlab.core.model import Model3D
from geoinvlab.core.operators import ForwardOperator
from geoinvlab.methods.gravity.kernel import gravity_kernel


@dataclass
class GravityOperator(ForwardOperator):
    """
    Forward operator for gravity observations.

    Parameters
    ----------
    observation_locations : numpy.ndarray
        Observation coordinates in GeoInvLab3D convention:
        x East, y North, z positive downward, in metres.

    Notes
    -----
    The gravity sensitivity matrix depends on observation
    locations and mesh geometry, but not on model values.
    The Jacobian is cached after its first computation.

    The gravity kernel expects z positive upward. The operator
    therefore changes the sign of the z coordinate for both
    observation locations and cell centres before calling it.
    """

    observation_locations: np.ndarray

    _jacobian: np.ndarray | None = field(
        default=None,
        init=False,
        repr=False,
    )

    _cached_mesh_signature: tuple | None = field(
        default=None,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        """Validate observation locations."""

        self.observation_locations = np.asarray(
            self.observation_locations,
            dtype=float,
        )

        if self.observation_locations.ndim != 2:
            raise ValueError(
                "Observation locations must be two-dimensional."
            )

        if self.observation_locations.shape[1] != 3:
            raise ValueError(
                "Observation locations must have shape (n_data, 3)."
            )

        if not np.all(np.isfinite(self.observation_locations)):
            raise ValueError(
                "Observation locations must contain only finite values."
            )

    @property
    def n_data(self) -> int:
        """Return the number of gravity observations."""
        return self.observation_locations.shape[0]

    @property
    def n_parameters(self) -> int | None:
        """Return the number of parameters if the Jacobian is cached."""
        if self._jacobian is None:
            return None

        return self._jacobian.shape[1]

    def _get_mesh_signature(self, model: Model3D) -> tuple:
        """Return a signature describing the model mesh."""

        mesh = model.mesh

        return (
            mesh.nx,
            mesh.ny,
            mesh.nz,
            mesh.dx,
            mesh.dy,
            mesh.dz,
            tuple(mesh.origin),
        )

    def clear_cache(self) -> None:
        """Clear the cached Jacobian."""
        self._jacobian = None
        self._cached_mesh_signature = None

    def jacobian(self, model: Model3D) -> np.ndarray:
        """
        Compute or retrieve the gravity sensitivity matrix.

        Returns
        -------
        numpy.ndarray
            Sensitivity matrix with shape (n_data, n_parameters).
            The entries are vertical acceleration per unit density
            contrast, in (m/s²)/(kg/m³).
        """

        signature = self._get_mesh_signature(model)

        if (
            self._jacobian is not None
            and self._cached_mesh_signature == signature
        ):
            return self._jacobian

        # Coordinates in GeoInvLab3D: z positive downward.
        centers = np.asarray(
            model.mesh.cell_centers(),
            dtype=float,
        ).copy()

        # The analytical gravity kernel uses z positive upward.
        centers[:, 2] *= -1.0

        kernel_observations = self.observation_locations.copy()
        kernel_observations[:, 2] *= -1.0

        matrix = np.empty(
            (self.n_data, model.n_parameters),
            dtype=float,
        )

        for i, observation in enumerate(kernel_observations):
            for j, center in enumerate(centers):
                matrix[i, j] = gravity_kernel(
                    observation=observation,
                    cell_center=center,
                    dx=model.mesh.dx,
                    dy=model.mesh.dy,
                    dz=model.mesh.dz,
                )

        self._jacobian = matrix
        self._cached_mesh_signature = signature

        return matrix

    def predict(self, model: Model3D) -> np.ndarray:
        """Compute predicted gravity observations in m/s²."""
        return self.jacobian(model) @ model.values
