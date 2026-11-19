"""
MIT License Copyright (c) 17/09/2026 [Boukar Abdelghafour]
"""
import numpy as np

from geoinvlab.core.data import DataSet
from geoinvlab.core.mesh import StructuredMesh3D
from geoinvlab.core.model import Model3D
from geoinvlab.core.operators.linear import LinearOperator


def create_test_model():
    mesh = StructuredMesh3D(
        nx=3,
        ny=1,
        nz=1,
        dx=10.0,
        dy=10.0,
        dz=10.0,
    )

    values = np.zeros(mesh.n_cells)

    return Model3D(
        mesh=mesh,
        values=values,
    )


def create_test_data():
    return DataSet(
        values=np.array([
            1.0,
            2.0,
            3.0,
        ]),
        locations=np.array([
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
        ]),
    )


def create_test_operator():
    matrix = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 3.0],
    ])

    return LinearOperator(matrix)

