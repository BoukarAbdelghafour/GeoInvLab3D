import numpy as np

from geoinvlab.core.data import DataSet
from geoinvlab.core.inversion import (
    InverseProblem,
    LinearLeastSquaresSolver,
)
from geoinvlab.core.mesh import StructuredMesh3D
from geoinvlab.core.model import Model3D
from geoinvlab.core.operators.linear import LinearOperator
from geoinvlab.project import Project



def test_complete_project_workflow():

    # 1. Mesh
    mesh = StructuredMesh3D(
        nx=2,
        ny=1,
        nz=1,
        dx=1.0,
        dy=1.0,
        dz=1.0,
    )

    # 2. Initial model
    model = Model3D(
        mesh=mesh,
        values=[0.0, 0.0],
    )

    # 3. Forward operator
    matrix = np.array([
        [2.0, 1.0],
        [1.0, 3.0],
    ])

    operator = LinearOperator(matrix)

    # 4. Synthetic observations
    true_model = np.array([2.0, 4.0])

    data = DataSet(
        values=matrix @ true_model,
        locations=[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
    )

    # 5. Inverse problem
    problem = InverseProblem(
        model=model,
        data=data,
        operator=operator,
    )

    # 6. Project
    project = Project(
        name="Integration Test",
        dataset=data,
        mesh=mesh,
        model=model,
        operator=operator,
        inverse_problem=problem,
    )

    assert project.name == "Integration Test"
    assert project.dataset is data
    assert project.mesh is mesh
    assert project.model is model
    assert project.operator is operator
    assert project.inverse_problem is problem

    # 7. Solve
    solver = LinearLeastSquaresSolver()

    result = solver.solve(problem)

    # 8. Store result in project
    project.result = result

    assert project.result is result

    # 9. Verify inversion
    assert np.allclose(
        project.result.values,
        true_model,
    )
"""
MIT License Copyright (c) 17/09/2026 [Boukar Abdelghafour]
"""
