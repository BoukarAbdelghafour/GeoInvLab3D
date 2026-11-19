"""
Discrete operators used by the 3D magnetotelluric solver.

 MIT License Copyright (c) 17/09/2026 [Boukar Abdelghafour]
"""

from .curl import CurlOperator, build_curl
from .divergence import DivergenceOperator, build_divergence

from .topology import (
EdgeTopology,
FaceTopology,
StructuredTopology,
build_edge_topology,
build_face_topology,
build_topology,
)

from .metric import (
CellMetric,
EdgeMetric,
FaceMetric,
StructuredMetric,
build_metric,
)

from .maxwell import (
MaxwellSystem,
build_maxwell_system,
)

from .boundary import (
BoundaryCondition,
BoundaryEdges,
build_boundary_edges,
make_dirichlet_boundary,
)

from .source import (
MTPlaneWaveSource,
)

from .solver_3d import (
LinearSolveResult,
MaxwellSolver,
solve_maxwell,
)

__all__ = [
# Topology
"EdgeTopology",
"FaceTopology",
"StructuredTopology",
"build_edge_topology",
"build_face_topology",
"build_topology",

# Discrete differential operators
"CurlOperator",
"DivergenceOperator",
"build_curl",
"build_divergence",

# Metric
"EdgeMetric",
"FaceMetric",
"CellMetric",
"StructuredMetric",
"build_metric",

# Maxwell system
"MaxwellSystem",
"build_maxwell_system",

# Boundary conditions
"BoundaryCondition",
"BoundaryEdges",
"build_boundary_edges",
"make_dirichlet_boundary",

# MT source
"MTPlaneWaveSource",

# Linear solver
"LinearSolveResult",
"MaxwellSolver",
"solve_maxwell",


]

