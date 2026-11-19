import numpy as np

from geoinvlab.methods.mt.operator import MTOperator

observed = np.array([
    100.0, 50.0, 60.0, 80.0,
    179.0, 45.0, -135.0, 170.0,
])

predicted = np.array([
    110.0, 55.0, 65.0, 85.0,
    -179.0, 46.0, -134.0, -170.0,
])

residual = MTOperator.data_residual(
    predicted,
    observed,
)

print("Residual:", residual)


observed = np.array([100., 50., 60., 80., 179., 45., -135., 170.])
predicted = np.array([110., 55., 65., 85., -179., 46., -134., -170.])

residual = MTOperator.data_residual(predicted, observed)

print(residual)
