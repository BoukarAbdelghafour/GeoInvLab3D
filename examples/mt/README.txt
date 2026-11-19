GeoInvLab3D - Synthetic 3-D MT model
MIT License Copyright (c) 01/10/2026 [Boukar Abdelghafour]
=======================================

MESH
----
Dimensions: 6 x 6 x 4
Number of cells: 144
Cell dimensions: 100.0 x 100.0 x 100.0 m
Origin: (-300.0, -300.0, 0.0)

Coordinates:
    x = East
    y = North
    z = positive downward
    Surface = z = 0 m

RESISTIVITY MODEL
-----------------
Background: 100.0 ohm.m
Conductive anomaly: 10.0 ohm.m
Resistive anomaly: 1000.0 ohm.m

Initial model:
    Homogeneous 100.0 ohm.m

MT ACQUISITION
--------------
Number of stations: 10
Number of frequencies: 10
Number of observations: 100

Frequency range:
    0.01 to 100 Hz

Station elevation:
    z = 0 m

NOISE
-----
Apparent resistivity:
    10.0% multiplicative lognormal noise

Impedance phase:
    3.0 degree Gaussian noise

Random seed:
    20261018

OUTPUT FILES
------------
mesh.dat
model_true.dat
model_initial.dat
stations.dat
frequencies.dat
mt_observations_clean.dat
mt_observations_noisy.dat
mt_uncertainties.dat

OBSERVATION COLUMN ORDER
------------------------
station_id frequency_Hz x_m y_m z_m
rho_xx rho_xy rho_yx rho_yy
phase_xx_deg phase_xy_deg phase_yx_deg phase_yy_deg
