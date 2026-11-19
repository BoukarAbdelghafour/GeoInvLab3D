# GeoInvLab3D

**An open-source Python framework for three-dimensional geophysical modelling and inversion**

GeoInvLab3D is a modular Python framework for 3-D geophysical forward modelling and inversion. It combines reusable data structures, structured-mesh and physical-property models, method-specific forward operators, inversion components, and a graphical user interface.

The framework provides a common computational workflow for three geophysical methods:

- **Gravity** — modelling and inversion of subsurface density contrasts.
- **Magnetic** — modelling and inversion of magnetic responses associated with magnetic susceptibility distributions.
- **Magnetotelluric (MT)** — frequency-domain electromagnetic modelling and inversion using electrical resistivity.

The graphical application provides tools to organize projects, import observations, create or load meshes and models, configure inversion runs, and inspect results.

> **Project status:** version `0.1.0`. Features and APIs may evolve as the framework develops.

---

## Contents

- [Main capabilities](#main-capabilities)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Launch the graphical interface](#launch-the-graphical-interface)
- [Typical workflow](#typical-workflow)
- [Mesh and model data](#mesh-and-model-data)
- [Geophysical methods](#geophysical-methods)
- [Development and tests](#development-and-tests)
- [Project structure](#project-structure)
- [License](#license)
- [Citation](#citation)

---

## Main capabilities

- Common 3-D data, mesh, and physical-property model structures.
- Structured Cartesian meshes with optional padding support.
- Gravity, magnetic, and magnetotelluric modelling and inversion workflows.
- Method-specific forward operators connected to shared inversion components.
- Data-misfit and model-regularization configuration.
- Gauss–Newton inversion workflows.
- Import and export utilities for geophysical observations and model data.
- A desktop graphical user interface based on PySide6.
- Optional 3-D visualization using PyVista and PyVistaQt.
- Synthetic examples and automated tests.

## Architecture

GeoInvLab3D separates shared framework components from method-specific physical formulations.

At a high level, the framework contains:

1. **Core data structures** — observations, meshes, and physical-property models.
2. **Geophysical methods** — gravity, magnetic, and MT forward-modelling components.
3. **Inversion framework** — data misfit, regularization, optimization settings, and inversion solvers.
4. **I/O components** — data, mesh, and model import/export.
5. **Graphical interface** — project management, data tables, mesh/model inspection, inversion configuration, and results.
6. **Visualization** — 2-D plots and optional interactive 3-D views.

This organization allows method-specific forward problems to use common mesh, model, and inversion infrastructure.

## Requirements

- Python **3.10** or later.
- Core dependencies: NumPy, SciPy, and Matplotlib.

Optional components:

- **GUI:** PySide6
- **3-D visualization:** PyVista and PyVistaQt
- **Development and testing:** pytest, pytest-cov, pytest-qt, and Ruff

## Installation

Clone the repository and move into the project directory:

```bash
git clone <REPOSITORY_URL>
cd GeoInvLab3D
```

Create and activate a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```powershell
.venv\Scripts\activate
```

Upgrade pip and install the package.

### Core framework

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

### Graphical interface

```bash
python -m pip install -e ".[gui]"
```

### GUI with interactive 3-D visualization

```bash
python -m pip install -e ".[gui,visualization]"
```

### Full development installation

This option installs the GUI, visualization dependencies, and development tools:

```bash
python -m pip install -e ".[all]"
```

The optional dependency groups are defined in `pyproject.toml`.

## Launch the graphical interface

After installing the GUI extra, launch GeoInvLab3D with:

```bash
geoinvlab3d
```

The main window provides:

- **Project panel** — project contents and workflow items.
- **Data Table** — imported observation data.
- **3D View** — optional 3-D inspection of available project objects.
- **Mesh** — mesh information and operations.
- **Model** — physical-property and reference-model operations.
- **Inversion** — method selection and inversion configuration.
- **Properties panel** — information about the selected project object.

The application menu provides File, Project, Data, Mesh, Model, Inversion, View, and Help commands. Press `F1` to open the integrated help when available.

## Typical workflow

A typical graphical workflow is:

1. Create a new project or open an existing project.
2. Import geophysical observations using **File → Import Data**.
3. Select the geophysical method: Gravity, Magnetic, or MT.
4. Create a structured 3-D mesh or load a mesh file.
5. Create or load the corresponding physical-property model.
6. Load or define a reference model when required by the inversion configuration.
7. Configure the inversion parameters.
8. Run the inversion.
9. Inspect the recovered model and inversion results.
10. Save the project and export results as needed.

**Note:** when using the GUI, import observations before creating a mesh. Replacing an existing mesh may clear the active model, reference model, and inversion result.

Project files managed by the GUI use the `.json` extension.

## Mesh and model data

GeoInvLab3D uses a structured Cartesian 3-D mesh. Mesh geometry defines the cell layout and the spatial locations at which physical-property values are assigned.

The framework's mesh I/O supports its parameter-based ASCII representation and supported tabular mesh representations. A tabular cell-centred mesh file can use the following nine-column layout:

```text
# ix iy iz x_center_m y_center_m z_center_m dx_m dy_m dz_m
0 0 0 50 50 50 100 100 100
0 0 1 50 50 150 100 100 100
0 0 2 50 50 250 100 100 100
```

The columns represent:

| Column | Description |
|---|---|
| `ix`, `iy`, `iz` | Integer cell indices along the X, Y, and Z axes |
| `x_center_m`, `y_center_m`, `z_center_m` | Cell-centre coordinates in metres |
| `dx_m`, `dy_m`, `dz_m` | Cell dimensions in metres |

For a regular structured mesh, the cell indices and geometry must describe a complete, consistent grid. The current `StructuredMesh3D` representation uses uniform main-cell widths; a tabular file describing nonuniform cell widths cannot be represented faithfully by this class.

Physical-property models contain one value per mesh cell. The relevant property depends on the selected method:

| Method | Model parameter | Typical unit |
|---|---|---|
| Gravity | Density or density contrast | kg/m³ |
| Magnetic | Magnetic susceptibility | SI |
| MT | Electrical resistivity | ohm·m |

Model files must correspond to the active mesh and follow the format expected by the model loader.

## Geophysical methods

### Gravity

The gravity workflow relates subsurface density contrasts to gravity observations. It supports forward-response calculation and inversion of density distributions.

### Magnetic

The magnetic workflow calculates magnetic responses associated with subsurface magnetic-susceptibility distributions. The data component, coordinate convention, units, and susceptibility convention must be consistent with the selected forward operator.

### Magnetotelluric (MT)

The MT workflow uses frequency-dependent electromagnetic responses to estimate subsurface electrical resistivity. Each station-frequency observation can contain the four impedance-tensor components and their corresponding phases:

- `rho_xx`, `rho_xy`, `rho_yx`, `rho_yy`
- `phase_xx`, `phase_xy`, `phase_yx`, `phase_yy`

The MT data table therefore represents eight scalar data components per station-frequency observation. The exact input schema and uncertainty-column handling are defined by the data importer.

## Development and tests

Install the full development dependencies:

```bash
python -m pip install -e ".[all]"
```

Run the test suite from the repository root:

```bash
pytest -q
```

Run Ruff checks:

```bash
ruff check .
```

To run tests with coverage:

```bash
pytest --cov=geoinvlab --cov-report=term-missing
```

## Project structure

The repository is organized around the Python package in `src/`, with examples, tests, and documentation alongside it. The principal areas are:

```text
GeoInvLab3D/
├── src/
│   └── geoinvlab/
│       ├── core/          # Shared data, mesh, and model structures
│       ├── methods/       # Geophysical forward and inversion methods
│       ├── inversion/     # Inversion and optimization components
│       ├── io/            # Data and model input/output
│       └── gui/           # Desktop application and integrated help
├── examples/              # Synthetic and workflow examples
├── tests/                 # Automated tests
├── docs/                  # Project documentation
├── pyproject.toml
├── LICENSE
└── README.md
```

The exact set of subpackages and example directories may expand as development continues.

## License

GeoInvLab3D is distributed under the **MIT License**. See the [`LICENSE`](LICENSE) file for the complete license text.

## Citation

If GeoInvLab3D is used in scientific work, please cite the software release or the associated publication when a formal citation is available.

**Software:** GeoInvLab3D, version 0.1.0.

**Manuscript title:** *GeoInvLab3D: An Open-Source Python Framework for Three-Dimensional Geophysical Modeling and Inversion.*

## Acknowledgements

GeoInvLab3D is developed as an open-source framework to support reproducible 3-D geophysical modelling, forward simulation, and inversion research.
