"""
ASCII input/output utilities for structured meshes.

This module provides functions to save and load StructuredMesh3D objects.

The mesh file stores:
    - Main-domain cell counts.
    - Main-domain cell dimensions.
    - Mesh origin.
    - Padding-cell counts.
    - Padding expansion factor.

Padding cells are reconstructed when the mesh is loaded.

Current file format:

    GeoInvLab3D structured mesh
    format_version = 1
    mesh_type = structured
    nx = 10
    ny = 10
    nz = 5
    dx = 100.0
    dy = 100.0
    dz = 100.0
    origin = -500.0, -500.0, 0.0
    padding_x = 2, 2
    padding_y = 2, 2
    padding_z = 1, 1
    padding_factor = 1.5

Legacy file format also accepted:

    nx ny nz dx dy dz x0 y0 z0

Legacy files are interpreted as meshes without padding.
MIT License Copyright (c) 01/10/2026 [Boukar Abdelghafour]
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Union

from .structured import StructuredMesh3D


PathLike = Union[str, Path]

_FORMAT_VERSION = 1
_HEADER = "GeoInvLab3D structured mesh"

_REQUIRED_KEYS = {
    "format_version",
    "mesh_type",
    "nx",
    "ny",
    "nz",
    "dx",
    "dy",
    "dz",
    "origin",
    "padding_x",
    "padding_y",
    "padding_z",
    "padding_factor",
}


def _parse_integer(value: str, key: str) -> int:
    """Parse an integer parameter."""
    try:
        return int(value.strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid integer value for '{key}': {value!r}."
        ) from exc


def _parse_float(value: str, key: str) -> float:
    """Parse a finite floating-point parameter."""
    try:
        result = float(value.strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid floating-point value for '{key}': {value!r}."
        ) from exc

    if not math.isfinite(result):
        raise ValueError(
            f"Parameter '{key}' must be finite, got {result!r}."
        )

    return result


def _parse_tuple(
    value: str,
    key: str,
    expected_length: int,
    value_type: str,
) -> tuple:
    """Parse a comma-separated tuple."""
    parts = [part.strip() for part in value.split(",")]

    if len(parts) != expected_length:
        raise ValueError(
            f"Parameter '{key}' must contain exactly "
            f"{expected_length} comma-separated values."
        )

    if value_type == "int":
        return tuple(_parse_integer(part, key) for part in parts)

    if value_type == "float":
        return tuple(_parse_float(part, key) for part in parts)

    raise ValueError(
        f"Unsupported tuple value type: {value_type!r}."
    )


def _legacy_numeric_parameters(
    file_path: Path,
) -> Dict[str, str] | None:
    """
    Try to read the legacy numeric mesh format.

    Expected format:
        nx ny nz dx dy dz x0 y0 z0

    The first line may be a comment/header beginning with '#'.

    Cell counts may use scientific notation, for example:
        2.0000000000e+01

    Returns None when the file is not a valid legacy numeric mesh.
    """
    try:
        with file_path.open("r", encoding="ascii") as stream:
            lines = []

            for raw_line in stream:
                line = raw_line.strip()

                if not line:
                    continue

                if line.startswith("#"):
                    continue

                lines.append(line)

    except UnicodeDecodeError:
        return None

    if len(lines) != 1:
        return None

    tokens = lines[0].split()

    if len(tokens) != 9:
        return None

    try:
        # Accept scientific notation for cell counts.
        count_values = [
            float(token)
            for token in tokens[:3]
        ]

        # Cell counts must be finite integer-valued numbers.
        if not all(
            math.isfinite(value) and value.is_integer()
            for value in count_values
        ):
            return None

        nx, ny, nz = (
            int(value)
            for value in count_values
        )

        dx = float(tokens[3])
        dy = float(tokens[4])
        dz = float(tokens[5])

        x0 = float(tokens[6])
        y0 = float(tokens[7])
        z0 = float(tokens[8])

    except (TypeError, ValueError, OverflowError):
        return None

    values = (
        dx,
        dy,
        dz,
        x0,
        y0,
        z0,
    )

    if not all(math.isfinite(value) for value in values):
        return None

    if nx <= 0 or ny <= 0 or nz <= 0:
        return None

    if dx <= 0.0 or dy <= 0.0 or dz <= 0.0:
        return None

    return {
        "format_version": str(_FORMAT_VERSION),
        "mesh_type": "structured",
        "nx": str(nx),
        "ny": str(ny),
        "nz": str(nz),
        "dx": str(dx),
        "dy": str(dy),
        "dz": str(dz),
        "origin": f"{x0}, {y0}, {z0}",
        "padding_x": "0, 0",
        "padding_y": "0, 0",
        "padding_z": "0, 0",
        "padding_factor": "1.0",
    }


def _axis_coordinate_mesh(path: PathLike) -> StructuredMesh3D | None:
    """
    Read a mesh described by axis id, edge index, and edge coordinate.

    Expected rows:
        axis_id edge_index coordinate_m

    Axis ids are 0 (X), 1 (Y), and 2 (Z). Each axis must contain
    consecutive edge indices starting at zero. StructuredMesh3D
    currently represents uniform main-cell widths, so nonuniform
    edge coordinates are rejected rather than approximated.
    """
    file_path = Path(path)

    try:
        with file_path.open("r", encoding="ascii") as stream:
            rows = []
            saw_axis_header = False

            for line_number, raw_line in enumerate(stream, start=1):
                line = raw_line.strip()
                if not line:
                    continue

                if line.startswith("#"):
                    header = line[1:].strip().lower()
                    if "axis_id" in header and "edge_index" in header:
                        saw_axis_header = True
                    continue

                tokens = line.split()
                if len(tokens) != 3:
                    if saw_axis_header:
                        raise ValueError(
                            f"Invalid axis-coordinate mesh row at line "
                            f"{line_number}: expected three columns."
                        )
                    return None

                try:
                    axis_value = float(tokens[0])
                    index_value = float(tokens[1])
                    coordinate = float(tokens[2])
                except ValueError:
                    if saw_axis_header:
                        raise ValueError(
                            f"Invalid numeric value at line {line_number}."
                        )
                    return None

                if (
                    not math.isfinite(axis_value)
                    or not axis_value.is_integer()
                    or not math.isfinite(index_value)
                    or not index_value.is_integer()
                    or not math.isfinite(coordinate)
                ):
                    raise ValueError(
                        f"Invalid axis, edge index, or coordinate "
                        f"at line {line_number}."
                    )

                axis = int(axis_value)
                edge_index = int(index_value)
                if axis not in (0, 1, 2):
                    raise ValueError(
                        f"Invalid axis id {axis} at line {line_number}; "
                        "expected 0 (X), 1 (Y), or 2 (Z)."
                    )
                if edge_index < 0:
                    raise ValueError(
                        f"Negative edge index at line {line_number}."
                    )

                rows.append((axis, edge_index, coordinate))

    except UnicodeDecodeError as exc:
        raise ValueError(
            f"Mesh file must use ASCII encoding: {file_path}"
        ) from exc

    if not rows:
        return None

    axes = {0: {}, 1: {}, 2: {}}
    for axis, edge_index, coordinate in rows:
        if edge_index in axes[axis]:
            raise ValueError(
                f"Duplicate edge index {edge_index} for axis {axis}."
            )
        axes[axis][edge_index] = coordinate

    names = ("X", "Y", "Z")
    counts = []
    spacings = []
    origins = []

    for axis in (0, 1, 2):
        axis_edges = axes[axis]
        if len(axis_edges) < 2:
            raise ValueError(
                f"Axis {names[axis]} must contain at least two edges."
            )

        expected_indices = list(range(len(axis_edges)))
        actual_indices = sorted(axis_edges)
        if actual_indices != expected_indices:
            raise ValueError(
                f"Edge indices for axis {names[axis]} must be consecutive "
                "and start at zero."
            )

        coordinates = [axis_edges[i] for i in expected_indices]
        differences = [
            coordinates[i + 1] - coordinates[i]
            for i in range(len(coordinates) - 1)
        ]

        if any(value <= 0.0 for value in differences):
            raise ValueError(
                f"Coordinates for axis {names[axis]} must be strictly "
                "increasing."
            )

        spacing = differences[0]
        tolerance = max(1.0e-10, abs(spacing) * 1.0e-8)
        if any(abs(value - spacing) > tolerance for value in differences):
            raise ValueError(
                f"Nonuniform cell widths detected on axis {names[axis]}. "
                "The current StructuredMesh3D class supports uniform "
                "main-cell widths only."
            )

        counts.append(len(coordinates) - 1)
        spacings.append(spacing)
        origins.append(coordinates[0])

    try:
        return StructuredMesh3D(
            nx=counts[0],
            ny=counts[1],
            nz=counts[2],
            dx=spacings[0],
            dy=spacings[1],
            dz=spacings[2],
            origin=tuple(origins),
            padding_x=(0, 0),
            padding_y=(0, 0),
            padding_z=(0, 0),
            padding_factor=1.0,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid axis-coordinate mesh definition: {exc}"
        ) from exc



def _cell_table_mesh(path: PathLike) -> StructuredMesh3D | None:
    """Read a cell-based structured mesh table.

    Supported columns::

        ix iy iz x_center_m y_center_m z_center_m dx_m dy_m dz_m

    The current StructuredMesh3D representation supports uniform
    main-cell widths. Therefore, this reader accepts cell tables whose
    widths are constant along each coordinate axis. The complete
    Cartesian grid must be present and indices must start at zero.
    """
    file_path = Path(path)
    try:
        with file_path.open("r", encoding="ascii") as stream:
            lines = list(stream)
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"Mesh file must use ASCII encoding: {file_path}"
        ) from exc

    header_found = False
    rows = []
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            header = line[1:].strip().lower().replace(",", " ")
            if all(token in header for token in (
                "ix", "iy", "iz", "x_center_m", "y_center_m",
                "z_center_m", "dx_m", "dy_m", "dz_m",
            )):
                header_found = True
            continue
        if not header_found:
            # This is not the cell-table format. Let other readers try it.
            return None
        tokens = line.split()
        if len(tokens) != 9:
            raise ValueError(
                f"Invalid cell-table row at line {line_number}: "
                f"expected 9 columns, got {len(tokens)}."
            )
        try:
            values = [float(token) for token in tokens]
        except ValueError as exc:
            raise ValueError(
                f"Non-numeric value in cell-table row at line {line_number}."
            ) from exc
        if not all(math.isfinite(value) for value in values):
            raise ValueError(
                f"Non-finite value in cell-table row at line {line_number}."
            )
        index_values = values[:3]
        if any(not value.is_integer() for value in index_values):
            raise ValueError(
                f"Cell indices must be integers at line {line_number}."
            )
        indices = tuple(int(value) for value in index_values)
        centers = tuple(values[3:6])
        widths = tuple(values[6:9])
        if any(index < 0 for index in indices):
            raise ValueError(
                f"Cell indices must be non-negative at line {line_number}."
            )
        if any(width <= 0.0 for width in widths):
            raise ValueError(
                f"Cell dimensions must be positive at line {line_number}."
            )
        rows.append((indices, centers, widths, line_number))

    if not header_found:
        return None
    if not rows:
        raise ValueError("Cell-table mesh file contains no cell records.")

    counts = tuple(max(row[0][axis] for row in rows) + 1 for axis in range(3))
    expected_indices = {
        (ix, iy, iz)
        for ix in range(counts[0])
        for iy in range(counts[1])
        for iz in range(counts[2])
    }
    index_to_geometry = [{}, {}, {}]
    seen = set()
    for indices, centers, widths, line_number in rows:
        if indices in seen:
            raise ValueError(
                f"Duplicate cell index {indices} at line {line_number}."
            )
        seen.add(indices)
        for axis, index in enumerate(indices):
            geometry = (centers[axis], widths[axis])
            previous = index_to_geometry[axis].get(index)
            if previous is None:
                index_to_geometry[axis][index] = geometry
            else:
                scale = max(1.0, abs(previous[0]), abs(geometry[0]),
                            abs(previous[1]), abs(geometry[1]))
                tolerance = 1.0e-8 * scale
                if (abs(previous[0] - geometry[0]) > tolerance
                        or abs(previous[1] - geometry[1]) > tolerance):
                    raise ValueError(
                        f"Inconsistent cell geometry for axis {axis} "
                        f"and index {index} (line {line_number})."
                    )

    missing = expected_indices.difference(seen)
    if missing:
        example = min(missing)
        raise ValueError(
            "Cell-table mesh is incomplete: "
            f"{len(missing)} cells are missing (first missing index: {example})."
        )

    spacings = []
    origins = []
    axis_names = ("X", "Y", "Z")
    for axis in range(3):
        geometry = index_to_geometry[axis]
        if sorted(geometry) != list(range(counts[axis])):
            raise ValueError(
                f"Cell indices along {axis_names[axis]} must be consecutive "
                "and start at zero."
            )
        widths = [geometry[index][1] for index in range(counts[axis])]
        spacing = widths[0]
        tolerance = max(1.0e-10, abs(spacing) * 1.0e-8)
        if any(abs(width - spacing) > tolerance for width in widths):
            raise ValueError(
                f"Nonuniform cell widths detected along {axis_names[axis]}. "
                "The current StructuredMesh3D reader can reconstruct only "
                "uniform cell tables; padding reconstruction requires the "
                "original key=value mesh file."
            )
        first_center = geometry[0][0]
        origin = first_center - 0.5 * spacing
        for index in range(counts[axis]):
            expected_center = origin + (index + 0.5) * spacing
            if abs(geometry[index][0] - expected_center) > tolerance:
                raise ValueError(
                    f"Cell centers along {axis_names[axis]} are not regularly "
                    "spaced."
                )
        spacings.append(spacing)
        origins.append(origin)

    try:
        return StructuredMesh3D(
            nx=counts[0], ny=counts[1], nz=counts[2],
            dx=spacings[0], dy=spacings[1], dz=spacings[2],
            origin=tuple(origins),
            padding_x=(0, 0), padding_y=(0, 0), padding_z=(0, 0),
            padding_factor=1.0,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid cell-table mesh definition: {exc}"
        ) from exc

def _read_parameters(path: PathLike) -> Dict[str, str]:
    """Read mesh parameters from an ASCII file."""
    file_path = Path(path)

    if not file_path.is_file():
        raise FileNotFoundError(
            f"Mesh file does not exist: {file_path}"
        )

    # First try the legacy numeric mesh format.
    legacy = _legacy_numeric_parameters(file_path)

    if legacy is not None:
        return legacy

    parameters: Dict[str, str] = {}

    try:
        with file_path.open("r", encoding="ascii") as stream:

            for line_number, raw_line in enumerate(
                stream,
                start=1,
            ):
                line = raw_line.strip()

                if not line:
                    continue

                if line.startswith("#"):
                    continue

                if line == _HEADER:
                    continue

                if "=" not in line:
                    raise ValueError(
                        f"Invalid syntax at line {line_number}: "
                        "expected 'key = value'."
                    )

                key, value = line.split("=", maxsplit=1)

                key = key.strip().lower()
                value = value.strip()

                if not key:
                    raise ValueError(
                        f"Empty parameter name at line {line_number}."
                    )

                if key in parameters:
                    raise ValueError(
                        f"Duplicate parameter '{key}' "
                        f"at line {line_number}."
                    )

                parameters[key] = value

    except UnicodeDecodeError as exc:
        raise ValueError(
            f"Mesh file must use ASCII encoding: {file_path}"
        ) from exc

    missing = _REQUIRED_KEYS.difference(parameters)

    if missing:
        raise ValueError(
            "Mesh file is missing required parameters: "
            + ", ".join(sorted(missing))
        )

    return parameters


def save_mesh(
    mesh: StructuredMesh3D,
    path: PathLike,
) -> None:
    """
    Save a StructuredMesh3D to an ASCII text file.

    Parameters
    ----------
    mesh : StructuredMesh3D
        Structured mesh to save.
    path : str or pathlib.Path
        Destination mesh file.

    Raises
    ------
    TypeError
        If mesh is not a StructuredMesh3D instance.
    ValueError
        If mesh parameters are invalid.
    """
    if not isinstance(mesh, StructuredMesh3D):
        raise TypeError(
            "save_mesh supports StructuredMesh3D objects only."
        )

    file_path = Path(path)

    origin = tuple(
        float(value)
        for value in mesh.origin
    )

    if len(origin) != 3:
        raise ValueError(
            "Mesh origin must contain exactly three coordinates."
        )

    dimensions = (
        float(mesh.dx),
        float(mesh.dy),
        float(mesh.dz),
    )

    if not all(math.isfinite(value) for value in dimensions):
        raise ValueError(
            "Mesh cell dimensions must be finite."
        )

    if any(value <= 0.0 for value in dimensions):
        raise ValueError(
            "Mesh cell dimensions must be strictly positive."
        )

    if not all(math.isfinite(value) for value in origin):
        raise ValueError(
            "Mesh origin coordinates must be finite."
        )

    padding_factor = float(mesh.padding_factor)

    if (
        not math.isfinite(padding_factor)
        or padding_factor < 1.0
    ):
        raise ValueError(
            "Padding factor must be finite and greater than "
            "or equal to 1."
        )

    file_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    lines = [
        _HEADER,
        "# Structured mesh geometry.",
        "# Main-domain dimensions exclude padding cells.",
        "",
        f"format_version = {_FORMAT_VERSION}",
        "mesh_type = structured",
        "",
        "# Main-domain cell counts",
        f"nx = {int(mesh.nx)}",
        f"ny = {int(mesh.ny)}",
        f"nz = {int(mesh.nz)}",
        "",
        "# Main-domain cell dimensions",
        f"dx = {mesh.dx:.17g}",
        f"dy = {mesh.dy:.17g}",
        f"dz = {mesh.dz:.17g}",
        "",
        "# Mesh origin (x, y, z)",
        "origin = " + ", ".join(
            f"{value:.17g}"
            for value in origin
        ),
        "",
        "# Padding cells (negative side, positive side)",
        (
            f"padding_x = {int(mesh.padding_x[0])}, "
            f"{int(mesh.padding_x[1])}"
        ),
        (
            f"padding_y = {int(mesh.padding_y[0])}, "
            f"{int(mesh.padding_y[1])}"
        ),
        (
            f"padding_z = {int(mesh.padding_z[0])}, "
            f"{int(mesh.padding_z[1])}"
        ),
        f"padding_factor = {padding_factor:.17g}",
        "",
    ]

    file_path.write_text(
        "\n".join(lines),
        encoding="ascii",
    )


def load_mesh(path: PathLike) -> StructuredMesh3D:
    """
    Load a StructuredMesh3D from an ASCII text file.

    Supported formats:
        - Cell tables: ix iy iz x_center_m y_center_m z_center_m dx_m dy_m dz_m.
        - Axis-edge coordinates: axis_id edge_index coordinate_m.
        - Current key=value format.
        - Legacy nine-column numeric format.

    Axis-edge files must describe uniform cell widths because the
    current StructuredMesh3D representation uses uniform main cells.

    Parameters
    ----------
    path : str or pathlib.Path
        Source mesh file.

    Returns
    -------
    StructuredMesh3D
        Reconstructed structured mesh.

    Raises
    ------
    FileNotFoundError
        If the mesh file does not exist.
    ValueError
        If the file format or mesh parameters are invalid.
    """
    # Cell-centered tables are used by mesh exports and are detected
    # from their explicit nine-column header.
    cell_mesh = _cell_table_mesh(path)
    if cell_mesh is not None:
        return cell_mesh

    # Also support explicit axis-edge coordinate files.
    axis_mesh = _axis_coordinate_mesh(path)
    if axis_mesh is not None:
        return axis_mesh

    parameters = _read_parameters(path)

    version = _parse_integer(
        parameters["format_version"],
        "format_version",
    )

    if version != _FORMAT_VERSION:
        raise ValueError(
            f"Unsupported mesh format version {version}. "
            f"Supported version: {_FORMAT_VERSION}."
        )

    mesh_type = parameters["mesh_type"].strip().lower()

    if mesh_type != "structured":
        raise ValueError(
            f"Unsupported mesh type: {mesh_type!r}. "
            "Expected 'structured'."
        )

    nx = _parse_integer(
        parameters["nx"],
        "nx",
    )

    ny = _parse_integer(
        parameters["ny"],
        "ny",
    )

    nz = _parse_integer(
        parameters["nz"],
        "nz",
    )

    dx = _parse_float(
        parameters["dx"],
        "dx",
    )

    dy = _parse_float(
        parameters["dy"],
        "dy",
    )

    dz = _parse_float(
        parameters["dz"],
        "dz",
    )

    origin = _parse_tuple(
        parameters["origin"],
        key="origin",
        expected_length=3,
        value_type="float",
    )

    padding_x = _parse_tuple(
        parameters["padding_x"],
        key="padding_x",
        expected_length=2,
        value_type="int",
    )

    padding_y = _parse_tuple(
        parameters["padding_y"],
        key="padding_y",
        expected_length=2,
        value_type="int",
    )

    padding_z = _parse_tuple(
        parameters["padding_z"],
        key="padding_z",
        expected_length=2,
        value_type="int",
    )

    padding_factor = _parse_float(
        parameters["padding_factor"],
        "padding_factor",
    )

    try:
        return StructuredMesh3D(
            nx=nx,
            ny=ny,
            nz=nz,
            dx=dx,
            dy=dy,
            dz=dz,
            origin=origin,
            padding_x=padding_x,
            padding_y=padding_y,
            padding_z=padding_z,
            padding_factor=padding_factor,
        )

    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid structured mesh definition: {exc}"
        ) from exc


def write_mesh(
    mesh: StructuredMesh3D,
    path: PathLike,
) -> None:
    """Alias for save_mesh()."""
    save_mesh(mesh, path)


def read_mesh(path: PathLike) -> StructuredMesh3D:
    """Alias for load_mesh()."""
    return load_mesh(path)
