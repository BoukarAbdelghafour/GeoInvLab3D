"""
ASCII input/output utilities for cell-based physical models.
MIT License Copyright (c) 01/10/2026 [Boukar Abdelghafour]

Supported formats
-----------------
1. GeoInvLab3D native format:
   Metadata followed by one physical-property value per cell.

2. Simple ASCII format:
   One physical-property value per cell.

3. Coordinate-based ASCII format:
   X center, Y center, Z center, physical-property value.

4. Indexed coordinate-based ASCII format:
   ix, iy, iz, X center, Y center, Z center, physical-property value.

All coordinate-based formats are validated against the supplied mesh.

Values follow GeoInvLab3D's x-y-z ordering, with z varying fastest.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Union

import numpy as np

from ..mesh.structured import StructuredMesh3D
from .model import Model3D


PathLike = Union[str, Path]

_FORMAT_VERSION = 1
_HEADER = "GeoInvLab3D physical model"

_COORDINATE_TOLERANCE = 1e-8
_COORDINATE_RTOL = 1e-10

_REQUIRED_KEYS = {
    "format_version",
    "model_type",
    "name",
    "unit",
    "value_count",
    "ordering",
    "mesh_type",
    "mesh_nx",
    "mesh_ny",
    "mesh_nz",
    "mesh_dx",
    "mesh_dy",
    "mesh_dz",
    "mesh_origin",
    "mesh_padding_x",
    "mesh_padding_y",
    "mesh_padding_z",
    "mesh_padding_factor",
}


# ==============================================================
# PARSING UTILITIES
# ==============================================================

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
            f"Parameter '{key}' must be finite."
        )

    return result


def _parse_tuple(
    value: str,
    key: str,
    expected_length: int,
    value_type: str,
) -> tuple:
    """Parse a comma-separated tuple."""

    parts = [
        part.strip()
        for part in value.split(",")
    ]

    if len(parts) != expected_length:
        raise ValueError(
            f"Parameter '{key}' must contain exactly "
            f"{expected_length} comma-separated values."
        )

    if value_type == "int":
        return tuple(
            _parse_integer(part, key)
            for part in parts
        )

    if value_type == "float":
        return tuple(
            _parse_float(part, key)
            for part in parts
        )

    raise ValueError(
        f"Unsupported tuple value type: {value_type!r}."
    )


def _validate_text(value: str, key: str) -> str:
    """Validate a single-line ASCII metadata value."""

    value = str(value)

    if "\n" in value or "\r" in value or "=" in value:
        raise ValueError(
            f"'{key}' must not contain line breaks or '='."
        )

    try:
        value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError(
            f"'{key}' must contain ASCII characters only."
        ) from exc

    return value


def _format_tuple(values: tuple) -> str:
    """Format a numeric tuple for an ASCII metadata line."""

    return ", ".join(
        f"{value:.17g}"
        if isinstance(value, (float, np.floating))
        else str(value)
        for value in values
    )


# ==============================================================
# PROPERTY NAME AND UNIT DETECTION
# ==============================================================

def _infer_property_metadata(
    comments: list[str],
    n_columns: int,
) -> tuple[str, str]:
    """
    Infer the physical-property name and unit from ASCII comments.

    Recognized examples
    -------------------
    density_contrast_kg_m3 -> density_contrast, kg/m3
    resistivity_ohm_m      -> resistivity, ohm.m
    conductivity_s_m       -> conductivity, S/m
    susceptibility_si      -> susceptibility, SI
    """

    property_name = "model"
    unit = ""

    candidates = []

    for comment in comments:

        tokens = comment.split()

        if not tokens:
            continue

        lowered = [token.lower() for token in tokens]

        if n_columns == 7:

            if lowered[:3] == ["ix", "iy", "iz"]:
                candidates.append(tokens[-1])

        elif n_columns == 4:

            if len(tokens) >= 4:
                candidates.append(tokens[-1])

    if not candidates:
        return property_name, unit

    property_name = candidates[-1]

    unit_patterns = {
        "_kg_m3": "kg/m3",
        "_ohm_m": "ohm.m",
        "_s_m": "S/m",
        "_si": "SI",
    }

    for suffix, detected_unit in unit_patterns.items():

        if property_name.lower().endswith(suffix):

            property_name = property_name[:-len(suffix)]
            unit = detected_unit
            break

    return property_name, unit


# ==============================================================
# MESH PARAMETERS
# ==============================================================

def _mesh_parameters(
    mesh: StructuredMesh3D,
) -> Dict[str, object]:
    """Return structured-mesh parameters."""

    if not isinstance(mesh, StructuredMesh3D):
        raise TypeError(
            "Model I/O currently supports StructuredMesh3D meshes only."
        )

    return {
        "mesh_type": "structured",
        "mesh_nx": int(mesh.nx),
        "mesh_ny": int(mesh.ny),
        "mesh_nz": int(mesh.nz),
        "mesh_dx": float(mesh.dx),
        "mesh_dy": float(mesh.dy),
        "mesh_dz": float(mesh.dz),
        "mesh_origin": tuple(
            float(v) for v in mesh.origin
        ),
        "mesh_padding_x": tuple(
            int(v) for v in mesh.padding_x
        ),
        "mesh_padding_y": tuple(
            int(v) for v in mesh.padding_y
        ),
        "mesh_padding_z": tuple(
            int(v) for v in mesh.padding_z
        ),
        "mesh_padding_factor": float(
            mesh.padding_factor
        ),
    }


# ==============================================================
# FILE FORMAT DETECTION
# ==============================================================

def _detect_format(lines: list[str]) -> str:
    """Detect the model file format."""

    meaningful = [
        line.strip()
        for line in lines
        if line.strip()
        and not line.strip().startswith("#")
    ]

    if not meaningful:
        raise ValueError("Model file is empty.")

    if (
        meaningful[0] == _HEADER
        or any(line.lower() == "values" for line in meaningful)
        or any("=" in line for line in meaningful)
    ):
        return "native"

    return "ascii"


# ==============================================================
# NATIVE GEOINVLAB3D FORMAT
# ==============================================================

def _read_native_model_file(
    lines: list[str],
) -> tuple[Dict[str, object], np.ndarray]:
    """Read a GeoInvLab3D native model file."""

    parameters: Dict[str, object] = {}
    value_tokens = []
    reading_values = False

    for line_number, raw_line in enumerate(lines, start=1):

        line = raw_line.strip()

        if not line or line.startswith("#"):
            continue

        if line == _HEADER:
            continue

        if not reading_values:

            if line.lower() == "values":
                reading_values = True
                continue

            if "=" not in line:
                raise ValueError(
                    f"Invalid syntax at line {line_number}: "
                    "expected 'key = value' or the 'values' marker."
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

        else:
            value_tokens.extend(line.split())

    if not reading_values:
        raise ValueError(
            "Model file is missing the 'values' section."
        )

    missing = _REQUIRED_KEYS.difference(parameters)

    if missing:
        raise ValueError(
            "Model file is missing required parameters: "
            + ", ".join(sorted(missing))
        )

    expected_count = _parse_integer(
        str(parameters["value_count"]),
        "value_count",
    )

    if expected_count < 1:
        raise ValueError(
            "'value_count' must be greater than zero."
        )

    if len(value_tokens) != expected_count:
        raise ValueError(
            f"Model file contains {len(value_tokens)} values, "
            f"but value_count is {expected_count}."
        )

    try:
        values = np.asarray(value_tokens, dtype=float)
    except ValueError as exc:
        raise ValueError(
            "The model values section contains a non-numeric value."
        ) from exc

    if not np.all(np.isfinite(values)):
        raise ValueError(
            "Model values must all be finite."
        )

    parameters["format"] = "native"

    return parameters, values


# ==============================================================
# ASCII SIMPLE / COORDINATE FORMATS
# ==============================================================

def _read_ascii_model_file(
    lines: list[str],
) -> tuple[Dict[str, object], np.ndarray]:
    """
    Read an ASCII model file.

    Supported layouts
    -----------------
    Simple:
        value

    Coordinate-based:
        x_center y_center z_center physical_property

    Indexed coordinate-based:
        ix iy iz x_center y_center z_center physical_property
    """

    rows = []
    comments = []

    for line_number, raw_line in enumerate(lines, start=1):

        line = raw_line.strip()

        if not line:
            continue

        if line.startswith("#"):
            comments.append(line[1:].strip())
            continue

        tokens = line.replace(",", " ").split()

        try:
            row = [float(token) for token in tokens]
        except ValueError as exc:
            raise ValueError(
                f"Non-numeric value at line {line_number}."
            ) from exc

        if not row:
            continue

        if not np.all(np.isfinite(row)):
            raise ValueError(
                f"Non-finite value at line {line_number}."
            )

        rows.append(row)

    if not rows:
        raise ValueError(
            "ASCII model file contains no numerical data."
        )

    column_counts = {
        len(row) for row in rows
    }

    if len(column_counts) != 1:
        raise ValueError(
            "ASCII model file contains inconsistent column counts."
        )

    n_columns = column_counts.pop()

    # ----------------------------------------------------------
    # One value per cell.
    # ----------------------------------------------------------

    if n_columns == 1:

        values = np.asarray(
            rows,
            dtype=float,
        ).ravel()

        parameters = {
            "format": "ascii_simple",
            "name": "model",
            "unit": "",
        }

        return parameters, values

    # ----------------------------------------------------------
    # X, Y, Z and physical property.
    # ----------------------------------------------------------

    if n_columns == 4:

        data = np.asarray(rows, dtype=float)

        coordinates = data[:, :3]
        values = data[:, 3]

        name, unit = _infer_property_metadata(
            comments,
            n_columns,
        )

        parameters = {
            "format": "ascii_coordinates",
            "name": name,
            "unit": unit,
            "coordinates": coordinates,
        }

        return parameters, values

    # ----------------------------------------------------------
    # ix, iy, iz, X, Y, Z and physical property.
    # ----------------------------------------------------------

    if n_columns == 7:

        data = np.asarray(rows, dtype=float)

        raw_indices = data[:, :3]

        rounded_indices = np.rint(raw_indices)

        if not np.allclose(
            raw_indices,
            rounded_indices,
            rtol=0.0,
            atol=1e-10,
        ):
            raise ValueError(
                "Cell indices ix, iy and iz must be integers."
            )

        indices = rounded_indices.astype(np.int64)

        coordinates = data[:, 3:6]
        values = data[:, 6]

        name, unit = _infer_property_metadata(
            comments,
            n_columns,
        )

        parameters = {
            "format": "ascii_indexed_coordinates",
            "name": name,
            "unit": unit,
            "indices": indices,
            "coordinates": coordinates,
        }

        return parameters, values

    raise ValueError(
        "Unsupported ASCII model format. Expected one, four "
        "or seven columns:\n"
        "  1: physical-property values\n"
        "  4: X, Y, Z, physical property\n"
        "  7: ix, iy, iz, X, Y, Z, physical property."
    )


# ==============================================================
# GENERIC FILE READER
# ==============================================================

def _read_model_data(
    path: PathLike,
) -> tuple[Dict[str, object], np.ndarray]:
    """Read a model file and detect its format."""

    file_path = Path(path)

    if not file_path.is_file():
        raise FileNotFoundError(
            f"Model file does not exist: {file_path}"
        )

    try:
        with file_path.open("r", encoding="ascii") as stream:
            lines = stream.readlines()

    except UnicodeDecodeError as exc:
        raise ValueError(
            f"Model file must use ASCII encoding: {file_path}"
        ) from exc

    file_format = _detect_format(lines)

    if file_format == "native":
        return _read_native_model_file(lines)

    return _read_ascii_model_file(lines)


def _read_model_file(
    path: PathLike,
) -> tuple[Dict[str, object], np.ndarray]:
    """Backward-compatible internal reader."""

    return _read_model_data(path)


# ==============================================================
# MESH COMPATIBILITY
# ==============================================================

def _check_mesh_compatibility(
    parameters: Dict[str, object],
    mesh: StructuredMesh3D,
) -> None:
    """Check native model metadata against a supplied mesh."""

    if not isinstance(mesh, StructuredMesh3D):
        raise TypeError(
            "load_model currently requires a StructuredMesh3D mesh."
        )

    if str(parameters["mesh_type"]).strip().lower() != "structured":
        raise ValueError(
            "The model file was not created for a structured mesh."
        )

    # Mesh dimensions.
    for field in ("nx", "ny", "nz"):

        stored = _parse_integer(
            str(parameters[f"mesh_{field}"]),
            f"mesh_{field}",
        )

        current = int(getattr(mesh, field))

        if stored != current:
            raise ValueError(
                f"Mesh incompatibility: {field} is {stored} "
                f"in the model file but {current} "
                "in the supplied mesh."
            )

    # Cell dimensions and padding factor.
    for field in ("dx", "dy", "dz", "padding_factor"):

        stored = _parse_float(
            str(parameters[f"mesh_{field}"]),
            f"mesh_{field}",
        )

        current = float(getattr(mesh, field))

        if not math.isclose(
            stored,
            current,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"Mesh incompatibility: {field} differs."
            )

    # Origin.
    stored_origin = _parse_tuple(
        str(parameters["mesh_origin"]),
        "mesh_origin",
        expected_length=3,
        value_type="float",
    )

    for axis, (stored, current) in enumerate(
        zip(stored_origin, mesh.origin)
    ):

        if not math.isclose(
            stored,
            float(current),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"Mesh incompatibility: origin coordinate {axis} differs."
            )

    # Padding.
    for axis in ("x", "y", "z"):

        stored_padding = _parse_tuple(
            str(parameters[f"mesh_padding_{axis}"]),
            f"mesh_padding_{axis}",
            expected_length=2,
            value_type="int",
        )

        current_padding = tuple(
            int(value)
            for value in getattr(mesh, f"padding_{axis}")
        )

        if stored_padding != current_padding:
            raise ValueError(
                f"Mesh incompatibility: padding_{axis} differs."
            )


# ==============================================================
# COORDINATE VALIDATION
# ==============================================================

def _check_coordinate_compatibility(
    coordinates: np.ndarray,
    mesh: StructuredMesh3D,
) -> None:
    """
    Verify ASCII cell coordinates against the supplied mesh.

    The comparison is performed row by row. The file must follow
    GeoInvLab3D's x-y-z ordering, with z varying fastest.
    """

    expected = mesh.cell_centers()

    if coordinates.shape != expected.shape:
        raise ValueError(
            "Coordinate array has an incompatible shape. "
            f"Expected {expected.shape}, "
            f"received {coordinates.shape}."
        )

    matching = np.isclose(
        coordinates,
        expected,
        rtol=_COORDINATE_RTOL,
        atol=_COORDINATE_TOLERANCE,
    )

    if not np.all(matching):

        differences = np.abs(
            coordinates - expected
        )

        row_errors = np.any(
            ~matching,
            axis=1,
        )

        first_error = int(
            np.flatnonzero(row_errors)[0]
        )

        max_error = float(
            np.max(differences)
        )

        raise ValueError(
            "Model coordinates are incompatible with the supplied "
            "mesh or use a different cell ordering. "
            f"First incompatible cell: {first_error}. "
            f"Maximum coordinate difference: {max_error:.6g} m. "
            "Expected x-y-z ordering with z varying fastest."
        )


# ==============================================================
# INDEX VALIDATION
# ==============================================================

def _check_index_compatibility(
    indices: np.ndarray,
    mesh: StructuredMesh3D,
) -> None:
    """
    Verify indexed ASCII cell ordering against the supplied mesh.

    Expected indices are zero-based:

        ix: 0 ... nx_total - 1
        iy: 0 ... ny_total - 1
        iz: 0 ... nz_total - 1

    The ordering is x outer, y middle and z fastest.
    """

    expected = np.indices(
        mesh.shape,
        dtype=np.int64,
    ).reshape(3, -1).T

    if indices.shape != expected.shape:
        raise ValueError(
            "Index array has an incompatible shape. "
            f"Expected {expected.shape}, received {indices.shape}."
        )

    if not np.array_equal(indices, expected):

        mismatch = np.any(
            indices != expected,
            axis=1,
        )

        first_error = int(
            np.flatnonzero(mismatch)[0]
        )

        raise ValueError(
            "Model cell indices are incompatible with the supplied "
            "mesh or use a different cell ordering. "
            f"First incompatible cell: {first_error}. "
            "Expected zero-based x-y-z ordering with z fastest."
        )


# ==============================================================
# SAVE MODEL
# ==============================================================

def save_model(
    model: Model3D,
    path: PathLike,
    unit: str = "",
) -> None:
    """
    Save a cell-based Model3D to an ASCII text file.

    One physical-property value is written per mesh cell.

    The associated structured mesh geometry is stored in the file.
    """

    if not isinstance(model, Model3D):
        raise TypeError(
            "save_model expects a Model3D instance."
        )

    if not isinstance(model.mesh, StructuredMesh3D):
        raise TypeError(
            "save_model currently supports StructuredMesh3D meshes only."
        )

    name = _validate_text(model.name, "name")
    unit = _validate_text(unit, "unit")

    values = np.asarray(
        model.values,
        dtype=float,
    )

    if values.ndim != 1:
        raise ValueError(
            "Model values must be a one-dimensional array."
        )

    if values.size != model.mesh.n_cells:
        raise ValueError(
            f"Model contains {values.size} values, "
            f"but the mesh contains {model.mesh.n_cells} cells."
        )

    if not np.all(np.isfinite(values)):
        raise ValueError(
            "Model values must all be finite."
        )

    mesh = _mesh_parameters(model.mesh)

    lines = [
        _HEADER,
        "# One physical-property value per mesh cell.",
        "# Ordering: x-y-z, with z as the fastest-varying index.",
        "",
        f"format_version = {_FORMAT_VERSION}",
        "model_type = cell_property",
        f"name = {name}",
        f"unit = {unit}",
        f"value_count = {values.size}",
        "ordering = x-y-z (z fastest)",
        "",
        "# Associated structured mesh",
        f"mesh_type = {mesh['mesh_type']}",
        f"mesh_nx = {mesh['mesh_nx']}",
        f"mesh_ny = {mesh['mesh_ny']}",
        f"mesh_nz = {mesh['mesh_nz']}",
        f"mesh_dx = {mesh['mesh_dx']:.17g}",
        f"mesh_dy = {mesh['mesh_dy']:.17g}",
        f"mesh_dz = {mesh['mesh_dz']:.17g}",
        f"mesh_origin = {_format_tuple(mesh['mesh_origin'])}",
        f"mesh_padding_x = {_format_tuple(mesh['mesh_padding_x'])}",
        f"mesh_padding_y = {_format_tuple(mesh['mesh_padding_y'])}",
        f"mesh_padding_z = {_format_tuple(mesh['mesh_padding_z'])}",
        f"mesh_padding_factor = {mesh['mesh_padding_factor']:.17g}",
        "",
        "values",
    ]

    lines.extend(
        f"{value:.17g}"
        for value in values
    )

    lines.append("")

    file_path = Path(path)

    file_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_path.write_text(
        "\n".join(lines),
        encoding="ascii",
    )


# ==============================================================
# LOAD MODEL
# ==============================================================

def load_model(
    path: PathLike,
    mesh: StructuredMesh3D,
    *,
    return_metadata: bool = False,
):
    """
    Load a cell-based Model3D from an ASCII text file.

    Supported formats
    -----------------
    1. GeoInvLab3D native format.
    2. Simple ASCII format with one value per cell.
    3. Coordinate-based ASCII format with four columns.
    4. Indexed coordinate-based ASCII format with seven columns.

    Coordinate-based files are checked against the supplied mesh.

    Parameters
    ----------
    path : str or pathlib.Path
        Source model file.

    mesh : StructuredMesh3D
        Mesh associated with the model.

    return_metadata : bool, optional
        If True, return (model, metadata).

    Returns
    -------
    Model3D
        Loaded physical-property model.

    tuple
        (model, metadata) when return_metadata=True.
    """

    if not isinstance(mesh, StructuredMesh3D):
        raise TypeError(
            "load_model currently requires a StructuredMesh3D mesh."
        )

    parameters, values = _read_model_data(path)

    file_format = parameters.get(
        "format",
        "native",
    )

    # ----------------------------------------------------------
    # Native format.
    # ----------------------------------------------------------

    if file_format == "native":

        version = _parse_integer(
            str(parameters["format_version"]),
            "format_version",
        )

        if version != _FORMAT_VERSION:
            raise ValueError(
                f"Unsupported model format version {version}. "
                f"Supported version: {_FORMAT_VERSION}."
            )

        if str(
            parameters["model_type"]
        ).strip().lower() != "cell_property":
            raise ValueError(
                "Unsupported model type. "
                "Expected 'cell_property'."
            )

        ordering = str(
            parameters["ordering"]
        ).strip().lower()

        if ordering != "x-y-z (z fastest)":
            raise ValueError(
                f"Unsupported model ordering: "
                f"{parameters['ordering']!r}."
            )

        _check_mesh_compatibility(
            parameters,
            mesh,
        )

        name = str(parameters["name"])
        unit = str(parameters["unit"])

    # ----------------------------------------------------------
    # Simple ASCII format.
    # ----------------------------------------------------------

    elif file_format == "ascii_simple":

        version = None

        name = str(parameters["name"])
        unit = str(parameters["unit"])

    # ----------------------------------------------------------
    # Coordinate-based ASCII format.
    # ----------------------------------------------------------

    elif file_format == "ascii_coordinates":

        version = None

        _check_coordinate_compatibility(
            np.asarray(parameters["coordinates"], dtype=float),
            mesh,
        )

        name = str(parameters["name"])
        unit = str(parameters["unit"])

    # ----------------------------------------------------------
    # Indexed coordinate-based ASCII format.
    # ----------------------------------------------------------

    elif file_format == "ascii_indexed_coordinates":

        version = None

        _check_index_compatibility(
            np.asarray(parameters["indices"], dtype=np.int64),
            mesh,
        )

        _check_coordinate_compatibility(
            np.asarray(parameters["coordinates"], dtype=float),
            mesh,
        )

        name = str(parameters["name"])
        unit = str(parameters["unit"])

    else:

        raise ValueError(
            f"Unsupported model file format: {file_format!r}."
        )

    # ----------------------------------------------------------
    # Number of model parameters.
    # ----------------------------------------------------------

    if values.size != mesh.n_cells:
        raise ValueError(
            f"Model contains {values.size} values, "
            f"but the supplied mesh contains "
            f"{mesh.n_cells} cells."
        )

    # ----------------------------------------------------------
    # Build Model3D.
    # ----------------------------------------------------------

    model = Model3D(
        mesh=mesh,
        values=values,
        name=name,
    )

    if not return_metadata:
        return model

    metadata = {
        "name": name,
        "unit": unit,
        "value_count": int(values.size),
        "ordering": "x-y-z (z fastest)",
        "format_version": version,
        "format": file_format,
    }

    return model, metadata


# ==============================================================
# PUBLIC ALIASES
# ==============================================================

def write_model(
    model: Model3D,
    path: PathLike,
    unit: str = "",
) -> None:
    """Alias for save_model()."""

    save_model(
        model,
        path,
        unit=unit,
    )


def read_model(
    path: PathLike,
    mesh: StructuredMesh3D,
    *,
    return_metadata: bool = False,
):
    """Alias for load_model()."""

    return load_model(
        path,
        mesh,
        return_metadata=return_metadata,
    )
