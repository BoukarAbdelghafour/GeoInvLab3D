"""
Data import utilities for GeoInvLab3D.

Supported formats
-----------------
CSV, TXT, DAT, XYZ

Classical data
--------------
3 columns: x, y, value
4 columns: x, y, z, value
5 columns: x, y, z, value, uncertainty

Magnetotelluric data
--------------------
12 columns:
    x, y, z, frequency,
    rho_xx, rho_xy, rho_yx, rho_yy,
    phase_xx, phase_xy, phase_yx, phase_yy

13 columns:
    station_id, frequency, x, y, z,
    rho_xx, rho_xy, rho_yx, rho_yy,
    phase_xx, phase_xy, phase_yx, phase_yy

20 columns:
    Standard MT data with eight uncertainty columns.

21 columns:
    MT data with station identifiers and eight uncertainty columns.
    
    MIT License Copyright (c) 01/10/2026 [Boukar Abdelghafour]
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from geoinvlab.core.data import DataSet
from geoinvlab.core.data.mt_dataset import MTDataSet


SUPPORTED_EXTENSIONS = {
    ".csv",
    ".txt",
    ".dat",
    ".xyz",
}


def load_data_file(
    filename: str | Path,
    *,
    name: str | None = None,
    units: str | None = None,
    delimiter: str | None = None,
) -> DataSet | MTDataSet:
    """Load classical or magnetotelluric data."""

    path = Path(filename)

    if not path.exists():
        raise FileNotFoundError(
            f"Data file does not exist: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Data path is not a file: {path}"
        )

    extension = path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported data format '{extension}'. "
            f"Supported formats: "
            f"{', '.join(sorted(SUPPORTED_EXTENSIONS))}."
        )

    dataset_name = name or path.stem

    if delimiter is None and extension == ".csv":
        delimiter = ","

    try:
        table = _read_text_file(
            path,
            delimiter=delimiter,
        )
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"Could not read data file '{path}': {exc}"
        ) from exc

    if table.ndim != 2 or table.shape[0] == 0:
        raise ValueError(
            "Data file must contain a non-empty table."
        )

    n_columns = table.shape[1]

    # ==============================================================
    # Magnetotelluric data with station identifiers
    # ==============================================================

    if n_columns in (13, 21):

        station_ids = table[:, 0]

        if np.any(station_ids == ""):
            raise ValueError(
                "MT station identifiers must not be empty."
            )

        try:
            numeric = table[:, 1:].astype(float)
        except ValueError as exc:
            raise ValueError(
                "MT data after the station identifier must "
                "contain only numeric values."
            ) from exc

        if not np.all(np.isfinite(numeric)):
            raise ValueError(
                "MT data contains NaN or infinite values."
            )

        # File order:
        # frequency, x, y, z, responses, [uncertainties]
        #
        # MTDataSet order:
        # x, y, z, frequency, responses, [uncertainties]

        column_order = [
            1, 2, 3, 0,
            4, 5, 6, 7,
            8, 9, 10, 11,
        ]

        if n_columns == 21:
            column_order.extend(
                range(12, 20)
            )

        reordered = numeric[:, column_order]

        try:
            dataset = MTDataSet.from_array(
                reordered,
                name=dataset_name,
            )
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"Invalid MT dataset in '{path}': {exc}"
            ) from exc

        # Keep station identifiers available on the imported object.
        # MTDataSet currently does not define station_ids as a dataclass
        # field, so this is an additional runtime attribute.
        dataset.station_ids = station_ids.copy()

        return dataset

    # ==============================================================
    # Numeric data formats
    # ==============================================================

    try:
        data = table.astype(float)
    except ValueError as exc:
        raise ValueError(
            "Data file must contain numeric values. "
            "For MT files with station identifiers, use the "
            "13-column or 21-column format."
        ) from exc

    if not np.all(np.isfinite(data)):
        raise ValueError(
            "Data file contains NaN or infinite values."
        )

    # ==============================================================
    # Magnetotelluric data without station identifiers
    # ==============================================================

    if n_columns in (12, 20):

        try:
            return MTDataSet.from_array(
                data,
                name=dataset_name,
            )
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"Invalid MT dataset in '{path}': {exc}"
            ) from exc

    # ==============================================================
    # Classical geophysical data
    # ==============================================================

    if n_columns == 3:

        locations = np.column_stack(
            (
                data[:, 0],
                data[:, 1],
                np.zeros(data.shape[0]),
            )
        )

        values = data[:, 2]
        uncertainties = None

    elif n_columns == 4:

        locations = data[:, :3]
        values = data[:, 3]
        uncertainties = None

    elif n_columns == 5:

        locations = data[:, :3]
        values = data[:, 3]
        uncertainties = data[:, 4]

    else:

        raise ValueError(
            "Unsupported data layout.\n\n"
            "Classical data:\n"
            "  3 columns: x, y, value\n"
            "  4 columns: x, y, z, value\n"
            "  5 columns: x, y, z, value, uncertainty\n\n"
            "MT data:\n"
            "  12 columns: coordinates, frequency and 8 responses\n"
            "  13 columns: station ID, frequency, coordinates "
            "and 8 responses\n"
            "  20 columns: coordinates, frequency, responses "
            "and uncertainties\n"
            "  21 columns: station ID, frequency, coordinates, "
            "responses and uncertainties"
        )

    if uncertainties is not None:
        if np.any(uncertainties < 0):
            raise ValueError(
                "Data uncertainties must be non-negative."
            )

    return DataSet(
        values=np.asarray(values, dtype=float),
        locations=np.asarray(locations, dtype=float),
        name=dataset_name,
        units=units,
        uncertainties=(
            None
            if uncertainties is None
            else np.asarray(uncertainties, dtype=float)
        ),
    )


def _read_text_file(
    path: Path,
    *,
    delimiter: str | None = None,
) -> np.ndarray:
    """
    Read a text table without assuming that every column is numeric.

    This is necessary for MT files containing station identifiers
    such as STA01, STA02, etc.
    """

    try:
        table = np.genfromtxt(
            path,
            delimiter=delimiter,
            comments="#",
            dtype=str,
            encoding="utf-8",
            autostrip=True,
            invalid_raise=True,
        )

    except (OSError, ValueError) as exc:
        raise ValueError(str(exc)) from exc

    if table.size == 0:
        raise ValueError("The file is empty.")

    table = np.asarray(table, dtype=str)

    if table.ndim == 1:
        table = table.reshape(1, -1)

    return table


def import_data(
    filename: str | Path,
    *,
    name: str | None = None,
    units: str | None = None,
    delimiter: str | None = None,
) -> DataSet | MTDataSet:
    """Convenience alias for load_data_file."""

    return load_data_file(
        filename,
        name=name,
        units=units,
        delimiter=delimiter,
    )
