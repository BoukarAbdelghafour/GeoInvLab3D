#!/usr/bin/env python3
"""
MIT License Copyright (c) 17/09/2026 [Boukar Abdelghafour]
"""
from pathlib import Path
import shutil
import argparse


ROOT = Path(__file__).resolve().parent.parent


PROTECTED = {
    ".git",
    ".venv",
    "venv",
    "env",
    "src",
    "tests",
    "scripts",
}


def remove_directory(path, dry_run):
    if not path.exists():
        return False

    if not path.is_dir():
        return False

    if path.name in PROTECTED:
        return False

    if dry_run:
        print(f"[DRY RUN] directory: {path}")
    else:
        shutil.rmtree(path)
        print(f"Removed directory: {path}")

    return True


def remove_file(path, dry_run):
    if not path.exists():
        return False

    if not path.is_file():
        return False

    protected_files = {
        "pyproject.toml",
        "README.md",
        "LICENSE",
    }

    if path.name in protected_files:
        return False

    if dry_run:
        print(f"[DRY RUN] file: {path}")
    else:
        path.unlink()
        print(f"Removed file: {path}")

    return True


def clean_python_cache(dry_run):
    count = 0

    for path in list(ROOT.rglob("__pycache__")):
        if remove_directory(path, dry_run):
            count += 1

    for path in list(ROOT.rglob("*.pyc")):
        if remove_file(path, dry_run):
            count += 1

    for path in list(ROOT.rglob("*.pyo")):
        if remove_file(path, dry_run):
            count += 1

    return count


def clean_test_cache(dry_run):
    count = 0

    directories = [
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        ".hypothesis",
        ".tox",
        ".nox",
        "htmlcov",
    ]

    for name in directories:
        for path in list(ROOT.rglob(name)):
            if remove_directory(path, dry_run):
                count += 1

    files = [
        ".coverage",
        "coverage.xml",
    ]

    for name in files:
        for path in list(ROOT.rglob(name)):
            if remove_file(path, dry_run):
                count += 1

    return count


def clean_build_files(dry_run):
    count = 0

    for name in ["build", "dist"]:
        path = ROOT / name

        if remove_directory(path, dry_run):
            count += 1

    for path in list(ROOT.rglob("*.egg-info")):
        if remove_directory(path, dry_run):
            count += 1

    return count


def clean_temporary_files(dry_run):
    count = 0

    for pattern in ["*.tmp", "*.temp", "*~"]:
        for path in list(ROOT.rglob(pattern)):
            if remove_file(path, dry_run):
                count += 1

    return count


def main():
    parser = argparse.ArgumentParser(
        description="Clean GeoInvLab3D generated files."
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be removed without deleting anything.",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Also remove temporary files.",
    )

    args = parser.parse_args()

    print()
    print("=" * 60)
    print("GeoInvLab3D - CLEAN")
    print("=" * 60)
    print()
    print(f"Project root: {ROOT}")
    print()

    total = 0

    print("Cleaning Python caches...")
    total += clean_python_cache(args.dry_run)

    print()
    print("Cleaning test caches...")
    total += clean_test_cache(args.dry_run)

    print()
    print("Cleaning build files...")
    total += clean_build_files(args.dry_run)

    if args.all:
        print()
        print("Cleaning temporary files...")
        total += clean_temporary_files(args.dry_run)

    print()
    print("=" * 60)

    if args.dry_run:
        print(f"DRY RUN: {total} item(s) would be removed.")
    else:
        print(f"Cleanup complete: {total} item(s) removed.")

    print("=" * 60)
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
