"""
schema.py

Canonical localization schema for the SMLM wrapper pipeline.

Every backend output should eventually be converted to this structure.
"""

CANONICAL_COLUMNS = [
    "frame",
    "x",
    "y",
    "z",
    "photons",
    "background",
    "confidence",
    "backend",
    "source_file",
]


REQUIRED_COLUMNS = [
    "frame",
    "x",
    "y",
    "backend",
]


def get_canonical_columns():
    return CANONICAL_COLUMNS.copy()


def get_required_columns():
    return REQUIRED_COLUMNS.copy()
