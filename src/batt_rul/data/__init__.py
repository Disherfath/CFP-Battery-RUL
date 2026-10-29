"""Dataset adapters used by the experiment protocol."""

from .hust import (
    HUST_FIT_IDS,
    HUST_TEST_IDS,
    HUST_TRAIN_IDS,
    HUST_VALID_IDS,
    build_hust_cell_cache,
    resample_discharge_cycle,
)
from .xjtu import (
    XJTU_CURVE_COLUMNS,
    assign_xjtu_roles,
    assign_xjtu_test_folds,
    build_xjtu_cell_cache,
    resample_xjtu_discharge,
)

__all__ = [
    "HUST_FIT_IDS",
    "HUST_TEST_IDS",
    "HUST_TRAIN_IDS",
    "HUST_VALID_IDS",
    "XJTU_CURVE_COLUMNS",
    "assign_xjtu_roles",
    "assign_xjtu_test_folds",
    "build_hust_cell_cache",
    "build_xjtu_cell_cache",
    "resample_discharge_cycle",
    "resample_xjtu_discharge",
]
