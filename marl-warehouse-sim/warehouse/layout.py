"""
Warehouse grid layouts and task generation.

Grid cell types:
  0 = open floor
  1 = shelf / wall (impassable)
  2 = pickup zone  (walk here to load a package)
  3 = delivery zone (walk here to deliver a package)
"""

from __future__ import annotations
import numpy as np

OPEN     = 0
SHELF    = 1
PICKUP   = 2
DELIVERY = 3

# ── 12×12 warehouse ───────────────────────────────────────────────────────────
GRID_12 = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 2, 2, 0, 0, 0, 0, 0, 0, 2, 2, 1],
    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
    [1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1],
    [1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1],
    [1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1],
    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
    [1, 3, 3, 0, 0, 0, 0, 0, 0, 3, 3, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
], dtype=np.int8)


def get_layout(size: str = "medium") -> np.ndarray:
    return GRID_12.copy()


def get_open_cells(grid: np.ndarray) -> list[tuple[int, int]]:
    rows, cols = np.where(grid == OPEN)
    return list(zip(rows.tolist(), cols.tolist()))


def get_pickup_cells(grid: np.ndarray) -> list[tuple[int, int]]:
    rows, cols = np.where(grid == PICKUP)
    return list(zip(rows.tolist(), cols.tolist()))


def get_delivery_cells(grid: np.ndarray) -> list[tuple[int, int]]:
    rows, cols = np.where(grid == DELIVERY)
    return list(zip(rows.tolist(), cols.tolist()))
