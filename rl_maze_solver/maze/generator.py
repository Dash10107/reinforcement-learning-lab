"""
Maze generation using recursive DFS backtracker.
Guarantees a fully connected, solvable maze with proper corridors.
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np


class Cell(IntEnum):
    OPEN = 0
    WALL = 1
    START = 2
    GOAL = 3


PRESETS = {
    "🐣 Tiny (5×5)": {"size": 5, "wall_density": 0.0},
    "🐇 Small (7×7)": {"size": 7, "wall_density": 0.0},
    "🐢 Medium (9×9)": {"size": 9, "wall_density": 0.0},
    "🦊 Large (13×13)": {"size": 13, "wall_density": 0.0},
    "🐉 XL (17×17)": {"size": 17, "wall_density": 0.0},
}

THEMES = {
    "🏰 Classic": {"wall": "#2d3561", "open": "#f5f0e8", "path": "#e94560"},
    "🌲 Forest": {"wall": "#1b4332", "open": "#d8f3dc", "path": "#f77f00"},
    "🌌 Space": {"wall": "#03045e", "open": "#0d1b2a", "path": "#00b4d8"},
    "🔥 Lava": {"wall": "#370617", "open": "#ffd166", "path": "#ef233c"},
}


def generate_dfs_maze(size: int, seed: int = 42) -> np.ndarray:
    """
    DFS recursive backtracker on a (size×size) grid.
    Works on a logical grid where passages exist between cells.
    Returns a (2*size-1) × (2*size-1) wall grid where 0=open, 1=wall.
    """
    rng = np.random.default_rng(seed)
    # Full grid size (cells + walls between them)
    H = W = 2 * size - 1
    grid = np.ones((H, W), dtype=np.int8)  # start all walls

    # Mark logical cells as open
    for r in range(size):
        for c in range(size):
            grid[2 * r, 2 * c] = 0

    visited = np.zeros((size, size), dtype=bool)
    stack = [(0, 0)]
    visited[0, 0] = True

    while stack:
        r, c = stack[-1]
        neighbors = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < size and 0 <= nc < size and not visited[nr, nc]:
                neighbors.append((nr, nc, dr, dc))
        if neighbors:
            nr, nc, dr, dc = neighbors[rng.integers(len(neighbors))]
            # Carve passage
            grid[2 * r + dr, 2 * c + dc] = 0
            visited[nr, nc] = True
            stack.append((nr, nc))
        else:
            stack.pop()

    return grid


def generate_open_maze(
    size: int, wall_frac: float = 0.20, seed: int = 42
) -> np.ndarray:
    """Simple random-wall maze — fast, less structured."""
    rng = np.random.default_rng(seed)
    grid = np.zeros((size, size), dtype=np.int8)
    n_walls = int(size * size * wall_frac)
    cells = [
        (r, c)
        for r in range(size)
        for c in range(size)
        if not (r == 0 and c == 0) and not (r == size - 1 and c == size - 1)
    ]
    rng.shuffle(cells)
    for r, c in cells[:n_walls]:
        grid[r, c] = 1
    return grid
