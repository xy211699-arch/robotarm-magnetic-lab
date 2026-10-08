"""Three-dimensional magnetic streamline integration."""

import math

import numpy as np


def trace_streamlines(field_function, cube_dimensions_m, config):
    """Trace external B-field lines from seed rings on the local +Z pole.

    ``field_function`` must accept an ``(N, 3)`` array in the magnet's local
    metric frame and return B vectors in tesla. Midpoint integration is batched
    across all live lines so the analytical field evaluator remains fast.
    """
    dimensions = np.asarray(cube_dimensions_m, dtype=float)
    half = 0.5 * dimensions
    seeds_per_ring = int(config["streamline_seeds_per_ring"])
    seed_z = half[2] + 1.0e-5
    seeds = []
    for radius in config["streamline_seed_radii_m"]:
        for index in range(seeds_per_ring):
            angle = 2.0 * math.pi * index / seeds_per_ring
            seeds.append(
                [float(radius) * math.cos(angle), float(radius) * math.sin(angle), seed_z]
            )

    positions = np.asarray(seeds, dtype=float)
    lines = [[point.copy()] for point in positions]
    active = np.ones(len(positions), dtype=bool)
    step_size = float(config["streamline_step_m"])
    max_radius = float(config["streamline_max_radius_m"])
    min_field = float(config["streamline_min_field_t"])

    for step_index in range(int(config["streamline_max_steps"])):
        live_indices = np.flatnonzero(active)
        if len(live_indices) == 0:
            break

        live_positions = positions[live_indices]
        field = np.asarray(field_function(live_positions), dtype=float)
        magnitude = np.linalg.norm(field, axis=1)
        valid = np.isfinite(field).all(axis=1) & (magnitude >= min_field)
        direction = np.zeros_like(field)
        direction[valid] = field[valid] / magnitude[valid, None]

        midpoint = live_positions + 0.5 * step_size * direction
        midpoint_field = np.asarray(field_function(midpoint), dtype=float)
        midpoint_magnitude = np.linalg.norm(midpoint_field, axis=1)
        midpoint_valid = (
            np.isfinite(midpoint_field).all(axis=1)
            & (midpoint_magnitude >= min_field)
            & valid
        )
        midpoint_direction = np.zeros_like(midpoint_field)
        midpoint_direction[midpoint_valid] = (
            midpoint_field[midpoint_valid] / midpoint_magnitude[midpoint_valid, None]
        )
        new_positions = live_positions + step_size * midpoint_direction

        for local_index, global_index in enumerate(live_indices):
            if not midpoint_valid[local_index]:
                active[global_index] = False
                continue
            point = new_positions[local_index]
            positions[global_index] = point
            lines[global_index].append(point.copy())
            outside_limit = np.linalg.norm(point) > max_radius
            entered_magnet = step_index > 2 and np.all(np.abs(point) <= half)
            if outside_limit or entered_magnet:
                active[global_index] = False

    return [np.asarray(line, dtype=float) for line in lines if len(line) >= 4]
