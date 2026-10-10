"""Instance-local reuse of immutable finite-magnet geometry, not a force cache.

Positions and rotations are refreshed for every force evaluation. getFT still
uses the original finite differences and quadrature at every physical tick.
The local target mesh depends only on dimension, meshing and magnetization.
"""
import numpy as np


class SameTickPoseCache:
    """Only immutable reads in the same physical tick; reset invalidates it."""
    def __init__(self): self.invalidate()

    def invalidate(self): self.key, self.value = None, None

    def read(self, key, fetch):
        if self.key != key or self.value is None:
            self.value = fetch()
            self.key = key
        return self.value


def cache_finite_geometry(model):
    cache = {}
    for name in ('_cube', '_cylinder'):
        original = getattr(model, name)

        def reuse(position, rotation_matrix, *args, _name=name, _original=original, **kwargs):
            key = (_name, repr(args), repr(sorted(kwargs.items())))
            if key not in cache:
                magnet = _original(position, rotation_matrix, *args, **kwargs)
                if magnet.meshing is not None:
                    mesh = magnet._generate_mesh()
                    for value in mesh.values():
                        if isinstance(value, np.ndarray): value.setflags(write=False)
                    magnet._generate_mesh = lambda _mesh=mesh: _mesh
                cache[key] = magnet
            magnet = cache[key]
            magnet.position = np.asarray(position, dtype=float)
            magnet.orientation = model._rotation.from_matrix(np.asarray(rotation_matrix, dtype=float))
            return magnet

        setattr(model, name, reuse)
    model._geometry_cache = cache
    return model
