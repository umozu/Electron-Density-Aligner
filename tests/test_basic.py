"""
Basic tests for density aligner.
"""

import numpy as np
import pytest
from density_aligner.cube_io import Atom, CubeData, ATOMIC_SYMBOLS
from density_aligner.spatial_aligner import (
    rotate_density_180,
    quaternion_to_rotation_matrix_numpy,
    calculate_true_binary_overlap
)
from density_aligner.prealign import (
    calculate_center_of_mass,
    calculate_inertia_tensor
)


class TestAtom:
    def test_atom_creation(self):
        atom = Atom(6, 0.0, np.array([0.0, 0.0, 0.0]))
        assert atom.atomic_num == 6
        assert atom.symbol == 'C'
        assert np.allclose(atom.coords, [0, 0, 0])

    def test_atom_coords_angstrom(self):
        # 1 Bohr = 0.529177 Angstrom
        atom = Atom(6, 0.0, np.array([1.0, 0.0, 0.0]))
        assert np.isclose(atom.coords_angstrom[0], 0.529177, rtol=1e-4)


class TestRotation:
    def test_rotate_180_x(self):
        density = np.arange(8).reshape(2, 2, 2).astype(float)
        rotated = rotate_density_180(density, 'x')
        # 180 around X: y -> -y, z -> -z
        assert rotated.shape == density.shape
        assert not np.allclose(rotated, density)

    def test_rotate_180_y(self):
        density = np.arange(8).reshape(2, 2, 2).astype(float)
        rotated = rotate_density_180(density, 'y')
        assert rotated.shape == density.shape

    def test_quaternion_identity(self):
        q = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
        R = quaternion_to_rotation_matrix_numpy(q)
        assert np.allclose(R, np.eye(3), atol=1e-6)


class TestOverlap:
    def test_identical_overlap(self):
        density = np.random.rand(10, 10, 10)
        overlap = calculate_true_binary_overlap(density, density)
        assert np.isclose(overlap, 1.0, atol=1e-6)

    def test_no_overlap(self):
        d1 = np.zeros((10, 10, 10))
        d2 = np.zeros((10, 10, 10))
        d1[0:5, :, :] = 1.0
        d2[5:10, :, :] = 1.0
        overlap = calculate_true_binary_overlap(d1, d2)
        assert np.isclose(overlap, 0.0, atol=1e-6)

    def test_partial_overlap(self):
        d1 = np.zeros((10, 10, 10))
        d2 = np.zeros((10, 10, 10))
        d1[0:6, :, :] = 1.0  # 60% of grid
        d2[4:10, :, :] = 1.0  # 60% of grid, 20% overlap
        overlap = calculate_true_binary_overlap(d1, d2)
        # Overlap = 2/10, Union = 10/10
        assert 0 < overlap < 1


class TestPrealign:
    def test_center_of_mass(self):
        atoms = [
            Atom(6, 0.0, np.array([0.0, 0.0, 0.0])),
            Atom(6, 0.0, np.array([2.0, 0.0, 0.0]))
        ]
        com = calculate_center_of_mass(atoms)
        assert np.allclose(com, [1.0, 0.0, 0.0])

    def test_inertia_tensor_symmetric(self):
        atoms = [
            Atom(6, 0.0, np.array([1.0, 0.0, 0.0])),
            Atom(6, 0.0, np.array([-1.0, 0.0, 0.0])),
            Atom(6, 0.0, np.array([0.0, 1.0, 0.0])),
            Atom(6, 0.0, np.array([0.0, -1.0, 0.0]))
        ]
        center = np.array([0.0, 0.0, 0.0])
        I = calculate_inertia_tensor(atoms, center)
        # Inertia tensor should be symmetric
        assert np.allclose(I, I.T)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
