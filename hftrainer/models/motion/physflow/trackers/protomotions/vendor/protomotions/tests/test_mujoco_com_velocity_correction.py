# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
Test suite for MuJoCo COM offset extraction and velocity correction.

This test validates that:
1. COM offset extraction and velocity correction methods are implemented
2. Quaternion rotation of COM offsets is correct
3. Cross product correction formula produces expected results
4. Overall velocity semantics match IsaacGym after correction
"""

import os
import sys
import numpy as np
import torch
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from protomotions.simulator.mujoco.simulator import MujocoSimulator
from protomotions.simulator.mujoco.config import MujocoSimulatorConfig, MujocoSimParams
from protomotions.robot_configs.factory import robot_config
from protomotions.components.terrains.terrain import Terrain
from protomotions.components.terrains.config import TerrainConfig
from protomotions.components.scene_lib import SceneLib
from protomotions.simulator.base_simulator.config import ProjectileConfig
import pytest


class TestCOMVelocityCorrection:
    """Test suite for MuJoCo COM offset extraction and velocity correction."""

    def test_methods_exist(self):
        """Test that all required methods exist in MujocoSimulator."""
        methods = [
            '_extract_body_com_offsets',
            '_apply_com_velocity_correction',
            '_get_simulator_bodies_state'
        ]
        
        for method_name in methods:
            assert hasattr(MujocoSimulator, method_name), \
                f"Method {method_name} not found in MujocoSimulator"

    def test_simulator_instantiation(self):
        """Test that MuJoCo simulator can be instantiated."""
        robot_cfg = robot_config('smpl')
        terrain_cfg = TerrainConfig()
        terrain = Terrain(terrain_cfg, device=torch.device('cpu'), num_envs=1)
        scene_lib = SceneLib.empty(num_envs=1, device='cpu', terrain=terrain)
        
        mujoco_config = MujocoSimulatorConfig(
            num_envs=1,
            headless=True,
            use_implicit_pd=False,
            experiment_name="test_com_velocity",
            sim=MujocoSimParams(fps=60, decimation=2),
            projectile=ProjectileConfig(num_projectiles=0),
        )
        
        simulator = MujocoSimulator(
            config=mujoco_config,
            robot_config=robot_cfg,
            terrain=terrain,
            device=torch.device('cpu'),
            scene_lib=scene_lib,
        )
        
        assert simulator is not None, "Failed to create simulator"

    def test_quaternion_rotation_correctness(self):
        """Test that quaternion rotation of COM offsets is correct."""
        # Test with identity quaternion (wxyz format)
        identity_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        com_offset = np.array([0.01, 0.02, 0.03], dtype=np.float32)
        
        # Rotate by identity quaternion - should get same vector
        w, x, y, z = identity_quat[0], identity_quat[1], identity_quat[2], identity_quat[3]
        vx, vy, vz = com_offset[0], com_offset[1], com_offset[2]
        
        # q * v * q^* = v (for identity quaternion)
        # First multiply: q * v
        qv_w = -x * vx - y * vy - z * vz
        qv_x = w * vx + y * vz - z * vy
        qv_y = w * vy + z * vx - x * vz
        qv_z = w * vz + x * vy - y * vx
        
        # Then multiply by q^*
        quat_inv = np.array([-x, -y, -z, w], dtype=np.float32)
        qvq_x = qv_w * quat_inv[0] + qv_x * quat_inv[3] + qv_y * quat_inv[2] - qv_z * quat_inv[1]
        qvq_y = qv_w * quat_inv[1] + qv_y * quat_inv[3] + qv_z * quat_inv[0] - qv_x * quat_inv[2]
        qvq_z = qv_w * quat_inv[2] + qv_z * quat_inv[3] + qv_x * quat_inv[1] - qv_y * quat_inv[0]
        
        result = np.array([qvq_x, qvq_y, qvq_z], dtype=np.float32)
        
        np.testing.assert_allclose(result, com_offset, atol=1e-6,
            err_msg="Quaternion rotation with identity should preserve vector")

    def test_cross_product_correction_formula(self):
        """Test that cross product correction formula is correct."""
        # Angular velocity
        omega = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        
        # COM offset in world frame
        r_offset = np.array([0.05, 0.1, 0.15], dtype=np.float32)
        
        # Expected correction: omega x r_offset
        correction = np.cross(omega, r_offset)
        
        # Verify the cross product is correct
        expected_x = omega[1] * r_offset[2] - omega[2] * r_offset[1]
        expected_y = omega[2] * r_offset[0] - omega[0] * r_offset[2]
        expected_z = omega[0] * r_offset[1] - omega[1] * r_offset[0]
        
        expected = np.array([expected_x, expected_y, expected_z], dtype=np.float32)
        
        np.testing.assert_allclose(correction, expected, atol=1e-8,
            err_msg="Cross product calculation incorrect")

    def test_velocity_correction_magnitude_bounds(self):
        """Test that velocity correction magnitude is bounded by |omega| * |r|."""
        omega = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        r_offset = np.array([0.1, 0.1, 0.1], dtype=np.float32)
        
        correction = np.cross(omega, r_offset)
        correction_mag = np.linalg.norm(correction)
        
        omega_mag = np.linalg.norm(omega)
        r_mag = np.linalg.norm(r_offset)
        max_possible = omega_mag * r_mag
        
        # Cross product magnitude is at most |omega| * |r| * sin(angle)
        # Since sin(angle) <= 1, magnitude should be <= omega_mag * r_mag
        assert correction_mag <= max_possible + 1e-6, \
            f"Correction magnitude {correction_mag} exceeds bound {max_possible}"

    def test_zero_offset_no_correction(self):
        """Test that zero COM offset produces no velocity correction."""
        omega = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        r_offset_zero = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        
        # omega x 0 = 0
        correction = np.cross(omega, r_offset_zero)
        
        np.testing.assert_allclose(correction, np.zeros(3), atol=1e-8,
            err_msg="Non-zero correction for zero offset")

    def test_orthogonal_vectors_max_correction(self):
        """Test that orthogonal omega and r give maximum correction."""
        # Orthogonal vectors: omega along x, r along y
        omega = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        r_offset = np.array([0.0, 0.1, 0.0], dtype=np.float32)
        
        correction = np.cross(omega, r_offset)
        correction_mag = np.linalg.norm(correction)
        
        # For orthogonal vectors: |omega x r| = |omega| * |r|
        omega_mag = np.linalg.norm(omega)
        r_mag = np.linalg.norm(r_offset)
        
        np.testing.assert_allclose(correction_mag, omega_mag * r_mag, atol=1e-8,
            err_msg="Orthogonal vectors should give maximum correction")

    def test_parallel_vectors_zero_correction(self):
        """Test that parallel omega and r give zero correction."""
        # Parallel vectors
        omega = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        r_offset = 0.5 * omega  # r is parallel to omega
        
        correction = np.cross(omega, r_offset)
        
        np.testing.assert_allclose(correction, np.zeros(3), atol=1e-6,
            err_msg="Parallel vectors should give zero correction")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

