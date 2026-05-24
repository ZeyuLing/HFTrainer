# PRISM 1B text-to-motion with KT-RoPE spectral mode
# Uses Laplacian spectral coordinates to encode kinematic tree topology
#
# KT-RoPE Advantages:
#   - Encodes skeletal structure: joints with similar kinematic roles get similar embeddings
#   - Higher correlation with kinematic tree distance (0.849 vs 0.397 for sequential)
#   - Zero additional parameters (spectral modes are precomputed constants)
#   - Better generalization to motion with different pose configurations

_base_ = './prism_1b_tp2m_1frame.py'

model = dict(
    transformer=dict(
        joint_pos_mode="spectral",  # KT-RoPE spectral mode
        num_spectral_modes=4,  # Use first 4 Laplacian eigenvectors
        spectral_scale=22.0,  # Scale spectral coords to num_joints
    ),
)
