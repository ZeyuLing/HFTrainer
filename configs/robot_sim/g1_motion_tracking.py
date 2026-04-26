# Robot simulation pipeline config: HyMotion T2M → G1 Retarget → ASAP
#
# This config defines the full pipeline parameters for driving a
# Unitree G1 humanoid robot from text-to-motion generation.
#
# Usage:
#   python tools/robot_sim/text_to_g1.py \
#       --config configs/robot_sim/g1_motion_tracking.py \
#       --checkpoint checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt \
#       --prompt "a person walks forward"

_base_ = '../hymotion_t2m/hymotion_t2m_201dim_046b.py'

# Override for robot sim pipeline
work_dir = 'work_dirs/robot_sim_g1'

# Inference settings optimized for robot control
# - More frames for longer motion clips (useful for locomotion)
# - Higher quality ODE steps
model = dict(
    infer_noise_scheduler_cfg=dict(validation_steps=50),
)

# ---- Robot Retargeting Config ----
robot_retarget = dict(
    # Target robot
    robot='unitree_g1',
    g1_dof=29,             # 29 for full version, 23 for basic

    # Retargeting options
    apply_limits=True,      # Clamp to hardware joint limits
    rest_pose_calibration=True,  # Account for T-pose → rest-pose difference

    # Motion generation
    fps=30.0,              # Frames per second
    default_num_frames=120,  # 4 seconds at 30fps
)

# ---- ASAP Training Config ----
# These settings configure the Isaac Gym RL training for motion imitation.
asap_training = dict(
    # Environment
    num_envs=4096,          # Parallel simulation environments
    simulator='isaacgym',   # or 'isaacsim', 'genesis'

    # Robot
    robot_cfg='g1/g1_29dof_anneal_23dof',
    terrain='terrain_locomotion_plane',

    # Observation
    obs_cfg='motion_tracking/deepmimic_a2c_nolinvel_LARGEnoise_history',

    # Rewards
    reward_cfg='motion_tracking/reward_motion_tracking_dm_2real',
    reward_penalty_curriculum=True,
    reward_penalty_degree=0.00001,

    # Training
    max_iterations=6000,
    save_interval=200,

    # Logging
    project_name='HyMotion_G1_MotionTracking',
)

# ---- Quality Checks ----
# Sanity checks applied after retargeting
quality_checks = dict(
    # Maximum allowed joint velocity (rad/s)
    max_joint_velocity=10.0,
    # Maximum allowed root velocity (m/s)
    max_root_velocity=5.0,
    # Minimum motion duration (seconds)
    min_duration=0.5,
    # Flag if >N% of frames hit joint limits
    limit_saturation_threshold=0.3,
)
