================================================================================
                    SMPL-MUJOCO DOCUMENTATION INDEX
================================================================================

Generated Analysis of:
  /ref_repo/OmniH2O/phc/phc/smpllib/smpl_mujoco.py
  
Functions analyzed:
  - smpl_to_qpose() (lines 331-405)
  - qpos_to_smpl() (lines 552-571)
  - smpl_to_qpose_torch() (lines 486-549)
  - smpl_to_qpose_multi() (lines 408-483)

================================================================================
                        DOCUMENTATION FILES
================================================================================

START HERE:
  📄 SMPL_MUJOCO_SUMMARY.txt
     └─ Executive summary with all 5 key findings
        Quick caveats and critical gotchas

FOR CODE DETAILS:
  📄 SMPL_MUJOCO_CODE_REFERENCE.md
     └─ Complete function code with line-by-line explanations
        Inline comments for critical operations
        Quick reference tables

FOR COMPREHENSIVE ANALYSIS:
  📄 SMPL_MUJOCO_DETAILED_ANALYSIS.md
     └─ Comprehensive breakdown of all 5 questions
        Flow diagrams and detailed explanations
        Helper function documentation

================================================================================
                        KEY FINDINGS SUMMARY
================================================================================

1. EULER CONVENTION: "ZYX"
   ✓ Z-Y-X order (yaw → pitch → roll)
   ✓ Hardcoded in both encoding and decoding
   ✓ Default parameter but inverse operation is rigid

2. QPOS SLOTS:
   ✓ [0:3]     = Translation (x, y, z)
   ✓ [3:7]     = Root Quaternion [x, y, z, w]
   ✓ [7:10]    = Body Joint 1 Euler angles (ZYX)
   ✓ [10:13]   = Body Joint 2 Euler angles (ZYX)
   ✓ Root is quaternion, all body joints are Euler angles

3. COORDINATE TRANSFORMS:
   ✓ Root: Quaternion (4 DOF) - global frame
   ✓ Body: Euler angles ZYX (3 DOF each) - local frame
   ✓ Different handling: root not reordered, body joints reordered

4. REORDER MAPPING (smpl_2_mujoco):
   ✓ Maps SMPL bone order → MuJoCo body order
   ✓ Only applied to body joints (indices 1+)
   ✓ Root quaternion inserted BEFORE reordering
   ✓ Created line 371-374: list comprehension over MuJoCo bodies

5. BODY_POS[1] OFFSET:
   ✓ Forward (smpl_to_qpose): ADD offset (relative → absolute)
   ✓ Inverse (qpos_to_smpl): SUBTRACT offset (absolute → relative)
   ✓ Default trans: [0, 0, 0.91437225] (standing height)
   ✓ Only applied when count_offset=True (default)

================================================================================
                        CRITICAL CAVEATS
================================================================================

⚠️  EULER ORDER NOT FLEXIBLE
    - euler_order parameter exists in smpl_to_qpose()
    - But qpos_to_smpl() has "ZYX" HARDCODED on line 567
    - Changing one without the other BREAKS round-trip conversion

⚠️  QUATERNION FORMAT MISMATCH
    - scipy: [w, x, y, z]
    - MuJoCo: [x, y, z, w]
    - Must reorder at boundaries: [:, [3, 0, 1, 2]] and [:, [1, 2, 3, 0]]

⚠️  BODY JOINTS REORDERED, ROOT IS NOT
    - Root quaternion is part of translation concat, not curr_spose
    - Body joints are reordered via smpl_2_mujoco
    - This is intentional, not a bug

⚠️  body_pos[1] HANDLING ASYMMETRIC
    - smpl_to_qpose: uses count_offset parameter (default=True)
    - qpos_to_smpl: ALWAYS subtracts (no parameter to control)
    - Assume qpos_to_smpl always expects count_offset behavior

⚠️  DEFAULT STANDING HEIGHT
    - trans=None → z = 0.91437225 automatically
    - This is pelvis height, not ground height
    - Can break if character model differs

================================================================================
                        QUICK CODE SNIPPETS
================================================================================

EULER CONVERSION (Line 388):
  curr_spose = curr_spose.as_euler(euler_order, degrees=False)
  # Converts rotation matrix → ZYX Euler angles

QUATERNION REORDERING:
  Encoding (line 384):   as_quat()[:, [3, 0, 1, 2]]
  Decoding (line 563):   from_quat(quat[:, [1, 2, 3, 0]])

SMPL_2_MUJOCO MAPPING (lines 371-374):
  smpl_2_mujoco = [
      joint_names.index(q) for q in list(get_body_qposaddr(mj_model).keys())
      if q in joint_names
  ]

BODY POSITION OFFSET:
  Forward:  qpos[:, :3] = trans + mj_model.body_pos[1]
  Inverse:  trans = qpos[:, :3] - mj_model.body_pos[1]

================================================================================
                        FILE READING GUIDE
================================================================================

New to SMPL-MuJoCo conversions?
  → Start with SMPL_MUJOCO_SUMMARY.txt (5 min read)
  → Then read SMPL_MUJOCO_DETAILED_ANALYSIS.md (15 min read)
  → Reference SMPL_MUJOCO_CODE_REFERENCE.md for specifics

Implementing similar conversion?
  → Use SMPL_MUJOCO_CODE_REFERENCE.md as template
  → Cross-reference with actual code (lines 331-405, 552-571)
  → Watch out for the 5 caveats listed above

Debugging conversion issues?
  → Check Euler order (must be consistent)
  → Check quaternion reordering (scipy vs MuJoCo)
  → Verify body_pos[1] handling
  → Confirm smpl_2_mujoco reordering matches model

================================================================================
                        HELPER FUNCTION DEPS
================================================================================

Required imports (lines 17-32):
  from scipy.spatial.transform import Rotation as sRot
  from uhc.khrylib.utils import get_body_qposaddr
  from uhc.smpllib.smpl_parser import SMPL_BONE_ORDER_NAMES
  from uhc.utils.torch_geometry_transforms import (
      angle_axis_to_rotation_matrix,
      rotation_matrix_to_quaternion,
  )

Key functions called:
  - angle_axis_to_rotation_matrix(): axis-angle → rotation matrix
  - rotation_matrix_to_quaternion(): rotation matrix → quaternion
  - sRot.from_matrix(): 3x3 matrix → Rotation object
  - sRot.as_euler(): Rotation → Euler angles
  - sRot.from_euler(): Euler angles → Rotation
  - sRot.as_rotvec(): Rotation → axis-angle
  - get_body_qposaddr(): {body_name: (start_idx, end_idx)}

================================================================================
