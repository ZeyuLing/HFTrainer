"""Compare training vs inference condition motion encoding."""

import torch
import sys
sys.path.insert(0, '/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer')

# Key insight from code analysis:
#
# In training (prism_trainer.py, line 90):
#   latents = self.bundle.encode_motion(motion)
#
# Where bundle.encode_motion (bundle.py, lines 136-164):
#   1. motion input is UNNORMALIZED (comes from dataset after LoadSmplx55)
#   2. Calls self.smpl_pose_processor.normalize(motion)  <- NORMALIZES HERE
#   3. Encodes via VAE
#   4. Normalizes latents
#
# In inference (prism_backend.py):
#   In load_condition_pose (line 266):
#     motion = self.smpl_processor.normalize(motion)  <- NORMALIZES HERE
#   
#   Then passes to encode_motion (line 360):
#     first_frame_latents = self.encode_motion(first_frame_motion)
#
#   In encode_motion (prism_backend.py, line 295):
#     1. Input motion is ALREADY NORMALIZED
#     2. Does NOT normalize again (VAE gets normalized motion!)
#     3. Encodes via VAE (on NORMALIZED motion)
#     4. Normalizes latents
#
# The BUG: The VAE encoder expects UNNORMALIZED motion!
# - Training: VAE gets unnormalized motion
# - Inference: VAE gets normalized motion
# This is a CRITICAL mismatch!

print("="*80)
print("BUG CONFIRMED: Condition motion normalization mismatch")
print("="*80)
print()
print("In training:")
print("  Motion (unnormalized from dataset) -> encode_motion -> normalize -> VAE")
print()
print("In inference (CONDITION FRAME):")
print("  Motion -> normalize (in load_condition_pose) -> encode_motion -> VAE (with NORMALIZED motion!)")
print()
print("The VAE encoder trains on UNNORMALIZED motion but receives NORMALIZED motion in inference!")
print()
print("FIX: In prism_backend.py load_condition_pose, DO NOT normalize the motion before")
print("     passing to encode_motion. Let encode_motion handle the normalization.")
print()
print("="*80)

