# PRISM 1B text+pose-to-motion, multi-frame conditioning (1/5/9 frames)
#
# Resume from versatilemotion checkpoint (iter=15000):
#   bash tools/taiji_dist_train.sh configs/prism/prism_1b_tp2m_multiframe.py --auto-resume
#
# This stage fine-tunes from the 1-frame pretrained model with multi-frame
# pose conditioning (condition_num_frames=[1, 5, 9], frame_condition_rate=0.1).

_base_ = './prism_1b_tp2m_1frame.py'

trainer = dict(
    condition_num_frames=[1, 5, 9],
    frame_condition_rate=0.1,
)
