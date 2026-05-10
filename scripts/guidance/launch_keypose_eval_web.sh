#!/bin/bash
# Run on debug machine: preprocess eval data + run MoGenDIT eval + start web server
set -e
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer

echo "=== Step 1: Preprocess existing eval data for web ==="
python3 scripts/preprocess_keypose_eval_for_web.py

echo "=== Step 2: Start web server on port 8095 ==="
# Kill any existing server on 8095
kill $(lsof -t -i:8095) 2>/dev/null || true
nohup python3 motion_annot_web/keypose_eval/app.py --port 8095 > output/eval_keyframe_pose/web_server.log 2>&1 &
echo "Web server PID: $!"
echo "Server starting at http://$(hostname -I | awk '{print $1}'):8095"

echo "=== Done ==="
