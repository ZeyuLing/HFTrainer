#!/bin/bash
cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
python3 scripts/ab_test_e14_pad_strategies.py > /tmp/ab_test_pad_strategies.log 2>&1
echo "DONE: exit code $?" >> /tmp/ab_test_pad_strategies.log
