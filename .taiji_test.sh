#!/bin/bash
nvidia-smi > /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/.taiji_test_output.txt 2>&1
hostname >> /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/.taiji_test_output.txt 2>&1
echo "DONE" >> /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/.taiji_test_output.txt

