#!/usr/bin/env python3
"""Parallel multi-node PRISM HumanML3D test-split generation on Taiji.

Re-runs the paper Table-1 T2M generation with the CLEANED inference pipeline
(the destructive first-chunk post-process has been removed). For each checkpoint
we launch NODES single-host jobs of NGPU GPUs each; together they cover NSHARDS
disjoint shards of the test list (run_gen_node.sh handles per-GPU sharding via
SHARD_START..SHARD_START+NGPU). All nodes for one checkpoint write to the SAME
OUT dir and use --skip-existing, so the work is split, not duplicated.

Usage:
    TOKEN=<taiji_token> python3 scripts/submit/submit_prism_h3d_parallel.py [--dry-run]
"""
import argparse
import sys

PROJ = "/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer"
sys.path.insert(0, PROJ + "/tools")
from taiji_submit import submit  # noqa: E402

ANNO = "data/annotation/test_hml3d.json"
REWRITTEN = "data/annotation/test_hml3d_rewritten.json"

# job-tag -> inference config (must include text_encoder) + checkpoint dir + OUT
JOBS = {
    "iter15k": dict(
        cfg="configs/prism/prism_1b_tp2m_multiframe_iter15k.py",
        ckpt="work_dirs/prism_1b_tp2m_multiframe/checkpoint-iter_15000",
        out="outputs/evaluation/prism_paper_iter15000_clean0603/h3d",
    ),
    "ktspectral": dict(
        cfg="configs/prism/prism_1b_tp2m_multiframe_kt_spectral_unified.py",
        # stable snapshot of the latest training checkpoint (epoch_4, 08:39),
        # used instead of the live dir which the running job overwrites.
        ckpt="work_dirs/_snap_ktspectral_latest_0603",
        out="outputs/evaluation/prism_kt_spectral_latest_clean0603/h3d",
    ),
}


def start_cmd(j, nshards, shard_start, ngpu):
    return (
        f"cd {PROJ} && export PYTHONPATH=$PWD "
        f"CONFIG={j['cfg']} CKPT={j['ckpt']} MODE=none "
        f"ANNO={ANNO} REWRITTEN={REWRITTEN} OUT={j['out']} "
        f"NSHARDS={nshards} SHARD_START={shard_start} NGPU={ngpu} "
        f"&& bash scripts/eval/run_gen_node.sh"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ngpu", type=int, default=8)
    ap.add_argument("--nodes", type=int, default=4, help="hosts per checkpoint")
    ap.add_argument("--models", nargs="+", default=list(JOBS.keys()))
    ap.add_argument("--elastic", action="store_true", default=True)
    ap.add_argument("--no-elastic", dest="elastic", action="store_false")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    nshards = args.ngpu * args.nodes
    for tag in args.models:
        j = JOBS[tag]
        for node in range(args.nodes):
            ss = node * args.ngpu
            flag = f"prism_{tag}_clean0603_s{ss}"
            cmd = start_cmd(j, nshards, ss, args.ngpu)
            print(f"\n{'='*60}\nJob: {flag}  ({args.ngpu}xV100, elastic={args.elastic})")
            print(f"  shards [{ss}..{ss+args.ngpu-1}]/{nshards} -> {j['out']}")
            print("  CMD:", cmd[:240], "...")
            if args.dry_run:
                print("  [DRY RUN]")
                continue
            submit(
                task_flag=flag,
                config_path=j["cfg"],  # ignored when start_cmd_override is set
                host_num=1,
                elastic=args.elastic,
                start_cmd_override=cmd,
                host_gpu_num=args.ngpu,
            )


if __name__ == "__main__":
    main()
