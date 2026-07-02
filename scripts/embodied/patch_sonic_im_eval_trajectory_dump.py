#!/usr/bin/env python3
"""Install an opt-in trajectory dump into SONIC's official IsaacLab eval.

The upstream ImEvalCallback computes reference and simulated body positions
for metrics but only writes scalar summaries.  This patch preserves upstream
behavior unless PHYSFLOW_SONIC_DUMP_TRAJECTORY=1 is set, then writes the
per-motion reference and rollout body trajectories for visualization.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


SENTINEL = "PHYSFLOW_SONIC_DUMP_TRAJECTORY"


HELPER_METHOD = r'''
    def _physflow_save_trajectory_dump(self, terminate_hist, progress_hist, motion_idxes):
        if os.environ.get("PHYSFLOW_SONIC_DUMP_TRAJECTORY", "0") != "1":
            return
        dump_dir = os.environ.get("PHYSFLOW_SONIC_TRAJECTORY_DIR") or self.output_dir
        if dump_dir is None:
            return
        os.makedirs(dump_dir, exist_ok=True)
        body_names = []
        if hasattr(self.env, "motion_command") and hasattr(self.env.motion_command, "cmd_body_names"):
            body_names = list(self.env.motion_command.cmd_body_names)

        def _pack(items):
            try:
                return np.stack(items)
            except ValueError:
                return np.array(items, dtype=object)

        path = os.path.join(dump_dir, "physflow_sonic_trajectories.npz")
        np.savez_compressed(
            path,
            pred_pos=_pack(self.pred_pos_all),
            gt_pos=_pack(self.gt_pos_all),
            body_names=np.array(body_names, dtype=str),
            motion_keys=np.array(self.env._motion_lib._motion_data_keys[motion_idxes], dtype=str),
            terminated=np.asarray(terminate_hist, dtype=bool),
            progress=np.asarray(progress_hist, dtype=np.float32),
        )
        print(f"Saved PhysFlow SONIC trajectories to {path}")
'''


CALL_INSERT = '''                if self.eval_only and self.accelerator.is_main_process:
                    self._physflow_save_trajectory_dump(
                        gathered_terminate_hist_stack.cpu().numpy(),
                        gathered_progress_hist_stack.cpu().numpy(),
                        gathered_motion_idxes.cpu().numpy(),
                    )

'''


def patch_callback(path: Path) -> bool:
    text = path.read_text()
    if SENTINEL in text:
        return False

    backup = path.with_suffix(path.suffix + ".bak_physflow_trajdump")
    if not backup.exists():
        shutil.copy2(path, backup)

    helper_anchor = "    def _post_evaluate_policy(self, eval_res):\n"
    if helper_anchor not in text:
        raise RuntimeError(f"Could not locate helper insertion point in {path}")
    text = text.replace(helper_anchor, HELPER_METHOD + "\n" + helper_anchor, 1)

    call_anchor = (
        "                actor_state[\"failed_idxes\"] = (\n"
        "                    gathered_terminate_hist_stack.cpu().numpy().nonzero()[0]\n"
        "                )\n\n"
    )
    if call_anchor not in text:
        raise RuntimeError(f"Could not locate trajectory dump insertion point in {path}")
    text = text.replace(call_anchor, call_anchor + CALL_INSERT, 1)

    path.write_text(text)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sonic-repo",
        required=True,
        type=Path,
        help="Path to the GR00T-WholeBodyControl checkout.",
    )
    args = parser.parse_args()
    callback = args.sonic_repo / "gear_sonic/trl/callbacks/im_eval_callback.py"
    changed = patch_callback(callback)
    print(f"{'patched' if changed else 'already_patched'} {callback}")


if __name__ == "__main__":
    main()
