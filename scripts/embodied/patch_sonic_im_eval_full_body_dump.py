#!/usr/bin/env python3
"""Extend the PhysFlow SONIC eval dump with full G1 articulation states.

This builds on ``patch_sonic_im_eval_trajectory_dump.py``.  The official
ImEvalCallback already records 14 tracked-body positions for metrics; this
patch additionally records IsaacLab's full robot body positions and quaternions
so the rollout can be inspected with the G1 mesh.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


FULL_SENTINEL = "PHYSFLOW_SONIC_FULL_BODY_STATE"


def patch_callback(path: Path) -> bool:
    text = path.read_text()
    if FULL_SENTINEL in text:
        return False

    backup = path.with_suffix(path.suffix + ".bak_physflow_fullbody")
    if not backup.exists():
        shutil.copy2(path, backup)

    pre_anchor = (
        "        self.pred_pos, self.pred_pos_all = [], []\n"
        "        self.pred_rot, self.pred_rot_all = [], []\n"
    )
    pre_insert = pre_anchor + (
        "        self.physflow_full_body_pos = []\n"
        "        self.physflow_full_body_quat = []\n"
        "        self.physflow_full_body_pos_all = []\n"
        "        self.physflow_full_body_quat_all = []\n"
        "        self.physflow_full_body_names = []\n"
        f"        self.physflow_full_body_state_marker = \"{FULL_SENTINEL}\"\n"
    )
    if pre_anchor not in text:
        raise RuntimeError(f"Could not locate eval-state initialization in {path}")
    text = text.replace(pre_anchor, pre_insert, 1)

    step_anchor = (
        "            self.gt_pos.append(gt_pos.cpu().numpy())\n"
        "            self.pred_pos.append(pred_pos.cpu().numpy())\n"
        "            self.mpjpe.append(mpjpe.cpu())\n\n"
    )
    step_insert = step_anchor + (
        "        if os.environ.get(\"PHYSFLOW_SONIC_DUMP_TRAJECTORY\", \"0\") == \"1\":\n"
        "            try:\n"
        "                robot = self.env.env.scene[\"robot\"]\n"
        "                self.physflow_full_body_pos.append(robot.data.body_pos_w.detach().cpu().numpy())\n"
        "                self.physflow_full_body_quat.append(robot.data.body_quat_w.detach().cpu().numpy())\n"
        "                if not self.physflow_full_body_names and hasattr(robot, \"body_names\"):\n"
        "                    self.physflow_full_body_names = list(robot.body_names)\n"
        "            except Exception as exc:  # noqa: BLE001\n"
        "                if not getattr(self, \"_physflow_full_body_warned\", False):\n"
        "                    print(f\"PhysFlow full-body trajectory dump unavailable: {exc}\")\n"
        "                    self._physflow_full_body_warned = True\n\n"
    )
    if step_anchor not in text:
        raise RuntimeError(f"Could not locate per-step position collection in {path}")
    text = text.replace(step_anchor, step_insert, 1)

    batch_anchor = (
        "            self.pred_pos_all += all_body_pos_pred\n"
        "            self.gt_pos_all += all_body_pos_gt\n"
        "            # self.pred_rot_all += all_body_rot_pred\n"
    )
    batch_insert = (
        "            self.pred_pos_all += all_body_pos_pred\n"
        "            self.gt_pos_all += all_body_pos_gt\n"
        "            if len(self.physflow_full_body_pos) > 0 and len(self.physflow_full_body_quat) > 0:\n"
        "                all_full_body_pos = np.stack(self.physflow_full_body_pos)\n"
        "                all_full_body_quat = np.stack(self.physflow_full_body_quat)\n"
        "                self.physflow_full_body_pos_all += [\n"
        "                    all_full_body_pos[: (i - 1), idx]\n"
        "                    for idx, i in enumerate(self.env._motion_lib.get_motion_num_steps(self.env.motion_ids))\n"
        "                ]\n"
        "                self.physflow_full_body_quat_all += [\n"
        "                    all_full_body_quat[: (i - 1), idx]\n"
        "                    for idx, i in enumerate(self.env._motion_lib.get_motion_num_steps(self.env.motion_ids))\n"
        "                ]\n"
        "            # self.pred_rot_all += all_body_rot_pred\n"
    )
    if batch_anchor not in text:
        raise RuntimeError(f"Could not locate batch trajectory aggregation in {path}")
    text = text.replace(batch_anchor, batch_insert, 1)

    save_anchor = (
        "            progress=np.asarray(progress_hist, dtype=np.float32),\n"
        "        )\n"
    )
    save_insert = (
        "            progress=np.asarray(progress_hist, dtype=np.float32),\n"
        "            full_body_names=np.array(getattr(self, \"physflow_full_body_names\", []), dtype=str),\n"
        "            full_body_pos=_pack(getattr(self, \"physflow_full_body_pos_all\", [])),\n"
        "            full_body_quat=_pack(getattr(self, \"physflow_full_body_quat_all\", [])),\n"
        "        )\n"
    )
    if save_anchor not in text:
        raise RuntimeError(f"Could not locate np.savez trajectory fields in {path}")
    text = text.replace(save_anchor, save_insert, 1)

    path.write_text(text)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sonic-repo", required=True, type=Path)
    args = parser.parse_args()
    callback = args.sonic_repo / "gear_sonic/trl/callbacks/im_eval_callback.py"
    changed = patch_callback(callback)
    print(f"{'patched' if changed else 'already_patched'} {callback}")


if __name__ == "__main__":
    main()
