#!/usr/bin/env python3
"""Install an opt-in MuJoCo qpos logger into the SONIC simulator loop.

The official SONIC deploy logger records commanded/measured joints but does not
export the simulator floating-base state needed by the unified G1 tracker
evaluator. This patch keeps the upstream path unchanged unless
SONIC_QPOS_LOGFILE is set.
"""

from __future__ import annotations

import argparse
from pathlib import Path


IMPORT_SENTINEL = "from pathlib import Path"
LOGGER_SENTINEL = "SONIC_QPOS_LOGFILE"


def patch_base_sim(path: Path) -> bool:
    text = path.read_text()
    if LOGGER_SENTINEL in text:
        return False

    if "import os\n" not in text:
        text = text.replace("import time\n", "import time\nimport os\n", 1)
    if IMPORT_SENTINEL not in text:
        text = text.replace("import os\n", "import os\nfrom pathlib import Path\n", 1)

    start_sig = "    def start(self):\n        try:\n"
    start_insert = (
        "    def start(self):\n"
        "        qpos_log_file = None\n"
        "        qpos_log_path = os.environ.get(\"SONIC_QPOS_LOGFILE\")\n"
        "        if qpos_log_path:\n"
        "            Path(qpos_log_path).parent.mkdir(parents=True, exist_ok=True)\n"
        "            qpos_log_file = open(qpos_log_path, \"w\", buffering=1)\n"
        "            qpos_log_file.write(\n"
        "                \"frame,time,\" + \",\".join(f\"qpos_{i}\" for i in range(36)) + \"\\n\"\n"
        "            )\n"
        "            print(f\"[SONIC qpos logger] writing MuJoCo qpos to {qpos_log_path}\", flush=True)\n"
        "        try:\n"
    )
    if start_sig not in text:
        raise RuntimeError(f"Could not locate BaseSimulator.start() prologue in {path}")
    text = text.replace(start_sig, start_insert, 1)

    step_sig = "                self.sim_env.sim_step()\n"
    step_insert = (
        "                self.sim_env.sim_step()\n"
        "                if qpos_log_file is not None:\n"
        "                    obs = self.sim_env.obs if self.sim_env.obs is not None else self.sim_env.prepare_obs()\n"
        "                    qpos = np.concatenate([obs[\"floating_base_pose\"], obs[\"body_q\"]]).astype(float)\n"
        "                    qpos_log_file.write(\n"
        "                        f\"{sim_cnt},{self.sim_env.mj_data.time:.9f},\"\n"
        "                        + \",\".join(f\"{x:.9f}\" for x in qpos[:36])\n"
        "                        + \"\\n\"\n"
        "                    )\n"
    )
    if step_sig not in text:
        raise RuntimeError(f"Could not locate sim_step() line in {path}")
    text = text.replace(step_sig, step_insert, 1)

    finally_sig = "        finally:\n            self.close()\n"
    finally_insert = (
        "        finally:\n"
        "            if qpos_log_file is not None:\n"
        "                qpos_log_file.close()\n"
        "            self.close()\n"
    )
    if finally_sig not in text:
        raise RuntimeError(f"Could not locate BaseSimulator.start() finally block in {path}")
    text = text.replace(finally_sig, finally_insert, 1)

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
    base_sim = args.sonic_repo / "gear_sonic/utils/mujoco_sim/base_sim.py"
    changed = patch_base_sim(base_sim)
    print(f"{'patched' if changed else 'already_patched'} {base_sim}")


if __name__ == "__main__":
    main()
