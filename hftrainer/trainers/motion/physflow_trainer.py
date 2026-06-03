"""PhysFlowTrainer: online adversarial fine-tuning of the KIMODO-G1 generator.

Strategy (Stage 1 -- online best-of-N reward-weighted SFT):
  For each training step, with the *current* generator:
    1. sample N motions per prompt from cached text embeddings (no 8B encoder);
    2. score each with a FROZEN judge tracker in MuJoCo (physics realism);
    3. select the best (most trackable) motion per prompt;
    4. take a supervised x0 diffusion step toward the selected motions
       (optionally reward-weighted across the N candidates).

This is genuinely *online* (samples come from the live policy every step) and
*adversarial* (the frozen tracker's physics reward pushes the generator toward
motions the robot can actually execute). It does not require a differentiable
sampler; true DDPO (policy-gradient through a stochastic sampler) is a planned
Stage-2 extension on top of this loop.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any, Dict, List, Optional

import torch

from hftrainer.registry import TRAINERS
from hftrainer.trainers.base_trainer import BaseTrainer


@TRAINERS.register_module()
class PhysFlowTrainer(BaseTrainer):
    """Online best-of-N reward-weighted SFT for PhysFlow."""

    def __init__(
        self,
        bundle,
        num_samples: int = 4,
        diffusion_steps: int = 30,
        cfg_weight: Optional[List[float]] = None,
        cfg_type: Optional[str] = None,
        reward_weighted: bool = False,
        reward_temperature: float = 0.5,
        judge_onnx: Optional[str] = None,
        judge_mjcf: Optional[str] = None,
        rollout_dir: Optional[str] = None,
        keep_rollouts: bool = False,
        enable_reward: bool = True,
        # ---- anti-collapse (RAFT/ReST) controls ----
        accept_min_completion: float = 0.9,
        accept_require_no_fall: bool = True,
        accept_max_score: Optional[float] = None,
        anchor_weight: float = 0.5,
        # ---- anti-freeze: reject degenerate "frozen-pose glide" candidates ----
        # A static pose is trivially trackable (no fall, completion 1.0, ~0 joint
        # error), so a pure trackability reward has a degenerate optimum at "don't
        # move the joints". We reject candidates whose articulation (temporal std
        # of joint angles over the valid window, rad) falls below this floor, and
        # optionally reject pure-translation slides (large root displacement with
        # near-frozen joints). Base KIMODO locomotion has joint_std ~0.09-0.16;
        # collapsed glides ~0.013 -- a 0.05 floor separates them cleanly.
        accept_min_joint_std: float = 0.0,
        accept_max_root_disp_if_frozen: Optional[float] = None,
        accept_frozen_joint_std: float = 0.03,
        # ---- trainee co-training: export accepted motions to a growing pool ----
        tracker_pool_dir: Optional[str] = None,
        pool_max_motions: int = 4000,
        **kwargs,
    ) -> None:
        super().__init__(bundle)
        self.num_samples = int(num_samples)
        self.diffusion_steps = int(diffusion_steps)
        self.cfg_weight = cfg_weight
        self.cfg_type = cfg_type
        self.reward_weighted = bool(reward_weighted)
        self.reward_temperature = float(reward_temperature)
        self.judge_onnx = judge_onnx
        self.judge_mjcf = judge_mjcf
        self.rollout_dir = rollout_dir
        self.keep_rollouts = bool(keep_rollouts)
        self.enable_reward = bool(enable_reward)
        self.accept_min_completion = float(accept_min_completion)
        self.accept_require_no_fall = bool(accept_require_no_fall)
        self.accept_max_score = accept_max_score
        self.anchor_weight = float(anchor_weight)
        self.accept_min_joint_std = float(accept_min_joint_std)
        self.accept_max_root_disp_if_frozen = accept_max_root_disp_if_frozen
        self.accept_frozen_joint_std = float(accept_frozen_joint_std)
        self.tracker_pool_dir = tracker_pool_dir
        self.pool_max_motions = int(pool_max_motions)
        self._reward = None
        if self.tracker_pool_dir:
            os.makedirs(self.tracker_pool_dir, exist_ok=True)

    def _export_to_pool(self, proto_dir: str, selected: List[tuple], prompt_ids: List[str]) -> int:
        """Copy accepted (trackable) ``.motion`` files into the shared tracker
        pool so the trainee (ProtoMotions PPO+AMP) can co-train on the live
        generator distribution. ``selected`` is a list of (b, best_local) for
        prompts whose best candidate was acceptable. Returns #exported."""
        import glob
        import shutil

        if not self.tracker_pool_dir:
            return 0
        step = self._global_step()
        n = 0
        for b, best_local in selected:
            stem = f"p{b:03d}_s{best_local:02d}"
            srcs = glob.glob(os.path.join(proto_dir, f"{stem}*.motion"))
            if not srcs:
                continue
            pid = prompt_ids[b] if b < len(prompt_ids) and prompt_ids[b] else f"b{b}"
            dst = os.path.join(self.tracker_pool_dir, f"it{step:06d}_{pid}_{stem}.motion")
            try:
                shutil.copy2(srcs[0], dst)
                n += 1
            except Exception:
                pass
        # cap pool size (keep most recent by mtime)
        try:
            allm = sorted(
                glob.glob(os.path.join(self.tracker_pool_dir, "*.motion")),
                key=lambda p: os.path.getmtime(p),
            )
            for old in allm[: max(0, len(allm) - self.pool_max_motions)]:
                os.remove(old)
        except Exception:
            pass
        return n

    @staticmethod
    def _motion_dynamics(qpos: "np.ndarray", length: int) -> Dict[str, float]:
        """Per-candidate articulation/translation stats from generated qpos.

        ``qpos`` is [T, 36] (root pos[:3] + root quat[3:7] + 29 joints[7:]).
        ``joint_std`` is the mean over joints of the temporal std of joint angles
        over the valid window -- a direct measure of how much the body actually
        moves. ``root_disp`` is the start->end root translation (m). A frozen-pose
        glide has tiny ``joint_std`` with large ``root_disp``.
        """
        import numpy as np
        a = np.asarray(qpos)[: max(int(length), 1)]
        if a.ndim == 1:
            a = a[None]
        joints = a[:, 7:] if a.shape[1] > 7 else a
        joint_std = float(np.std(joints, axis=0).mean()) if a.shape[0] > 1 else 0.0
        root_disp = float(np.linalg.norm(a[-1, :3] - a[0, :3])) if a.shape[1] >= 3 else 0.0
        return {"joint_std": joint_std, "root_disp": root_disp}

    def _is_acceptable(self, m: Dict[str, float]) -> bool:
        """A candidate is an acceptable SFT target only if the robot can actually
        execute it AND it is a non-degenerate motion: no fall + sufficient
        completion (+ optional score ceiling) + enough articulation (anti-freeze).

        Rejecting fallen/failed motions stops collapse onto *untrackable* modes;
        rejecting frozen-pose glides stops collapse onto the opposite degenerate
        mode -- a static pose that is trivially trackable but is not the motion the
        prompt asked for (legs frozen while the root slides across the floor)."""
        if not self.enable_reward:
            return True
        if "error" in m:
            return False
        if self.accept_require_no_fall and bool(m.get("fall_detected", True)):
            return False
        if float(m.get("completion", 0.0)) < self.accept_min_completion:
            return False
        if self.accept_max_score is not None and float(m.get("score", 1e9)) > self.accept_max_score:
            return False
        # anti-freeze: reject candidates whose joints barely move.
        js = m.get("joint_std")
        if js is not None:
            if self.accept_min_joint_std > 0.0 and float(js) < self.accept_min_joint_std:
                return False
            # reject pure-translation slides (root travels far on near-frozen legs).
            if (self.accept_max_root_disp_if_frozen is not None
                    and float(js) < self.accept_frozen_joint_std
                    and float(m.get("root_disp", 0.0)) > self.accept_max_root_disp_if_frozen):
                return False
        return True

    # ----------------------------------------------------------- reward (lazy)
    @property
    def reward(self):
        # The co-evolution orchestrator hot-swaps the judge between outer rounds
        # by writing a JSON judge spec and pointing PHYSFLOW_JUDGE_SPEC at it
        # (frozen / latest-trainee / blended). We rebuild the reward whenever the
        # spec file changes so a relaunched-per-round generator always scores
        # against the *current* judge ensemble. Falls back to the single frozen
        # judge_onnx when no spec is set (the original Stage-1 behaviour).
        from hftrainer.models.motion.physflow.reward import PhysicsJudgeReward

        spec = os.environ.get("PHYSFLOW_JUDGE_SPEC")
        if spec and os.path.isfile(spec):
            sig = (spec, os.path.getmtime(spec))
            if self._reward is None or getattr(self, "_judge_sig", None) != sig:
                self._reward = PhysicsJudgeReward.from_spec_file(spec, mjcf_path=self.judge_mjcf)
                self._judge_sig = sig
            return self._reward
        if self._reward is None:
            self._reward = PhysicsJudgeReward(onnx_path=self.judge_onnx, mjcf_path=self.judge_mjcf)
        return self._reward

    # ----------------------------------------------------------------- helpers
    def _global_step(self) -> int:
        try:
            return int(self.get_global_step())
        except Exception:
            return 0

    def _score_samples(
        self, qpos: "torch.Tensor", num_frames: List[int], group_size: int, work_dir: str
    ) -> List[Dict[str, float]]:
        """Write per-sample CSVs (trimmed to length) and score them. Returns a
        list of metric dicts aligned with the flat [B*N] sample order."""
        import numpy as np

        csv_dir = os.path.join(work_dir, "csv")
        os.makedirs(csv_dir, exist_ok=True)
        stems = []
        for flat_idx in range(qpos.shape[0]):
            b = flat_idx // group_size
            length = int(num_frames[b])
            stem = f"p{b:03d}_s{flat_idx % group_size:02d}"
            stems.append(stem)
            sample = np.asarray(qpos[flat_idx])[:length]
            self.bundle.save_qpos_csv(sample, os.path.join(csv_dir, f"{stem}.csv"))

        if not self.enable_reward:
            return [{"score": 0.0} for _ in stems]

        scored = self.reward.score_csv_dir(csv_dir, work_dir)
        return [scored.get(stem, {"score": self.reward.error_penalty}) for stem in stems]

    # -------------------------------------------------------------- train step
    def train_step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        text_feat = batch["text_feat"]            # [B, seq, 4096]
        text_pad_mask = batch["text_pad_mask"]    # [B, seq]
        num_frames = list(batch["num_frames"])    # [B]
        B = text_feat.shape[0]
        N = self.num_samples

        # Expand each prompt to N candidates: flat order = prompt-major.
        feat_rep = text_feat.repeat_interleave(N, dim=0)          # [B*N, seq, 4096]
        mask_rep = text_pad_mask.repeat_interleave(N, dim=0)      # [B*N, seq]
        lengths_rep = torch.tensor(
            [nf for nf in num_frames for _ in range(N)], dtype=torch.long
        )

        # 1) sample candidate motions from the live policy (no grad)
        latents = self.bundle.sample_latents(
            feat_rep, mask_rep, lengths_rep,
            diffusion_steps=self.diffusion_steps,
            cfg_weight=self.cfg_weight, cfg_type=self.cfg_type,
        )  # [B*N, Tmax, D]
        qpos = self.bundle.latents_to_qpos(latents)  # numpy [B*N, Tmax, 36]

        # 2) score with the frozen judge tracker (lower == more trackable)
        ctx = tempfile.TemporaryDirectory(
            prefix=f"physflow_step{self._global_step()}_", dir=self.rollout_dir
        )
        try:
            metrics = self._score_samples(qpos, num_frames, N, ctx.name)
            # attach articulation/translation stats so the accept filter can
            # reject degenerate frozen-pose glides (anti-freeze gate).
            for b in range(B):
                for j in range(N):
                    flat = b * N + j
                    metrics[flat].update(self._motion_dynamics(qpos[flat], num_frames[b]))
            scores = torch.tensor([m.get("score", 0.0) for m in metrics], dtype=torch.float32)

            # 3) per prompt: prefer the best *acceptable* candidate; mark whether
            #    any candidate was acceptable (good_mask) so unacceptable prompts
            #    contribute zero SFT gradient (only the anchor regularizes them).
            target_latents = []
            target_lengths = []
            sel_text_feat = []
            sel_text_mask = []
            good_flags = []
            selected_good = []   # (b, best_local) for accepted prompts -> pool
            best_scores, mean_scores, sel_joint_stds = [], [], []
            for b in range(B):
                g = scores[b * N:(b + 1) * N]
                metrics_b = metrics[b * N:(b + 1) * N]
                acceptable = [i for i in range(N) if self._is_acceptable(metrics_b[i])]
                if acceptable:
                    best_local = min(acceptable, key=lambda i: float(g[i]))
                    good_flags.append(1.0)
                    selected_good.append((b, best_local))
                else:
                    best_local = int(torch.argmin(g).item())
                    good_flags.append(0.0)
                best_scores.append(float(g[best_local]))
                mean_scores.append(float(g.mean()))
                sel_joint_stds.append(float(metrics_b[best_local].get("joint_std", 0.0)))
                flat = b * N + best_local
                target_latents.append(latents[flat])
                target_lengths.append(int(num_frames[b]))
                sel_text_feat.append(text_feat[b])
                sel_text_mask.append(text_pad_mask[b])

            target = torch.stack(target_latents, dim=0).detach()      # [B, Tmax, D]
            sel_feat = torch.stack(sel_text_feat, dim=0)              # [B, seq, 4096]
            sel_mask = torch.stack(sel_text_mask, dim=0)
            lengths = torch.tensor(target_lengths, dtype=torch.long)
            good_mask = torch.tensor(good_flags, dtype=torch.float32)

            # export accepted motions to the shared trainee pool (closed loop)
            n_pooled = 0
            if self.tracker_pool_dir and selected_good:
                n_pooled = self._export_to_pool(
                    os.path.join(ctx.name, "proto"),
                    selected_good,
                    list(batch.get("prompt_id", [])),
                )
        finally:
            if not self.keep_rollouts:
                ctx.cleanup()

        # 4) reward-filtered + anchored x0 step toward the accepted motions
        out = self.bundle.sft_loss(
            sel_feat, sel_mask, target, lengths,
            good_mask=good_mask, anchor_weight=self.anchor_weight,
        )

        result: Dict[str, Any] = {"loss": out["loss"]}
        result["loss_sft"] = out["sft_mse"]
        result["n_good"] = out.get("n_good", torch.tensor(float(sum(good_flags))))
        if self.tracker_pool_dir:
            result["n_pooled"] = torch.tensor(float(n_pooled))
        if "anchor_mse" in out:
            result["loss_anchor"] = out["anchor_mse"]
        result["reward_best_mean"] = torch.tensor(sum(best_scores) / max(B, 1))
        result["reward_cand_mean"] = torch.tensor(sum(mean_scores) / max(B, 1))
        # articulation telemetry: mean joint_std of the SELECTED targets. If this
        # trends toward ~0 the policy is collapsing into frozen-pose glides.
        result["sel_joint_std_mean"] = torch.tensor(sum(sel_joint_stds) / max(B, 1))
        return result
