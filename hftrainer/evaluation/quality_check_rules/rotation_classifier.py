"""Joint rotation validity classifier inference helpers."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ResidualBlock(nn.Module):
    """Linear -> BN -> ReLU -> Dropout -> Linear -> BN + residual."""

    def __init__(self, dim: int, dropout: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.block(x) + x)


class RotationClassifierMLP(nn.Module):
    """MLP classifier for a flattened 3x3 rotation matrix."""

    def __init__(
        self,
        input_dim: int = 9,
        hidden_dim: int = 64,
        num_res_blocks: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, dropout) for _ in range(num_res_blocks)]
        )
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        h = self.res_blocks(h)
        return self.output(h)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))


class JointRotationClassifier:
    """Loads per-joint MLPs and runs batched validity inference."""

    DEFAULT_THRESHOLD = 0.5

    def __init__(
        self,
        model_path: str,
        device: Optional[torch.device] = None,
        threshold: float = DEFAULT_THRESHOLD,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = float(threshold)

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {model_path}")

        logger.info("Loading rotation classifier from %s on %s", model_path, self.device)
        checkpoint = torch.load(model_path, map_location=self.device)

        model_config: Dict = checkpoint.get("model_config", {})
        input_dim = int(model_config.get("input_dim", 9))
        hidden_dim = int(model_config.get("hidden_dim", 64))
        num_res_blocks = int(model_config.get("num_res_blocks", 3))
        dropout = float(model_config.get("dropout", 0.2))

        self.models: Dict[int, RotationClassifierMLP] = {}
        joint_models: Dict = checkpoint.get("model_state_dicts") or checkpoint.get("joint_models", {})
        for joint_id, state_dict in joint_models.items():
            joint_id_int = int(joint_id)
            model = RotationClassifierMLP(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_res_blocks=num_res_blocks,
                dropout=dropout,
            )
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            self.models[joint_id_int] = model

        logger.info("Loaded classifiers for joints: %s", sorted(self.models.keys()))

    @torch.no_grad()
    def predict(self, joint_id: int, rotation: np.ndarray) -> Tuple[float, bool]:
        if joint_id not in self.models:
            raise KeyError(f"No trained model for joint {joint_id}")
        rotation = np.asarray(rotation, dtype=np.float32)
        if rotation.shape != (3, 3):
            raise ValueError(f"Expected rotation shape (3, 3), got {rotation.shape}")
        x = torch.from_numpy(rotation.reshape(1, 9)).to(self.device)
        prob = float(self.models[joint_id].predict_proba(x).item())
        return prob, prob >= self.threshold

    @torch.no_grad()
    def predict_batch(self, joint_id: int, rotations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if joint_id not in self.models:
            raise KeyError(f"No trained model for joint {joint_id}")
        rotations = np.asarray(rotations, dtype=np.float32)
        if rotations.ndim != 3 or rotations.shape[1:] != (3, 3):
            raise ValueError(f"Expected rotations shape (N, 3, 3), got {rotations.shape}")
        x = torch.from_numpy(rotations.reshape(rotations.shape[0], 9)).to(self.device)
        probs = self.models[joint_id].predict_proba(x).squeeze(-1).cpu().numpy()
        return probs, probs >= self.threshold

    @torch.no_grad()
    def predict_all_joints_batch(
        self,
        joint_ids: List[int],
        rotations: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict probabilities for all requested joints.

        Args:
            joint_ids: Joint ids to run.
            rotations: Array shaped ``(T, J, 3, 3)`` indexed by absolute joint id.
        """
        rotations = np.asarray(rotations, dtype=np.float32)
        if rotations.ndim != 4 or rotations.shape[2:] != (3, 3):
            raise ValueError(f"Expected rotations shape (T, J, 3, 3), got {rotations.shape}")

        checked = [joint_id for joint_id in joint_ids if joint_id in self.models]
        num_frames = int(rotations.shape[0])
        if not checked:
            return np.zeros((num_frames, 0), dtype=np.float32), np.zeros((num_frames, 0), dtype=bool)

        probs_matrix = np.empty((num_frames, len(checked)), dtype=np.float32)
        start = time.perf_counter()
        for idx, joint_id in enumerate(checked):
            x = torch.from_numpy(rotations[:, joint_id].reshape(num_frames, 9)).to(self.device)
            probs_matrix[:, idx] = self.models[joint_id].predict_proba(x).squeeze(-1).cpu().numpy()
        logger.debug(
            "predict_all_joints_batch: T=%d joints=%d device=%s elapsed=%.3fs",
            num_frames,
            len(checked),
            self.device,
            time.perf_counter() - start,
        )
        return probs_matrix, probs_matrix >= self.threshold

    def get_available_joints(self) -> List[int]:
        return sorted(self.models.keys())

    @classmethod
    def find_latest_model(cls, classifiers_dir: str) -> Optional[str]:
        path = Path(classifiers_dir)
        if not path.is_dir():
            return None
        candidates = sorted(path.glob("joint_classifiers_*.pt"))
        if not candidates:
            return None
        return str(candidates[-1])
