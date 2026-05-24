"""PhysFlow Curriculum: Adaptive text prompt scheduler for PhysFlow training.

Organizes prompts by difficulty level and advances based on physics correction
success rate. Easier motions (standing, slow walk) have higher physics success
rates, allowing the model to learn basic physical plausibility first before
progressing to harder motions.

Curriculum Levels:
  0 - standing:    Static poses, weight shifts (easiest for RL tracker)
  1 - walking:     Locomotion, stepping, slow movements
  2 - upper_body:  Upper body gestures, waving, reaching
  3 - transitions: Direction changes, combined locomotion + gesture
  4 - dynamic:     Fast movements, kicks, squats, balance challenges (hardest)

Usage:
    curriculum = PhysFlowCurriculum()

    # Direction A: get prompt for current difficulty
    prompt = curriculum.get_prompt()
    num_frames = curriculum.get_num_frames()
    # ... generate motion, physics correct, train ...
    curriculum.update(success=physics_oracle.is_good_quality(stats))

    # Direction B: get diverse prompts for RL training
    prompts = [curriculum.get_diverse_prompt() for _ in range(100)]
"""

from __future__ import annotations

import logging
import random
from collections import deque
from typing import Dict, List, Optional

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Curriculum Level Definitions
# ═══════════════════════════════════════════════════════════════════════════════

PHYSFLOW_LEVELS = [
    {
        "level": 0,
        "name": "standing",
        "description": "Static poses and weight shifts",
        "prompts": [
            "a person stands still",
            "a person shifts weight from left to right foot",
            "a person looks around while standing",
            "a person stands with arms at their sides",
            "a person stands and breathes deeply",
            "a person stands in a relaxed pose",
            "a person stands with hands on hips",
            "a person stands at attention",
            "a person stands and stretches their neck",
            "a person stands with arms crossed",
            "a person stands on one leg briefly",
            "a person stands and rotates their shoulders",
        ],
        "min_success_rate": 0.8,
        "num_frames": 90,  # 3s at 30fps
    },
    {
        "level": 1,
        "name": "walking",
        "description": "Simple locomotion",
        "prompts": [
            "a person walks forward slowly",
            "a person walks in a straight line",
            "a person takes a few steps forward",
            "a person walks forward at a normal pace",
            "a person walks and then stops",
            "a person walks backward slowly",
            "a person sidesteps to the left",
            "a person sidesteps to the right",
            "a person walks with long strides",
            "a person walks with short steps",
            "a person paces back and forth",
            "a person walks casually",
            "a person marches in place",
        ],
        "min_success_rate": 0.7,
        "num_frames": 120,  # 4s at 30fps
    },
    {
        "level": 2,
        "name": "upper_body",
        "description": "Upper body gestures while standing",
        "prompts": [
            "a person waves with their right hand",
            "a person raises both arms above their head",
            "a person claps their hands together",
            "a person points forward with their right hand",
            "a person stretches their arms",
            "a person puts hands on hips",
            "a person waves with their left hand",
            "a person reaches forward with both hands",
            "a person scratches their head",
            "a person shrugs their shoulders",
            "a person gives a thumbs up",
            "a person beckons with their hand",
            "a person stretches arms to the sides",
            "a person bows",
            "a person nods their head",
            "a person rubs their hands together",
        ],
        "min_success_rate": 0.6,
        "num_frames": 90,
    },
    {
        "level": 3,
        "name": "transitions",
        "description": "Motion transitions and direction changes",
        "prompts": [
            "a person walks forward then turns around",
            "a person walks forward and waves",
            "a person turns to the left",
            "a person turns to the right and walks",
            "a person walks in a small circle",
            "a person walks then stops and looks around",
            "a person jogs slowly then walks",
            "a person walks and picks up something",
            "a person steps forward and reaches out",
            "a person walks while gesturing",
        ],
        "min_success_rate": 0.5,
        "num_frames": 150,  # 5s at 30fps
    },
    {
        "level": 4,
        "name": "dynamic",
        "description": "Dynamic motions with balance challenges",
        "prompts": [
            "a person kicks with their right leg",
            "a person squats down and stands back up",
            "a person steps sideways to the left",
            "a person bends down to pick something up",
            "a person does a lunge",
            "a person balances on one foot",
            "a person kicks with their left leg",
            "a person jumps in place",
            "a person does a jumping jack",
            "a person crouches and stands",
            "a person spins in place",
            "a person does a high kick",
            "a person pivots on one foot",
            "a person does a twisting motion",
            "a person hops on one foot",
        ],
        "min_success_rate": 0.4,
        "num_frames": 120,
    },
]

# Extended prompt pool for Direction B (RL training diversity)
DIVERSE_PROMPTS_POOL = [
    # Daily activities
    "a person picks up an object from the ground",
    "a person sits down on a chair",
    "a person stands up from a seated position",
    "a person opens a door",
    "a person climbs stairs",
    "a person descends stairs",
    "a person sweeps the floor",
    "a person carries a heavy box",
    "a person drinks from a cup",
    "a person types on a keyboard",
    # Sports / exercise
    "a person does push ups",
    "a person does a yoga tree pose",
    "a person stretches their hamstrings",
    "a person shadow boxes",
    "a person throws a ball",
    "a person catches a ball",
    "a person swings a bat",
    "a person dribbles a basketball",
    # Emotional / expressive
    "a person celebrates with a fist pump",
    "a person stomps their feet in anger",
    "a person dances happily",
    "a person slumps with disappointment",
    "a person walks confidently",
    "a person walks sadly with shoulders drooped",
    "a person walks excitedly",
    # Social interactions (single person)
    "a person shakes an imaginary hand",
    "a person pushes something away",
    "a person pulls something toward them",
    "a person salutes",
    "a person bows deeply",
    # Complex locomotion
    "a person jogs in place",
    "a person runs forward",
    "a person skips forward",
    "a person crawls on all fours",
    "a person walks on tiptoes",
    "a person walks while looking behind",
    "a person walks and waves simultaneously",
    "a person shuffles forward",
    "a person tiptoes quietly",
]


class PhysFlowCurriculum:
    """Adaptive curriculum scheduler for PhysFlow training.

    Tracks physics correction success rate and advances to harder levels
    when the model consistently produces physically plausible motions.
    Supports regression on consistent failure.

    Args:
        levels: Custom curriculum levels (default: PHYSFLOW_LEVELS)
        history_size: Number of recent episodes to track for success rate
        min_attempts_to_advance: Minimum attempts at a level before allowing advance
        allow_regression: Whether to drop back to easier levels on failure
        regression_threshold: Success rate below which regression triggers
        seed: Random seed for prompt selection
    """

    def __init__(
        self,
        levels: Optional[List[dict]] = None,
        history_size: int = 50,
        min_attempts_to_advance: int = 15,
        allow_regression: bool = True,
        regression_threshold: float = 0.25,
        seed: Optional[int] = None,
        min_locomotion_ratio: float = 0.0,
    ):
        self.levels = levels or PHYSFLOW_LEVELS
        self.current_level = 0
        self.max_level = len(self.levels) - 1
        self.history_size = history_size
        self.min_attempts_to_advance = min_attempts_to_advance
        self.allow_regression = allow_regression
        self.regression_threshold = regression_threshold
        self.min_locomotion_ratio = min_locomotion_ratio

        self.history = deque(maxlen=history_size)
        self.rng = random.Random(seed)

        # Per-level statistics
        self.level_stats: Dict[int, dict] = {
            i: {"attempts": 0, "successes": 0, "recent": deque(maxlen=history_size)}
            for i in range(len(self.levels))
        }
        self.total_iterations = 0
        self.level_history: List[int] = []

        # Prompt tracking to reduce repetition
        self._recent_prompts: deque = deque(maxlen=8)

    @property
    def current_level_info(self) -> dict:
        """Get info about current curriculum level."""
        return self.levels[self.current_level]

    @property
    def success_rate(self) -> float:
        """Success rate at current level (recent window)."""
        recent = self.level_stats[self.current_level]["recent"]
        if len(recent) == 0:
            return 0.0
        return sum(recent) / len(recent)

    @property
    def global_success_rate(self) -> float:
        """Global success rate across all levels (recent window)."""
        if len(self.history) == 0:
            return 0.0
        return sum(self.history) / len(self.history)

    def get_prompt(self) -> str:
        """Sample a text prompt from current difficulty level.

        With small probability, samples from adjacent levels for diversity.
        Avoids repeating recently used prompts.

        If min_locomotion_ratio > 0, forces a minimum fraction of prompts
        to come from locomotion levels (level >= 1) to prevent catastrophic
        forgetting of walking/movement capabilities.
        """
        # Forced locomotion sampling to prevent catastrophic forgetting
        # When stuck at level 0 (standing), this ensures the model still
        # sees walking/movement prompts at the specified minimum ratio.
        if self.min_locomotion_ratio > 0 and self.rng.random() < self.min_locomotion_ratio:
            # Sample from locomotion levels (level 1+)
            locomotion_levels = [l for l in range(1, len(self.levels))]
            if locomotion_levels:
                level = self.rng.choice(locomotion_levels)
                prompts = self.levels[level]["prompts"]
                available = [p for p in prompts if p not in self._recent_prompts]
                if not available:
                    available = prompts
                prompt = self.rng.choice(available)
                self._recent_prompts.append(prompt)
                return prompt

        # 80% current level, 10% one below, 10% one above
        rand = self.rng.random()
        if rand < 0.80:
            level = self.current_level
        elif rand < 0.90 and self.current_level > 0:
            level = self.current_level - 1
        elif self.current_level < self.max_level:
            level = self.current_level + 1
        else:
            level = self.current_level

        prompts = self.levels[level]["prompts"]

        # Avoid recent repetition
        available = [p for p in prompts if p not in self._recent_prompts]
        if not available:
            available = prompts

        prompt = self.rng.choice(available)
        self._recent_prompts.append(prompt)
        return prompt

    def get_num_frames(self) -> int:
        """Get target number of frames for current level."""
        return self.levels[self.current_level]["num_frames"]

    def get_diverse_prompt(self) -> str:
        """Get a diverse prompt for Direction B (RL training data generation).

        Samples from ALL levels plus the extended diverse pool.
        Biased toward harder motions since RL tracker needs challenge.

        Returns:
            Text prompt for diverse motion generation.
        """
        # 30% from curriculum (all levels), 70% from diverse pool
        if self.rng.random() < 0.30:
            # Sample from curriculum with bias toward higher levels
            weights = [1.0, 2.0, 3.0, 4.0, 5.0][: len(self.levels)]
            level = self.rng.choices(
                range(len(self.levels)), weights=weights, k=1
            )[0]
            prompts = self.levels[level]["prompts"]
        else:
            prompts = DIVERSE_PROMPTS_POOL

        return self.rng.choice(prompts)

    def get_diverse_num_frames(self) -> int:
        """Get frame count for diverse generation (Direction B).

        Returns longer sequences for RL training variety.
        """
        return self.rng.choice([90, 120, 150, 180])

    def update(self, success: bool):
        """Update curriculum based on physics correction success.

        Args:
            success: Whether physics correction passed quality gate
        """
        self.history.append(success)
        self.total_iterations += 1
        self.level_history.append(self.current_level)

        # Update per-level stats
        stats = self.level_stats[self.current_level]
        stats["attempts"] += 1
        if success:
            stats["successes"] += 1
        stats["recent"].append(1 if success else 0)

        # Check level transitions
        self._check_level_transition()

    def _check_level_transition(self):
        """Check if we should advance or regress curriculum level."""
        stats = self.level_stats[self.current_level]
        recent = stats["recent"]

        # Need minimum attempts before transition
        if len(recent) < self.min_attempts_to_advance:
            return

        current_rate = sum(recent) / len(recent)
        level_info = self.levels[self.current_level]

        # Check advancement
        if (
            self.current_level < self.max_level
            and current_rate >= level_info["min_success_rate"]
        ):
            old_level = self.current_level
            self.current_level += 1
            log.info(
                f"[Curriculum] ADVANCED: level {old_level} "
                f"({self.levels[old_level]['name']}) -> "
                f"level {self.current_level} "
                f"({self.levels[self.current_level]['name']}) "
                f"[rate={current_rate:.2f} >= {level_info['min_success_rate']:.2f}]"
            )
            return

        # Check regression
        if (
            self.allow_regression
            and self.current_level > 0
            and current_rate < self.regression_threshold
        ):
            old_level = self.current_level
            self.current_level -= 1
            # Clear history at new (easier) level to give fresh start
            self.level_stats[self.current_level]["recent"].clear()
            log.info(
                f"[Curriculum] REGRESSED: level {old_level} "
                f"({self.levels[old_level]['name']}) -> "
                f"level {self.current_level} "
                f"({self.levels[self.current_level]['name']}) "
                f"[rate={current_rate:.2f} < {self.regression_threshold:.2f}]"
            )

    def get_state(self) -> dict:
        """Get serializable state for logging/checkpointing."""
        return {
            "current_level": self.current_level,
            "level_name": self.levels[self.current_level]["name"],
            "success_rate": self.success_rate,
            "global_success_rate": self.global_success_rate,
            "total_iterations": self.total_iterations,
            "level_stats": {
                i: {
                    "name": self.levels[i]["name"],
                    "attempts": stats["attempts"],
                    "successes": stats["successes"],
                    "recent_rate": (
                        sum(stats["recent"]) / len(stats["recent"])
                        if len(stats["recent"]) > 0
                        else 0.0
                    ),
                }
                for i, stats in self.level_stats.items()
            },
        }

    def state_dict(self) -> dict:
        """Serialize full curriculum state for checkpointing."""
        return {
            "current_level": self.current_level,
            "total_iterations": self.total_iterations,
            "level_history": list(self.level_history[-200:]),  # Keep last 200
            "history": list(self.history),
            "level_stats": {
                i: {
                    "attempts": stats["attempts"],
                    "successes": stats["successes"],
                    "recent": list(stats["recent"]),
                }
                for i, stats in self.level_stats.items()
            },
        }

    def load_state_dict(self, state: dict):
        """Restore curriculum state from checkpoint."""
        self.current_level = state["current_level"]
        self.total_iterations = state["total_iterations"]
        self.level_history = state.get("level_history", [])
        self.history = deque(state.get("history", []), maxlen=self.history_size)

        for level_key, level_state in state.get("level_stats", {}).items():
            level_int = int(level_key)
            if level_int in self.level_stats:
                self.level_stats[level_int]["attempts"] = level_state["attempts"]
                self.level_stats[level_int]["successes"] = level_state["successes"]
                self.level_stats[level_int]["recent"] = deque(
                    level_state["recent"], maxlen=self.history_size
                )

        log.info(
            f"[Curriculum] Restored: level={self.current_level} "
            f"({self.levels[self.current_level]['name']}), "
            f"iterations={self.total_iterations}"
        )

    def reset(self, level: Optional[int] = None):
        """Reset curriculum state.

        Args:
            level: If provided, reset to this level. Otherwise reset to 0.
        """
        self.current_level = level if level is not None else 0
        self.history.clear()
        self.total_iterations = 0
        self.level_history.clear()
        self._recent_prompts.clear()
        for stats in self.level_stats.values():
            stats["attempts"] = 0
            stats["successes"] = 0
            stats["recent"].clear()
        log.info(f"[Curriculum] Reset to level {self.current_level}")

    def __repr__(self):
        return (
            f"PhysFlowCurriculum(level={self.current_level}/{self.max_level}, "
            f"name='{self.levels[self.current_level]['name']}', "
            f"success_rate={self.success_rate:.2f})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Standalone test
# ═══════════════════════════════════════════════════════════════════════════════


def _test_curriculum():
    """Test curriculum progression logic."""
    logging.basicConfig(level=logging.INFO)

    curriculum = PhysFlowCurriculum(
        history_size=30, min_attempts_to_advance=10, seed=42
    )

    print("=" * 60)
    print("Test 1: Advance through levels with high success rate")
    print("=" * 60)

    for iteration in range(120):
        prompt = curriculum.get_prompt()
        # Simulate high success for lower levels
        if curriculum.current_level <= 1:
            success = random.random() < 0.9
        elif curriculum.current_level == 2:
            success = random.random() < 0.7
        else:
            success = random.random() < 0.55
        curriculum.update(success=success)

        if iteration % 20 == 0:
            state = curriculum.get_state()
            print(
                f"  iter={iteration:3d}: level={state['current_level']} "
                f"({state['level_name']}), "
                f"rate={state['success_rate']:.2f}, "
                f"prompt='{prompt[:40]}...'"
            )

    print(f"\nFinal: {curriculum}")
    print(f"State: {curriculum.get_state()}")

    print("\n" + "=" * 60)
    print("Test 2: Regression on consistent failure")
    print("=" * 60)

    curriculum.reset(level=3)
    for iteration in range(40):
        prompt = curriculum.get_prompt()
        success = random.random() < 0.15  # Very low success
        curriculum.update(success=success)

        if iteration % 8 == 0:
            print(
                f"  iter={iteration:3d}: level={curriculum.current_level} "
                f"({curriculum.current_level_info['name']}), "
                f"rate={curriculum.success_rate:.2f}"
            )

    print(f"\nFinal level after regression: {curriculum.current_level}")

    print("\n" + "=" * 60)
    print("Test 3: Diverse prompts for Direction B")
    print("=" * 60)

    diverse = [curriculum.get_diverse_prompt() for _ in range(10)]
    for p in diverse:
        print(f"  - {p}")

    print("\n" + "=" * 60)
    print("Test 4: State dict serialization round-trip")
    print("=" * 60)

    state = curriculum.state_dict()
    new_curriculum = PhysFlowCurriculum(seed=42)
    new_curriculum.load_state_dict(state)
    assert new_curriculum.current_level == curriculum.current_level
    assert new_curriculum.total_iterations == curriculum.total_iterations
    print(f"  Original:  {curriculum}")
    print(f"  Restored:  {new_curriculum}")
    print("  State dict round-trip: OK")

    print("\nAll tests passed!")


if __name__ == "__main__":
    _test_curriculum()
