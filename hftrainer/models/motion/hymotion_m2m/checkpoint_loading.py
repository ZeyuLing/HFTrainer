"""
Selective T2M-to-M2M v2 Checkpoint Loading.

This module implements selective checkpoint loading from HyMotion-T2M pretrained
weights into HyMotion-M2M v2 bundle, handling architecture differences:

- **Reusable modules** (exact shape match):
  - Text encoders (ctxt_encoder, vtxt_encoder, text_refiner)
  - Timestep encoder
  - Transformer blocks (double_blocks, single_blocks)

- **Non-reusable modules** (shape mismatch, reinitialized):
  - input_encoder: 135→594 (VACE expands input)
  - final_layer: 135→198 (M2M output dimension differs)

- **Bundle-level parameters** (NOT loaded from T2M):
  - null_vtxt_feat, null_ctxt_input: M2M keeps trainable randn*0.01 (T2M is frozen zeros)
  - mean, std: M2M uses 198-dim stats (different from T2M 135-dim)

**Usage**:
  from hftrainer.models.motion.hymotion_m2m.checkpoint_loading import load_t2m_pretrained_selective
  
  stats = load_t2m_pretrained_selective(
      bundle=m2m_bundle,
      t2m_checkpoint_path='checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt',
      freeze_strategy='encoders'  # 'none', 'encoders', 'text_refiner', 'full'
  )
  print(f"Loaded {stats['modules_loaded']} modules, "
        f"skipped {stats['modules_skipped']}, "
        f"reinitialized {stats['modules_reinitialized']}")
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ============================================================================
# Module Path Definitions
# ============================================================================

# Modules that exist in both T2M and M2M with identical structure/shape
REUSABLE_MODULES = {
    'motion_transformer.ctxt_encoder',
    'motion_transformer.vtxt_encoder',
    'motion_transformer.timestep_encoder',
    'motion_transformer.text_refiner',
    'motion_transformer.double_blocks',
    'motion_transformer.single_blocks',
}

# Modules with shape mismatches (skipped, reinitialized)
SHAPE_MISMATCH_MODULES = {
    'motion_transformer.input_encoder',    # 135→594 input dimension
    'motion_transformer.final_layer',       # 135→198 output dimension
}

# Bundle-level parameters to exclude (use config-initialized values)
EXCLUDED_BUNDLE_PARAMS = {
    'null_vtxt_feat',   # M2M needs trainable zeros, not T2M frozen values
    'null_ctxt_input',  # M2M needs trainable zeros, not T2M frozen values
    'mean',              # M2M uses 198-dim stats (T2M is 135-dim)
    'std',               # M2M uses 198-dim stats (T2M is 135-dim)
}


# ============================================================================
# Loading Utilities
# ============================================================================

def _load_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """
    Load checkpoint from .ckpt or .pt file.
    
    Returns:
        Checkpoint dictionary with keys like 'model', 'optimizer', etc.
        or direct state_dict if .pt format.
    """
    if checkpoint_path.endswith('.ckpt'):
        # PyTorch Lightning checkpoint format
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            # .ckpt with model/optimizer/trainer_state
            state_dict = checkpoint['model']
        elif isinstance(checkpoint, dict) and any(k.startswith('motion_transformer') or k.startswith('text_encoder') for k in checkpoint.keys()):
            # .ckpt is already a state_dict
            state_dict = checkpoint
        else:
            state_dict = checkpoint
    elif checkpoint_path.endswith('.pt'):
        # Direct state_dict
        state_dict = torch.load(checkpoint_path, map_location='cpu')
    else:
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}. Expected .ckpt or .pt")
    
    return state_dict


def _filter_reusable_params(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Extract only parameters from reusable modules.
    
    Args:
        state_dict: Checkpoint state_dict (possibly nested)
    
    Returns:
        Filtered state_dict containing only reusable module parameters
    """
    filtered = {}
    
    for key, value in state_dict.items():
        # Check if key starts with any reusable module path
        for reusable_mod in REUSABLE_MODULES:
            if key.startswith(reusable_mod):
                filtered[key] = value
                break
    
    return filtered


def _get_shape_mismatches(state_dict: Dict[str, torch.Tensor], bundle) -> Dict[str, tuple]:
    """
    Identify parameters with shape mismatches between checkpoint and model.
    
    Returns:
        Dict mapping parameter name to (ckpt_shape, model_shape)
    """
    mismatches = {}
    model_state = bundle.motion_transformer.state_dict()
    
    for key, ckpt_value in state_dict.items():
        if key in model_state:
            model_value = model_state[key]
            if ckpt_value.shape != model_value.shape:
                mismatches[key] = (tuple(ckpt_value.shape), tuple(model_value.shape))
    
    return mismatches


def _count_parameters(module: nn.Module) -> int:
    """Count total parameters in a module."""
    return sum(p.numel() for p in module.parameters())


def _freeze_module(module: nn.Module) -> None:
    """Freeze all parameters in a module."""
    module.requires_grad_(False)


def _unfreeze_module(module: nn.Module) -> None:
    """Unfreeze all parameters in a module."""
    module.requires_grad_(True)


# ============================================================================
# Main Loading Function
# ============================================================================

def load_t2m_pretrained_selective(
    bundle,
    t2m_checkpoint_path: str,
    freeze_strategy: str = 'none',
) -> Dict[str, Any]:
    """
    Selectively load T2M pretrained weights into M2M v2 bundle.
    
    Handles architecture differences between T2M (input_dim=135) and M2M v2
    (input_dim=594 with VACE conditioning). Loads all reusable modules
    (encoders, blocks) while reinitializing shape-mismatched layers
    (input_encoder, final_layer).
    
    Args:
        bundle: HyMotionM2MBundle instance
        t2m_checkpoint_path: Path to T2M pretrained checkpoint (.ckpt or .pt)
        freeze_strategy: Which modules to freeze after loading:
            - 'none': Don't freeze anything (default)
            - 'encoders': Freeze text encoders + timestep_encoder only
            - 'text_refiner': Also freeze text_refiner
            - 'blocks': Also freeze all transformer blocks (double + single)
            - 'full': Freeze all reusable modules (encoders + blocks + text_refiner)
    
    Returns:
        Dict with statistics:
            - 'modules_loaded': List of modules successfully loaded
            - 'modules_skipped': List of modules skipped due to shape mismatch
            - 'modules_reinitialized': List of modules with random reinitialization
            - 'num_params_loaded': Total parameters loaded
            - 'num_params_skipped': Total parameters skipped (shape mismatch)
            - 'num_params_reinitialized': Total parameters reinitialized
            - 'frozen_modules': List of modules frozen per freeze_strategy
    
    Raises:
        ValueError: If checkpoint_path doesn't exist or freeze_strategy is invalid
        RuntimeError: If checkpoint loading fails
    """
    import os
    
    if not os.path.exists(t2m_checkpoint_path):
        raise ValueError(f"Checkpoint not found: {t2m_checkpoint_path}")
    
    freeze_strategies = {'none', 'encoders', 'text_refiner', 'blocks', 'full'}
    if freeze_strategy not in freeze_strategies:
        raise ValueError(
            f"freeze_strategy must be one of {freeze_strategies}, got {freeze_strategy!r}"
        )
    
    logger.info(f"Loading T2M pretrained checkpoint: {t2m_checkpoint_path}")
    logger.info(f"Freeze strategy: {freeze_strategy}")
    
    # Load checkpoint
    try:
        t2m_state = _load_checkpoint(t2m_checkpoint_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {e}") from e
    
    # Extract only reusable module parameters
    reusable_state = _filter_reusable_params(t2m_state)
    
    if not reusable_state:
        logger.warning("No reusable parameters found in checkpoint. "
                      "This may indicate a format or path issue.")
        return {
            'modules_loaded': [],
            'modules_skipped': [],
            'modules_reinitialized': [],
            'num_params_loaded': 0,
            'num_params_skipped': 0,
            'num_params_reinitialized': 0,
            'frozen_modules': [],
        }
    
    # Detect shape mismatches
    mismatches = _get_shape_mismatches(reusable_state, bundle)
    
    # Load reusable parameters (strict=False to skip mismatches)
    bundle.load_state_dict_selective(
        {'motion_transformer': reusable_state},
        strict=False,
        exclude_bundle_keys=list(EXCLUDED_BUNDLE_PARAMS),
    )
    
    # Track statistics
    stats = {
        'modules_loaded': [],
        'modules_skipped': [],
        'modules_reinitialized': [],
        'num_params_loaded': 0,
        'num_params_skipped': 0,
        'num_params_reinitialized': 0,
        'frozen_modules': [],
    }
    
    # Count loaded parameters
    for key, value in reusable_state.items():
        if key not in mismatches:
            stats['num_params_loaded'] += value.numel()
            # Extract module name (e.g., "motion_transformer.double_blocks")
            parts = key.split('.')
            if len(parts) >= 2:
                mod_name = f"{parts[0]}.{parts[1]}"
                if mod_name not in stats['modules_loaded']:
                    stats['modules_loaded'].append(mod_name)
    
    # Count skipped/mismatched parameters
    for key, value in reusable_state.items():
        if key in mismatches:
            stats['num_params_skipped'] += value.numel()
            parts = key.split('.')
            if len(parts) >= 2:
                mod_name = f"{parts[0]}.{parts[1]}"
                if mod_name not in stats['modules_skipped']:
                    stats['modules_skipped'].append(mod_name)
    
    # Reinitialize shape-mismatch modules
    for mod_path in SHAPE_MISMATCH_MODULES:
        try:
            parts = mod_path.split('.')
            if len(parts) == 2:
                parent = getattr(bundle, parts[0], None)
                if parent and hasattr(parent, parts[1]):
                    mod = getattr(parent, parts[1])
                    _reinitialize_module(mod)
                    stats['modules_reinitialized'].append(mod_path)
                    stats['num_params_reinitialized'] += _count_parameters(mod)
        except Exception as e:
            logger.warning(f"Failed to reinitialize {mod_path}: {e}")
    
    # Apply freezing strategy
    frozen_modules = _apply_freeze_strategy(bundle, freeze_strategy)
    stats['frozen_modules'] = frozen_modules
    
    # Log summary
    logger.info(
        f"Loaded {len(stats['modules_loaded'])} module types "
        f"({stats['num_params_loaded']:,} params), "
        f"skipped {len(stats['modules_skipped'])} module types "
        f"({stats['num_params_skipped']:,} params), "
        f"reinitialized {len(stats['modules_reinitialized'])} module types "
        f"({stats['num_params_reinitialized']:,} params)"
    )
    
    if stats['modules_skipped']:
        logger.info(f"Skipped (shape mismatch): {', '.join(stats['modules_skipped'])}")
    
    if stats['modules_reinitialized']:
        logger.info(f"Reinitialized: {', '.join(stats['modules_reinitialized'])}")
    
    if stats['frozen_modules']:
        logger.info(f"Frozen (strategy={freeze_strategy}): {', '.join(stats['frozen_modules'])}")
    
    return stats


# ============================================================================
# Helper Functions
# ============================================================================

def _reinitialize_module(module: nn.Module) -> None:
    """
    Reinitialize all weights in a module using Xavier uniform initialization.
    
    This is used for layers that don't have direct equivalents in T2M
    (e.g., input_encoder for expanded VACE input, final_layer for different output dim).
    """
    for param in module.parameters():
        if param.dim() >= 2:
            nn.init.xavier_uniform_(param)
        else:
            # Bias or 1D params: zero initialization
            nn.init.zeros_(param)


def _apply_freeze_strategy(bundle, freeze_strategy: str) -> list:
    """
    Apply freezing strategy to specify which modules are frozen after loading.
    
    Args:
        bundle: HyMotionM2MBundle instance
        freeze_strategy: 'none', 'encoders', 'text_refiner', 'blocks', 'full'
    
    Returns:
        List of frozen module names
    """
    frozen = []
    
    if freeze_strategy == 'none':
        return frozen
    
    # Build list of modules to freeze based on strategy
    modules_to_freeze = []
    
    if freeze_strategy in ('encoders', 'text_refiner', 'blocks', 'full'):
        # Freeze text encoders and timestep encoder
        modules_to_freeze.extend([
            'motion_transformer.ctxt_encoder',
            'motion_transformer.vtxt_encoder',
            'motion_transformer.timestep_encoder',
        ])
    
    if freeze_strategy in ('text_refiner', 'blocks', 'full'):
        # Also freeze text refiner
        modules_to_freeze.append('motion_transformer.text_refiner')
    
    if freeze_strategy in ('blocks', 'full'):
        # Also freeze transformer blocks
        modules_to_freeze.extend([
            'motion_transformer.double_blocks',
            'motion_transformer.single_blocks',
        ])
    
    # Apply freezing
    for mod_path in modules_to_freeze:
        try:
            parts = mod_path.split('.')
            if len(parts) == 2:
                parent = getattr(bundle, parts[0], None)
                if parent and hasattr(parent, parts[1]):
                    mod = getattr(parent, parts[1])
                    _freeze_module(mod)
                    frozen.append(mod_path)
        except Exception as e:
            logger.warning(f"Failed to freeze {mod_path}: {e}")
    
    return frozen


def verify_loading(bundle, t2m_checkpoint_path: str) -> Dict[str, Any]:
    """
    Verify that T2M pretrained parameters were correctly loaded into M2M bundle.
    
    Compares weights before/after loading by reloading from checkpoint and
    comparing specific parameters.
    
    Args:
        bundle: HyMotionM2MBundle instance (after loading)
        t2m_checkpoint_path: Path to T2M checkpoint for verification
    
    Returns:
        Dict with verification results:
            - 'reusable_params_match': bool, whether loaded params match checkpoint
            - 'num_verified_params': int, how many parameters were checked
            - 'mismatches': list of parameter names with mismatches
    """
    t2m_state = _load_checkpoint(t2m_checkpoint_path)
    reusable_state = _filter_reusable_params(t2m_state)
    
    model_state = bundle.motion_transformer.state_dict()
    
    mismatches = []
    verified_count = 0
    
    for key, ckpt_value in reusable_state.items():
        if key in model_state:
            model_value = model_state[key]
            if ckpt_value.shape == model_value.shape:
                # Compare values (allow small numerical differences)
                if not torch.allclose(ckpt_value, model_value, atol=1e-4):
                    mismatches.append(key)
                verified_count += 1
    
    return {
        'reusable_params_match': len(mismatches) == 0,
        'num_verified_params': verified_count,
        'mismatches': mismatches,
    }
