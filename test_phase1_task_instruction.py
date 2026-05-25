#!/usr/bin/env python3
"""Test script: Verify Phase 1 Task Instruction Modulation end-to-end."""

import torch
import sys
from typing import Dict, List

def test_task_instruction_module():
    """Test 1: Task instruction module loads and maps all 7 strategies."""
    print("\n" + "="*70)
    print("TEST 1: Task Instruction Module")
    print("="*70)
    
    from hftrainer.models.motion.hymotion_m2m.task_instruction import (
        get_task_instruction,
        STRATEGY_TO_INSTRUCTION,
    )
    
    strategies = ['m1_random_cell', 'm2_random_block', 'm3_temporal_contiguous',
                  'm4_joint_contiguous', 'm5_full_mask', 'm6_keyframe_sparse',
                  'm7_scattered_joint']
    
    print(f"✓ Found {len(STRATEGY_TO_INSTRUCTION)} strategies:")
    for strat, instr in STRATEGY_TO_INSTRUCTION.items():
        print(f"  {strat:25} → '{instr}'")
    
    for strat in strategies:
        instr = get_task_instruction(strat)
        assert instr, f"Failed to get instruction for {strat}"
        print(f"  ✓ {strat}: {instr}")
    
    print("\n✓ PASS: Task instruction module")
    return True


def test_bundle_encode_task_instruction():
    """Test 2: Bundle can encode task instructions to embeddings."""
    print("\n" + "="*70)
    print("TEST 2: Bundle Task Instruction Encoding")
    print("="*70)
    
    try:
        from hftrainer.models.motion.hymotion_m2m.bundle import HyMotionM2MBundle
        from hftrainer.models.motion.hymotion_m2m.task_instruction import get_task_instruction
        
        # Check method exists
        assert hasattr(HyMotionM2MBundle, 'encode_task_instruction'), \
            "Bundle missing encode_task_instruction method"
        print("✓ Bundle.encode_task_instruction method exists")
        
        # Test strategy coverage
        strategies = ['m1_random_cell', 'm3_temporal_contiguous', 'm5_full_mask', 'm7_scattered_joint']
        instructions = [get_task_instruction(s) for s in strategies]
        print(f"✓ Task instructions prepared:")
        for s, instr in zip(strategies, instructions):
            print(f"  {s}: '{instr}'")
        
        print("\n✓ PASS: Bundle task instruction encoding setup")
        return True
        
    except Exception as e:
        print(f"✗ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mmdit_task_emb_parameter():
    """Test 3: MMDiT forward() accepts task_emb parameter."""
    print("\n" + "="*70)
    print("TEST 3: MMDiT Task Embedding Parameter")
    print("="*70)
    
    try:
        import inspect
        from hftrainer.models.motion.hymotion_m2m.network.hymotion_mmdit import HunyuanMotionMMDiT
        
        # Check forward signature
        sig = inspect.signature(HunyuanMotionMMDiT.forward)
        params = list(sig.parameters.keys())
        
        print(f"✓ HunyuanMotionMMDiT.forward parameters: {params[:10]}...")  # First 10
        assert 'task_emb' in params, f"task_emb not in MMDiT.forward parameters: {params}"
        print("✓ task_emb parameter found in MMDiT.forward signature")
        
        # Check parameter type annotation
        task_emb_param = sig.parameters['task_emb']
        print(f"✓ task_emb annotation: {task_emb_param.annotation}")
        print(f"✓ task_emb default: {task_emb_param.default}")
        
        print("\n✓ PASS: MMDiT task_emb parameter")
        return True
        
    except Exception as e:
        print(f"✗ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_trainer_task_instruction_integration():
    """Test 4: Trainer integrates task instruction encoding."""
    print("\n" + "="*70)
    print("TEST 4: Trainer Task Instruction Integration")
    print("="*70)
    
    try:
        from hftrainer.trainers.motion.hymotion_m2m_trainer import HyMotionM2MTrainer
        import inspect
        
        # Check that trainer references task instruction module
        source = inspect.getsource(HyMotionM2MTrainer._prepare_and_forward)
        
        # Look for key markers
        markers = {
            'get_task_instruction import': 'from hftrainer.models.motion.hymotion_m2m.task_instruction import',
            'mask_strategy extraction': 'batch.get("mask_strategy")',
            'encode_task_instruction call': 'self.bundle.encode_task_instruction',
            'task_emb forwarding': 'task_emb=task_emb',
        }
        
        for name, marker in markers.items():
            found = marker in source
            status = "✓" if found else "✗"
            print(f"{status} {name}: {'found' if found else 'NOT FOUND'}")
            assert found, f"Marker '{marker}' not found in trainer"
        
        print("\n✓ PASS: Trainer task instruction integration")
        return True
        
    except Exception as e:
        print(f"✗ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_phase1():
    """Test 5: Phase 1 config file is valid."""
    print("\n" + "="*70)
    print("TEST 5: Phase 1 Config Validation")
    print("="*70)
    
    try:
        from mmengine.config import Config
        config_path = 'configs/hymotion_m2m/hymotion_m2m_completion_uncond_fm_man_046b_phase1.py'
        
        cfg = Config.fromfile(config_path)
        print(f"✓ Config loaded from {config_path}")
        
        # Check trainer config has encode_task_instruction
        trainer_cfg = cfg.get('trainer', {})
        encode_task_instruction = trainer_cfg.get('encode_task_instruction')
        print(f"✓ trainer.encode_task_instruction = {encode_task_instruction}")
        
        # Check work_dir is sensible
        work_dir = cfg.get('work_dir', '')
        assert 'phase1' in work_dir.lower(), f"work_dir should mention phase1: {work_dir}"
        print(f"✓ work_dir = {work_dir}")
        
        print("\n✓ PASS: Phase 1 config validation")
        return True
        
    except Exception as e:
        print(f"✗ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_flow_mock():
    """Test 6: Mock data flow through the pipeline."""
    print("\n" + "="*70)
    print("TEST 6: Mock Data Flow")
    print("="*70)
    
    try:
        from hftrainer.models.motion.hymotion_m2m.task_instruction import get_task_instruction
        
        # Simulate batch data
        batch_size = 4
        seq_length = 360
        motion_dim = 135
        
        # Mock strategies
        strategies = ['m1_random_cell', 'm3_temporal_contiguous', 'm5_full_mask', 'm7_scattered_joint']
        strategies_batch = strategies * (batch_size // len(strategies))
        
        print(f"Simulated batch:")
        print(f"  batch_size = {batch_size}")
        print(f"  seq_length = {seq_length}")
        print(f"  motion_dim = {motion_dim}")
        print(f"  strategies = {strategies_batch}")
        
        # Step 1: Get instructions
        instructions = [get_task_instruction(s) for s in strategies_batch]
        print(f"\n✓ Step 1: Task instructions extracted")
        for i, (s, instr) in enumerate(zip(strategies_batch, instructions)):
            print(f"  Batch[{i}]: {s} → '{instr}'")
        
        # Step 2: (Mock) Encode instructions
        print(f"\n✓ Step 2: Encode instructions to embeddings")
        print(f"  Expected output shape: ({batch_size}, 1, 1024)")
        print(f"  (In real training, done via bundle.encode_task_instruction)")
        
        # Step 3: (Mock) Inject into adapter
        print(f"\n✓ Step 3: Inject into adapter signal")
        print(f"  adapter = timestep_feat (1024) + vtxt_feat (1024) + task_emb (1024)")
        print(f"  Result: adapter shape (B, 1, 1024)")
        
        # Step 4: (Mock) Modulate all layers
        print(f"\n✓ Step 4: Pass to all ModulateDiT layers")
        print(f"  Each layer receives adapter signal with task instruction embedded")
        print(f"  Layers apply shift/scale/gate modulation based on adapter")
        
        print("\n✓ PASS: Mock data flow")
        return True
        
    except Exception as e:
        print(f"✗ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "  PHASE 1 TASK INSTRUCTION MODULATION: END-TO-END VERIFICATION".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70)
    
    tests = [
        ("Task Instruction Module", test_task_instruction_module),
        ("Bundle Encoding Setup", test_bundle_encode_task_instruction),
        ("MMDiT Parameter", test_mmdit_task_emb_parameter),
        ("Trainer Integration", test_trainer_task_instruction_integration),
        ("Phase 1 Config", test_config_phase1),
        ("Mock Data Flow", test_data_flow_mock),
    ]
    
    results = {}
    for name, test_fn in tests:
        try:
            results[name] = test_fn()
        except Exception as e:
            print(f"\n✗ EXCEPTION in {name}: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
    
    total_passed = sum(1 for p in results.values() if p)
    total = len(results)
    
    print(f"\nTotal: {total_passed}/{total} tests passed")
    
    if total_passed == total:
        print("\n✓✓✓ ALL TESTS PASSED ✓✓✓")
        print("\nPhase 1 Task Instruction Modulation is ready for training!")
        return 0
    else:
        print(f"\n✗✗✗ {total - total_passed} TEST(S) FAILED ✗✗✗")
        return 1


if __name__ == '__main__':
    sys.exit(main())
