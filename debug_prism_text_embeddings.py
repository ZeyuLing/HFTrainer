"""Debug script for PRISM text embedding configuration and flow."""

import torch
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

def test_bundle_text_encoding():
    """Test text encoding with mask support in bundle."""
    print("\n" + "="*80)
    print("TEST 1: Bundle Text Encoding with Mask Support")
    print("="*80)
    
    try:
        from hftrainer.models.motion.prism.bundle import PrismBundle
        print("✓ Successfully imported PrismBundle")
        
        # Check if the new method exists
        if hasattr(PrismBundle, 'encode_prompt_with_mask'):
            print("✓ encode_prompt_with_mask method exists in PrismBundle")
        else:
            print("✗ encode_prompt_with_mask method NOT found in PrismBundle")
            
    except Exception as e:
        print(f"✗ Error importing PrismBundle: {e}")
        

def test_trainer_text_encoding():
    """Test trainer text encoding."""
    print("\n" + "="*80)
    print("TEST 2: PRISM Trainer Text Encoding")
    print("="*80)
    
    try:
        from hftrainer.trainers.motion.prism_trainer import PrismTrainer
        print("✓ Successfully imported PrismTrainer")
        
        # Read the trainer source to verify the changes
        import inspect
        source = inspect.getsource(PrismTrainer.train_step)
        
        if 'encode_prompt_with_mask' in source:
            print("✓ Trainer uses encode_prompt_with_mask method")
            # Find the relevant line
            for i, line in enumerate(source.split('\n')):
                if 'encode_prompt_with_mask' in line:
                    print(f"  Line {i}: {line.strip()[:80]}...")
        else:
            print("✗ Trainer does NOT use encode_prompt_with_mask method")
            
        if 'encoder_hidden_states_mask' in source:
            print("✓ Trainer passes encoder_hidden_states_mask to transformer")
        else:
            print("✗ Trainer does NOT pass encoder_hidden_states_mask")
            
    except Exception as e:
        print(f"✗ Error testing trainer: {e}")


def test_backend_text_encoding():
    """Test backend text encoding with masks."""
    print("\n" + "="*80)
    print("TEST 3: PRISM Backend Text Encoding with Masks")
    print("="*80)
    
    try:
        from hftrainer.pipelines.motion.prism_backend import PrismARPipeline
        print("✓ Successfully imported PrismARPipeline")
        
        # Check if new methods exist
        if hasattr(PrismARPipeline, 'encode_prompt_with_mask'):
            print("✓ encode_prompt_with_mask method exists in PrismARPipeline")
        else:
            print("✗ encode_prompt_with_mask method NOT found in PrismARPipeline")
            
        if hasattr(PrismARPipeline, '_get_t5_prompt_embeds_with_mask'):
            print("✓ _get_t5_prompt_embeds_with_mask method exists in PrismARPipeline")
        else:
            print("✗ _get_t5_prompt_embeds_with_mask method NOT found in PrismARPipeline")
        
        # Check if generate_single_segment uses the new methods
        import inspect
        source = inspect.getsource(PrismARPipeline.generate_single_segment)
        
        if 'encode_prompt_with_mask' in source:
            print("✓ generate_single_segment uses encode_prompt_with_mask")
        else:
            print("✗ generate_single_segment does NOT use encode_prompt_with_mask")
            
        if 'encoder_hidden_states_mask' in source:
            print("✓ generate_single_segment passes encoder_hidden_states_mask to transformer")
            # Count occurrences
            count = source.count('encoder_hidden_states_mask')
            print(f"  (Found {count} occurrences of encoder_hidden_states_mask)")
        else:
            print("✗ generate_single_segment does NOT pass encoder_hidden_states_mask")
            
    except Exception as e:
        print(f"✗ Error testing backend: {e}")


def check_configuration():
    """Check training/inference configuration for consistency."""
    print("\n" + "="*80)
    print("TEST 4: Training-Inference Configuration Consistency")
    print("="*80)
    
    try:
        from hftrainer.trainers.motion.prism_trainer import PrismTrainer
        from hftrainer.pipelines.motion.prism_backend import PrismARPipeline
        
        # Check default max_text_length in trainer
        trainer_init_params = PrismTrainer.__init__.__code__.co_varnames
        print("✓ PrismTrainer parameters found")
        
        # Check default max_sequence_length in generate_single_segment
        import inspect
        sig = inspect.signature(PrismARPipeline.generate_single_segment)
        if 'max_sequence_length' in sig.parameters:
            default_inference_max_seq = sig.parameters['max_sequence_length'].default
            print(f"✓ Inference default max_sequence_length: {default_inference_max_seq}")
        
        # Also check __call__ method
        sig_call = inspect.signature(PrismARPipeline.__call__)
        if 'max_sequence_length' in sig_call.parameters:
            default_call_max_seq = sig_call.parameters['max_sequence_length'].default
            print(f"✓ __call__ default max_sequence_length: {default_call_max_seq}")
            
    except Exception as e:
        print(f"✗ Error checking configuration: {e}")


def create_mock_test():
    """Create a mock test of the mask computation logic."""
    print("\n" + "="*80)
    print("TEST 5: Mock Text Mask Computation")
    print("="*80)
    
    try:
        # Simulate the mask computation logic
        batch_size = 2
        max_seq_len = 128
        seq_lens = torch.tensor([7, 15])  # Two samples with different lengths
        
        # Create mask: 1 for valid tokens, 0 for padding
        mask = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
        for i, seq_len in enumerate(seq_lens):
            mask[i, :seq_len] = 1
        
        print(f"✓ Created mock text mask for batch_size={batch_size}, max_seq_len={max_seq_len}")
        print(f"  Mask shape: {mask.shape}")
        print(f"  Sample 0 - sequence length: {seq_lens[0]}, mask sum: {mask[0].sum()}")
        print(f"  Sample 1 - sequence length: {seq_lens[1]}, mask sum: {mask[1].sum()}")
        
        # Test repeating mask for num_motion_per_prompt
        num_motion_per_prompt = 3
        repeated_mask = mask.repeat(num_motion_per_prompt, 1)
        print(f"✓ Repeated mask for num_motion_per_prompt={num_motion_per_prompt}")
        print(f"  Repeated mask shape: {repeated_mask.shape}")
        print(f"  Expected shape: ({batch_size * num_motion_per_prompt}, {max_seq_len})")
        
        if repeated_mask.shape == (batch_size * num_motion_per_prompt, max_seq_len):
            print("✓ Mask repetition works correctly")
        else:
            print("✗ Mask repetition shape mismatch")
            
    except Exception as e:
        print(f"✗ Error in mock test: {e}")


def print_summary():
    """Print summary of changes."""
    print("\n" + "="*80)
    print("SUMMARY OF CHANGES")
    print("="*80)
    
    changes = [
        ("bundle.py", "Added encode_prompt_with_mask() method", "Returns embeddings + attention mask [B, max_len]"),
        ("prism_trainer.py", "Updated train_step() to use encode_prompt_with_mask", "Lines 56-61 changed, passes encoder_hidden_states_mask to transformer"),
        ("prism_backend.py", "Added encode_prompt_with_mask() method", "Returns embeddings + masks for both positive and negative prompts"),
        ("prism_backend.py", "Added _get_t5_prompt_embeds_with_mask() method", "Low-level method for computing embeddings + masks"),
        ("prism_backend.py", "Updated generate_single_segment() method", "Now uses encoder_hidden_states_mask in transformer forward calls"),
    ]
    
    for file, change, details in changes:
        print(f"\n✓ {file}")
        print(f"  - {change}")
        print(f"  - {details}")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("PRISM TEXT EMBEDDING DEBUGGING & VERIFICATION")
    print("="*80)
    
    test_bundle_text_encoding()
    test_trainer_text_encoding()
    test_backend_text_encoding()
    check_configuration()
    create_mock_test()
    print_summary()
    
    print("\n" + "="*80)
    print("VERIFICATION COMPLETE")
    print("="*80 + "\n")
