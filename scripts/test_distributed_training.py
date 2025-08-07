#!/usr/bin/env python
import pytest; pytest.skip("Helper script – skip during pytest collection", allow_module_level=True)
"""Test script for distributed training with DDP and FSDP.

This file is intended to be executed manually; it should **not** be collected
as part of the automated `pytest` suite.  We therefore set `__test__ = False`
to signal this to the test runner.
"""

__test__ = False  # PyTest: don't collect this module as tests


import os
import torch
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import TrainConfig
from trainer.reinforce_trainer import ReinforceTrainer


def test_ddp_training():
    """Test DDP training functionality."""
    print("Testing DDP training...")
    
    config = TrainConfig(
        model_name="sshleifer/tiny-gpt2",  # Use a small model for testing
        batch_size=2,
        multi_gpu="ddp",
        epochs=1,
        learning_rate=1e-4
    )
    
    try:
        trainer = ReinforceTrainer(config)
        print("✓ DDP trainer created successfully")
        
        # Test forward pass
        trainer.model.eval()
        tokenizer = trainer.tokenizer
        inputs = tokenizer(["Hello world", "Test input"], return_tensors="pt", padding=True)
        inputs = {k: v.to(trainer.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = trainer.model(**inputs)
        
        print(f"✓ DDP forward pass successful, output shape: {outputs.logits.shape}")
        
        # Test backward pass
        trainer.model.train()
        outputs = trainer.model(**inputs)
        loss = outputs.logits.mean()
        
        trainer.optimizer.zero_grad()
        loss.backward()
        trainer.optimizer.step()
        
        print("✓ DDP backward pass successful")
        return True
        
    except Exception as e:
        print(f"✗ DDP training failed: {e}")
        return False


def test_fsdp_training():
    """Test FSDP training functionality."""
    print("Testing FSDP training...")
    
    config = TrainConfig(
        model_name="sshleifer/tiny-gpt2",  # Use a small model for testing
        batch_size=2,
        multi_gpu="fsdp",
        epochs=1,
        learning_rate=1e-4
    )
    
    try:
        trainer = ReinforceTrainer(config)
        print("✓ FSDP trainer created successfully")
        
        # Test forward pass
        trainer.model.eval()
        tokenizer = trainer.tokenizer
        inputs = tokenizer(["Hello world", "Test input"], return_tensors="pt", padding=True)
        inputs = {k: v.to(trainer.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = trainer.model(**inputs)
        
        print(f"✓ FSDP forward pass successful, output shape: {outputs.logits.shape}")
        
        # Test backward pass
        trainer.model.train()
        outputs = trainer.model(**inputs)
        loss = outputs.logits.mean()
        
        trainer.optimizer.zero_grad()
        loss.backward()
        trainer.optimizer.step()
        
        print("✓ FSDP backward pass successful")
        return True
        
    except Exception as e:
        print(f"✗ FSDP training failed: {e}")
        return False


def test_single_gpu_training():
    """Test single GPU training for comparison."""
    print("Testing single GPU training...")
    
    config = TrainConfig(
        model_name="sshleifer/tiny-gpt2",
        batch_size=2,
        multi_gpu="none",
        epochs=1,
        learning_rate=1e-4
    )
    
    try:
        trainer = ReinforceTrainer(config)
        print("✓ Single GPU trainer created successfully")
        
        # Test forward pass
        trainer.model.eval()
        tokenizer = trainer.tokenizer
        inputs = tokenizer(["Hello world", "Test input"], return_tensors="pt", padding=True)
        inputs = {k: v.to(trainer.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = trainer.model(**inputs)
        
        print(f"✓ Single GPU forward pass successful, output shape: {outputs.logits.shape}")
        
        # Test backward pass
        trainer.model.train()
        outputs = trainer.model(**inputs)
        loss = outputs.logits.mean()
        
        trainer.optimizer.zero_grad()
        loss.backward()
        trainer.optimizer.step()
        
        print("✓ Single GPU backward pass successful")
        return True
        
    except Exception as e:
        print(f"✗ Single GPU training failed: {e}")
        return False


def main():
    """Main test function."""
    print("=" * 60)
    print("Testing Distributed Training Functionality")
    print("=" * 60)
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"✓ Number of GPUs: {torch.cuda.device_count()}")
    else:
        print("⚠ CUDA not available, tests will run on CPU")
    
    print()
    
    # Test single GPU training first
    single_gpu_success = test_single_gpu_training()
    print()
    
    # Test DDP training
    ddp_success = test_ddp_training()
    print()
    
    # Test FSDP training
    fsdp_success = test_fsdp_training()
    print()
    
    # Summary
    print("=" * 60)
    print("Test Summary:")
    print("=" * 60)
    print(f"Single GPU Training: {'✓ PASS' if single_gpu_success else '✗ FAIL'}")
    print(f"DDP Training:        {'✓ PASS' if ddp_success else '✗ FAIL'}")
    print(f"FSDP Training:       {'✓ PASS' if fsdp_success else '✗ FAIL'}")
    print()
    
    if all([single_gpu_success, ddp_success, fsdp_success]):
        print("🎉 All tests passed! Distributed training is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 