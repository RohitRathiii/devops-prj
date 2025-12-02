#!/usr/bin/env python3
"""
Minimal test to isolate the mutex lock issue.

This script tests components one by one to identify exactly where the mutex lock occurs.
"""

import os
import sys
from pathlib import Path

# Set threading environment variables
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1" 
os.environ["MKL_NUM_THREADS"] = "1"

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("🔍 Testing components step by step to isolate mutex lock...")

# Test 1: Basic imports
print("\n1. Testing basic Python imports...")
try:
    import numpy as np
    import torch
    print("✅ NumPy and PyTorch imports successful")
except Exception as e:
    print(f"❌ Basic imports failed: {e}")
    sys.exit(1)

# Test 2: Protobuf
print("\n2. Testing protobuf...")
try:
    import google.protobuf
    print(f"✅ Protobuf version: {google.protobuf.__version__}")
except Exception as e:
    print(f"❌ Protobuf import failed: {e}")

# Test 3: gRPC
print("\n3. Testing gRPC...")
try:
    import grpc
    print(f"✅ gRPC version: {grpc.__version__}")
except Exception as e:
    print(f"❌ gRPC import failed: {e}")

# Test 4: Flower basic import
print("\n4. Testing Flower basic import...")
try:
    import flwr
    print(f"✅ Flower version: {flwr.__version__}")
except Exception as e:
    print(f"❌ Flower import failed: {e}")
    print("   This is expected if protobuf runtime_version error occurred")

# Test 5: Our local modules
print("\n5. Testing our local modules...")
try:
    from fed_drift.models import create_model, get_device
    print("✅ Local models module imported")
    
    from fed_drift.data import FederatedDataLoader
    print("✅ Local data module imported")
    
    from fed_drift.drift_detection import ADWINDriftDetector
    print("✅ Local drift detection module imported")
    
except Exception as e:
    print(f"❌ Local modules failed: {e}")

# Test 6: Model creation
print("\n6. Testing model creation...")
try:
    device = get_device()
    print(f"✅ Device detected: {device}")
    
    model = create_model()
    print("✅ Model created successfully")
    print(f"   Model type: {type(model)}")
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
except Exception as e:
    print(f"❌ Model creation failed: {e}")

# Test 7: Data loading
print("\n7. Testing data loading...")
try:
    config = {
        'model': {
            'model_name': 'prajjwal1/bert-tiny',
            'num_classes': 4,
            'max_length': 128
        },
        'federated': {
            'num_clients': 2,
            'partition_method': 'dirichlet',
            'alpha': 0.5
        }
    }
    
    data_loader = FederatedDataLoader(config)
    print("✅ Data loader created")
    
    # Try to load a small sample
    client_data = data_loader.load_federated_data(num_clients=2, apply_drift=False)
    print(f"✅ Sample data loaded: {len(client_data)} clients")
    
except Exception as e:
    print(f"❌ Data loading failed: {e}")
    import traceback
    traceback.print_exc()

# Test 8: Simple model training
print("\n8. Testing simple model training...")
try:
    if 'model' in locals() and 'client_data' in locals():
        # Get first client's data
        train_data = client_data[0]['train']
        
        # Take a small sample
        sample_data = []
        for i, batch in enumerate(train_data):
            sample_data.append(batch)
            if i >= 1:  # Just 2 batches
                break
        
        print(f"✅ Got {len(sample_data)} training batches")
        
        # Try one forward pass
        model.eval()
        with torch.no_grad():
            batch = sample_data[0]
            inputs = {
                'input_ids': batch['input_ids'],
                'attention_mask': batch['attention_mask']
            }
            outputs = model(**inputs)
            print("✅ Forward pass successful")
            print(f"   Output shape: {outputs.logits.shape}")
    
except Exception as e:
    print(f"❌ Model training test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n🎯 Component testing completed!")
print("\nIf all tests passed, the mutex lock is likely in Flower's distributed components.")
print("If any test failed, that's where we need to focus the fix.")