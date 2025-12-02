#!/usr/bin/env python3
"""
Debug script to identify the exact source of the mutex lock failure.
"""

import os
import sys
import traceback

# Set minimal threading environment
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

print("🔍 Debugging mutex lock failure...")
print(f"Python version: {sys.version}")
print(f"Platform: {sys.platform}")

# Test individual imports one by one
imports_to_test = [
    ("os", "import os"),
    ("sys", "import sys"),
    ("pathlib", "import pathlib"),
    ("numpy", "import numpy as np"),
    ("torch basic", "import torch"),
    ("torch device", "import torch; torch.cuda.is_available()"),
    ("torch mps", "import torch; torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False"),
    ("transformers", "import transformers"),
    ("datasets", "import datasets"), 
    ("google.protobuf", "import google.protobuf"),
    ("grpc", "import grpc"),
]

for name, import_code in imports_to_test:
    print(f"\n🧪 Testing: {name}")
    try:
        exec(import_code)
        print(f"   ✅ {name} imported successfully")
    except Exception as e:
        print(f"   ❌ {name} failed: {e}")
        traceback.print_exc()
        break

print("\n🔧 If mutex lock occurred above, the issue is in one of those basic imports.")
print("💡 This suggests a fundamental threading/library conflict in the environment.")

# Check library versions that might cause conflicts
print("\n📋 Checking installed versions:")
try:
    import subprocess
    result = subprocess.run([sys.executable, "-m", "pip", "list", "|", "grep", "-E", "(torch|numpy|protobuf|grpc|flwr)"], 
                          shell=True, capture_output=True, text=True)
    print("Installed packages:")
    print(result.stdout)
except Exception as e:
    print(f"Could not check versions: {e}")

print("\n💡 Recommendations:")
print("1. This appears to be a fundamental library conflict")
print("2. Consider creating a fresh virtual environment")
print("3. The issue might be with PyTorch compiled for different architecture")
print("4. Check if this is an Apple Silicon vs Intel compatibility issue")