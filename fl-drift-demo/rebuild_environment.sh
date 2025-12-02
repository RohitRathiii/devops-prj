#!/bin/bash

echo "🧹 Completely rebuilding environment to fix mutex lock conflicts..."

# Get current directory
CURRENT_DIR=$(pwd)

# Deactivate current environment if active
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "📦 Deactivating current environment..."
    deactivate
fi

# Remove the problematic environment
echo "🗑️  Removing conflicted fl_env..."
rm -rf fl_env

# Create fresh environment
echo "✨ Creating fresh Python environment..."
python3 -m venv fl_env_clean
source fl_env_clean/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install minimal compatible versions in order
echo "📦 Installing compatible packages step by step..."

# Core ML packages first (avoiding TensorFlow)
echo "1. Installing NumPy..."
pip install numpy==1.24.3

echo "2. Installing PyTorch (CPU only to avoid conflicts)..."
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu

echo "3. Installing transformers..."
pip install transformers==4.30.0

echo "4. Installing datasets..."
pip install datasets==2.12.0

# Drift detection packages (avoiding TensorFlow-dependent ones)
echo "5. Installing River (ADWIN)..."
pip install river==0.15.0

echo "6. Installing scikit-learn for basic drift detection..."
pip install scikit-learn==1.3.0

# Flower with specific compatible versions
echo "7. Installing protobuf and gRPC..."
pip install protobuf==4.23.4
pip install grpcio==1.56.2

echo "8. Installing Flower (older compatible version)..."
pip install flwr==1.4.0

# Additional packages
echo "9. Installing other dependencies..."
pip install pandas==2.0.3
pip install matplotlib==3.7.2
pip install seaborn==0.12.2
pip install pyyaml==6.0
pip install tqdm==4.65.0

# Verification
echo "✅ Verifying installation..."
python -c "
print('🧪 Testing imports...')

try:
    import numpy as np
    print(f'✅ NumPy: {np.__version__}')
except Exception as e:
    print(f'❌ NumPy: {e}')

try:
    import torch
    print(f'✅ PyTorch: {torch.__version__}')
    print(f'   MPS available: {torch.backends.mps.is_available() if hasattr(torch.backends, \"mps\") else False}')
except Exception as e:
    print(f'❌ PyTorch: {e}')

try:
    import transformers
    print(f'✅ Transformers: {transformers.__version__}')
except Exception as e:
    print(f'❌ Transformers: {e}')

try:
    import google.protobuf
    print(f'✅ Protobuf: {google.protobuf.__version__}')
except Exception as e:
    print(f'❌ Protobuf: {e}')

try:
    import grpc
    print(f'✅ gRPC: {grpc.__version__}')
except Exception as e:
    print(f'❌ gRPC: {e}')

try:
    import flwr
    print(f'✅ Flower: {flwr.__version__}')
except Exception as e:
    print(f'❌ Flower: {e}')

try:
    import river
    print(f'✅ River: {river.__version__}')
except Exception as e:
    print(f'❌ River: {e}')

print('\\n🎯 If all imports succeeded, environment is ready!')
"

echo ""
echo "🎉 Environment rebuild complete!"
echo ""
echo "📝 To use the new environment:"
echo "   source fl_env_clean/bin/activate"
echo ""
echo "🚀 Then test with a simple version:"
echo "   python simple_fl_test.py"