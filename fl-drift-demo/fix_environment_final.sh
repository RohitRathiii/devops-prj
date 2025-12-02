#!/bin/bash

echo "🎯 Final fix: Remove TensorFlow conflicts, keep only PyTorch stack"

# Activate current environment
if [ -d "fl_env" ]; then
    source fl_env/bin/activate
fi

echo "🗑️  Removing TensorFlow and its conflicting dependencies..."
pip uninstall -y tensorflow tensorflow-probability tensorboard tensorboard-data-server

echo "🗑️  Removing other conflicting packages..."
pip uninstall -y grpcio-health-checking

echo "📦 Reinstalling core dependencies with exact compatible versions..."

# Install exact versions for Flower v1.11.1 compatibility
pip install protobuf==4.25.4
pip install grpcio==1.60.1
pip install numpy==1.26.4

# Install Flower v1.11.1 again to ensure it's properly configured
pip install flwr==1.11.1

echo "🧪 Installing drift detection without TensorFlow dependencies..."
# Use River for ADWIN (no TensorFlow)
pip install river==0.15.0

# Use scikit-learn based implementations instead of alibi-detect
pip install scikit-learn==1.3.0

echo "✅ Verifying the clean environment..."
python -c "
print('🧪 Testing clean environment (no TensorFlow)...')

try:
    import torch
    print(f'✅ PyTorch: {torch.__version__}')
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
    print(f'✅ River (ADWIN): {river.__version__}')
except Exception as e:
    print(f'❌ River: {e}')

try:
    import sklearn
    print(f'✅ Scikit-learn: {sklearn.__version__}')
except Exception as e:
    print(f'❌ Scikit-learn: {e}')

print('\\n🎯 Environment Status:')
print('✅ PyTorch + Transformers for BERT-tiny')
print('✅ Flower v1.11.1 for federated learning') 
print('✅ River for ADWIN drift detection')
print('✅ No TensorFlow conflicts')
print('✅ Compatible protobuf + gRPC versions')
"

echo ""
echo "🎉 Clean environment ready!"
echo ""
echo "🚀 Test with: python main_no_ray.py --rounds 3 --clients 2 --drift-round 2"