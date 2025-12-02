#!/bin/bash

# Federated Learning Drift Detection System - Dependency Installation Script
# Fixes alibi-detect TensorFlow backend and other dependencies

echo "🔧 Installing Federated Learning Drift Detection Dependencies..."

# Activate virtual environment if it exists
if [ -d "fl_env" ]; then
    echo "📦 Activating virtual environment..."
    source fl_env/bin/activate
fi

# Install updated requirements with TensorFlow backend
echo "🚀 Installing core dependencies..."
pip install --upgrade pip

# Install alibi-detect with TensorFlow backend specifically
echo "🎯 Installing alibi-detect with TensorFlow backend..."
pip install "alibi-detect[tensorflow]>=0.12.0"

# Install other requirements
echo "📚 Installing remaining requirements..."
pip install -r requirements.txt

# Verify installations
echo "✅ Verifying installations..."

python -c "
import torch
print(f'✅ PyTorch: {torch.__version__}')

import flwr
print(f'✅ Flower: {flwr.__version__}')

try:
    from alibi_detect.cd import MMDDrift
    print('✅ alibi-detect: MMDDrift available')
except Exception as e:
    print(f'❌ alibi-detect issue: {e}')

try:
    import tensorflow as tf
    print(f'✅ TensorFlow: {tf.__version__}')
except Exception as e:
    print(f'⚠️  TensorFlow: {e}')

try:
    import evidently
    print(f'✅ Evidently: {evidently.__version__}')
except Exception as e:
    print(f'❌ Evidently issue: {e}')

try:
    import river
    print(f'✅ River: {river.__version__}')
except Exception as e:
    print(f'❌ River issue: {e}')
"

echo "🎉 Installation complete! You can now run the simulation with:"
echo "   python main.py --rounds 5 --clients 2 --drift-round 3"