#!/bin/bash

# Fix critical protobuf version conflict causing mutex lock error
echo "🔧 Fixing protobuf version conflict for Flower compatibility..."

# Activate virtual environment if it exists
if [ -d "fl_env" ]; then
    echo "📦 Activating virtual environment..."
    source fl_env/bin/activate
fi

# Uninstall conflicting protobuf version
echo "🗑️  Removing conflicting protobuf 6.x..."
pip uninstall -y protobuf grpcio

# Install compatible versions
echo "⬇️  Installing compatible protobuf 4.x for Flower..."
pip install protobuf==4.25.4
pip install grpcio==1.60.1

# Reinstall Flower to ensure compatibility
echo "🌸 Reinstalling Flower with compatible dependencies..."
pip uninstall -y flwr
pip install "flwr[simulation]==1.11.1"

# Install Ray with compatible version
echo "☀️  Installing compatible Ray version..."
pip uninstall -y ray
pip install ray==2.30.0

# Verify installations
echo "✅ Verifying compatible installations..."

python -c "
import sys
print('Python version:', sys.version)

try:
    import google.protobuf
    print(f'✅ protobuf: {google.protobuf.__version__}')
except Exception as e:
    print(f'❌ protobuf error: {e}')

try:
    import grpc
    print(f'✅ grpcio: {grpc.__version__}')
except Exception as e:
    print(f'❌ grpcio error: {e}')

try:
    import flwr
    print(f'✅ flwr: {flwr.__version__}')
except Exception as e:
    print(f'❌ flwr error: {e}')

try:
    import ray
    print(f'✅ ray: {ray.__version__}')
except Exception as e:
    print(f'❌ ray error: {e}')

print('\\n🎯 Compatibility Check:')
if hasattr(google.protobuf, '__version__'):
    version = google.protobuf.__version__
    if version.startswith('4.'):
        print('✅ protobuf version compatible with Flower')
    else:
        print(f'❌ protobuf {version} may cause conflicts')
"

echo "🎉 Dependency fix complete!"
echo ""
echo "🚀 Now test with: python main.py --rounds 3 --clients 2 --drift-round 2"