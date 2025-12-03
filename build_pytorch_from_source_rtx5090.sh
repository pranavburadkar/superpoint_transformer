#!/bin/bash

# Script to build PyTorch from SOURCE with RTX 5090 (sm_120) support
# This takes 2-3 hours but will give you full sm_120 kernel support

set -e

echo "======================================"
echo "Building PyTorch from Source for RTX 5090"
echo "======================================"
echo ""
echo "⏱️  This will take 2-3 hours. You can run this overnight."
echo ""

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate spt

# Install build dependencies
echo "Step 1/8: Installing build dependencies..."
echo "Installing GCC 11 (PyTorch v2.5.1 is not compatible with GCC 14)..."
conda install -y gcc=11 gxx=11 -c conda-forge

echo "Upgrading CMake to latest version..."
conda install -y "cmake>=3.20" ninja -c conda-forge

pip install pyyaml typing_extensions

# Set GCC 11 as the compiler
export CC=$(which gcc)
export CXX=$(which g++)
echo "✓ Using GCC: $($CC --version | head -1)"
echo "✓ Using G++: $($CXX --version | head -1)"
echo "✓ Using CMake: $(cmake --version | head -1)"
echo ""

# Clean build directory
BUILD_DIR="/tmp/pytorch_build"
echo "Step 2/8: Setting up build directory..."
if [ -d "$BUILD_DIR/pytorch" ]; then
    echo "Cleaning old build artifacts..."
    rm -rf "$BUILD_DIR/pytorch/build"
    rm -rf "$BUILD_DIR/pytorch/torch/lib"
fi
mkdir -p $BUILD_DIR
cd $BUILD_DIR

# Clone PyTorch if needed
if [ ! -d "pytorch" ]; then
    echo ""
    echo "Step 3/8: Cloning PyTorch v2.5.1 (this may take 10-15 minutes)..."
    git clone --recursive --branch v2.5.1 https://github.com/pytorch/pytorch
else
    echo ""
    echo "Step 3/8: Using existing PyTorch clone..."
fi

cd pytorch

# Set environment variables for RTX 5090
echo ""
echo "Step 4/8: Configuring build environment..."

export CMAKE_PREFIX_PATH=${CONDA_PREFIX}
export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0;12.0"  # Include sm_120 for RTX 5090
export USE_CUDA=1
export USE_CUDNN=1
export MAX_JOBS=4  # Limit to avoid running out of memory
export BUILD_TEST=0  # Don't build tests to save time

# Fix CUDA paths for conda environment
export CUDA_HOME=$CONDA_PREFIX
export CUDA_TOOLKIT_ROOT_DIR=$CONDA_PREFIX
export CUDNN_INCLUDE_DIR=$CONDA_PREFIX/include
export CUDNN_LIBRARY=$CONDA_PREFIX/lib
export CUDA_BIN_PATH=$CONDA_PREFIX/bin
export CUDA_INCLUDE_DIRS=$CONDA_PREFIX/targets/x86_64-linux/include

# Force ABI=0 for compatibility with PyG extensions
export GLIBCXX_USE_CXX11_ABI=0
export CXXFLAGS="-D_GLIBCXX_USE_CXX11_ABI=0 -Wno-error"
export CFLAGS="-Wno-error"

echo "✓ CUDA_HOME: $CUDA_HOME"
echo "✓ CUDA includes: $CUDA_INCLUDE_DIRS"
echo "✓ CUDA architectures: $TORCH_CUDA_ARCH_LIST"
echo "✓ ABI: _GLIBCXX_USE_CXX11_ABI=0 (compatible with PyG)"
echo ""

echo "Step 5/8: Cleaning previous build artifacts..."
python setup.py clean --all 2>/dev/null || true
rm -rf build

echo ""
echo "🔨 Step 6/8: Building PyTorch (this is the long part: ~2 hours)..."
echo "   Building with these CUDA architectures: $TORCH_CUDA_ARCH_LIST"
echo "   (includes sm_120 for your RTX 5090)"
echo ""
echo "   Progress indicators:"
echo "   - You'll see compilation messages scrolling"
echo "   - Build typically completes around [7500/7815] files"
echo "   - If it seems stuck, it's actually compiling (be patient!)"
echo ""

# Build and install
python setup.py install 2>&1 | tee /tmp/pytorch_build.log

echo ""
echo "======================================"
echo "✅ PyTorch build complete!"
echo "======================================"
echo ""

# Test RTX 5090
echo "Step 7/8: Testing RTX 5090 support..."
python -c "
import torch
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('CUDA capability:', torch.cuda.get_device_capability(0))
    x = torch.randn(100, 100, device='cuda')
    y = x @ x.T
    print('✅ Successfully ran matrix multiplication on RTX 5090!')
    print('Result sum:', y.sum().item())
else:
    print('❌ CUDA not available!')
    exit(1)
" || {
    echo ""
    echo "❌ PyTorch built but CUDA test failed!"
    echo "Check /tmp/pytorch_build.log for errors"
    exit 1
}

echo ""
echo "Step 8/8: Rebuilding dependencies with matching ABI..."
cd /home/pranav/Projects/Dissertation/superpoint_transformer

# Rebuild FRNN
export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0;12.0"
export CUDA_HOME=$CONDA_PREFIX

echo "→ Rebuilding prefix_sum..."
cd src/dependencies/FRNN/external/prefix_sum
rm -rf build *.egg-info
python setup.py install

echo "→ Rebuilding FRNN..."
cd ../../
rm -rf build *.egg-info
python setup.py install

cd /home/pranav/Projects/Dissertation/superpoint_transformer

# Rebuild PyG extensions
echo "→ Rebuilding PyG extensions (this takes ~15 minutes)..."
pip uninstall torch-scatter torch-sparse torch-cluster -y 2>/dev/null || true

export FORCE_CUDA=1
export TORCH_CUDA_ARCH_LIST="12.0"  # Only build for RTX 5090 to save time

pip install --no-build-isolation git+https://github.com/rusty1s/pytorch_scatter.git
pip install --no-build-isolation git+https://github.com/rusty1s/pytorch_sparse.git
pip install --no-build-isolation git+https://github.com/rusty1s/pytorch_cluster.git

echo ""
echo "======================================"
echo "🎉 Everything is ready for RTX 5090!"
echo "======================================"
echo ""
echo "Build log saved to: /tmp/pytorch_build.log"
echo ""
echo "You can now train with:"
echo "  conda activate spt"
echo "  python src/train.py experiment=semantic/s3dis datamodule.fold=5"
echo ""
