#!/bin/bash

# Script to install PyTorch with RTX 5090 (sm_120) support using nightly builds
# Much faster than building from source!

set -e

echo "======================================"
echo "Installing PyTorch for RTX 5090"
echo "======================================"
echo ""
echo "Using PyTorch nightly with sm_120 support"
echo "This will take ~5-10 minutes."
echo ""

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate spt

# Install PyTorch nightly with CUDA 12.6 (includes sm_120 support)
echo "Installing PyTorch nightly with CUDA 12.6..."
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126

# Set environment for building extensions
export CUDA_HOME=$CONDA_PREFIX
export FORCE_CUDA=1
export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0;12.0"

# Install PyTorch Geometric extensions from source
echo ""
echo "Building PyTorch Geometric extensions with RTX 5090 support..."
echo "This will take ~10-15 minutes..."
pip install --no-build-isolation git+https://github.com/rusty1s/pytorch_scatter.git
pip install --no-build-isolation git+https://github.com/rusty1s/pytorch_sparse.git
pip install --no-build-isolation git+https://github.com/rusty1s/pytorch_cluster.git
pip install --no-build-isolation git+https://github.com/pyg-team/pyg-lib.git

# Rebuild FRNN with new PyTorch
echo ""
echo "Rebuilding FRNN..."
cd /home/pranav/Projects/Dissertation/superpoint_transformer/src/dependencies/FRNN/external/prefix_sum
rm -rf build *.egg-info
python setup.py install

cd ../../
rm -rf build *.egg-info
python setup.py install

cd /home/pranav/Projects/Dissertation/superpoint_transformer

echo ""
echo "======================================"
echo "✅ Installation complete!"
echo "======================================"
echo ""
echo "Testing RTX 5090 support..."
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0)); x = torch.randn(100, 100, device='cuda'); y = x @ x.T; print('✅ RTX 5090 is working!'); print('Matrix multiplication result:', y.sum().item())"

echo ""
echo "You can now run training with:"
echo "  conda activate spt"
echo "  python src/train.py experiment=semantic/s3dis datamodule.fold=5"
