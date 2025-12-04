# RTX 5090 Setup Guide for Superpoint Transformer

Complete installation guide for running this project on **NVIDIA RTX 5090** (Blackwell architecture, sm_120).

## 🎯 Problem Summary

The RTX 5090 uses the new **sm_120** (compute capability 12.0) architecture, which is not supported by:
- ❌ PyTorch stable releases (up to 2.6.0)
- ❌ PyTorch nightly with CUDA 12.6 or earlier
- ❌ Pre-built wheels for PyTorch Geometric extensions

**Solution**: Use PyTorch nightly with **CUDA 12.8 (cu128)** which includes pre-compiled sm_120 kernels.

---

## 📋 Prerequisites

- NVIDIA RTX 5090 GPU
- CUDA 12.8+ drivers installed
- Conda/Miniconda
- Linux OS (tested on Ubuntu)

Check your CUDA driver version:
```bash
nvidia-smi
```

---

## 🚀 Installation Steps

### 1. Clean Environment Setup

Remove any existing installations:

```bash
conda activate spt

# Remove old PyTorch
pip uninstall torch torchvision torchaudio -y

# Remove PyG extensions
pip uninstall torch-scatter torch-sparse torch-cluster pyg-lib frnn -y

# Clean FRNN build artifacts
cd /path/to/superpoint_transformer
rm -rf src/dependencies/FRNN/build src/dependencies/FRNN/*.egg-info
rm -rf src/dependencies/FRNN/external/prefix_sum/build src/dependencies/FRNN/external/prefix_sum/*.egg-info

# Clean caches
conda clean --all -y
pip cache purge
```

### 2. Install PyTorch Nightly (cu128)

This is the **critical step** - must use CUDA 12.8:

```bash
conda activate spt
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

**Verify installation**:
```bash
python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA:', torch.version.cuda)
print('GPU:', torch.cuda.get_device_name(0))
print('Compute capability:', torch.cuda.get_device_capability(0))

# Test CUDA operation
x = torch.randn(1000, 1000, device='cuda')
y = x @ x.T
print('✅ Matrix multiplication successful!')
print('Result sum:', y.sum().item())
"
```

**Expected output**:
```
PyTorch: 2.10.0.dev20251202+cu128
CUDA: 12.8
GPU: NVIDIA GeForce RTX 5090
Compute capability: (12, 0)
✅ Matrix multiplication successful!
Result sum: <some number>
```

### 3. Install PyTorch Geometric Extensions

Build from source with sm_120 support:

```bash
conda activate spt

export CUDA_HOME=$CONDA_PREFIX
export FORCE_CUDA=1
export TORCH_CUDA_ARCH_LIST="12.0"

# Install torch-geometric first
pip install torch-geometric

# Build PyG extensions from source
pip install --no-build-isolation git+https://github.com/rusty1s/pytorch_scatter.git
pip install --no-build-isolation git+https://github.com/rusty1s/pytorch_sparse.git
pip install --no-build-isolation git+https://github.com/rusty1s/pytorch_cluster.git
```

**Note**: This takes 20-30 minutes total.

**Verify**:
```bash
python -c "
from torch_scatter import scatter_mean
from torch_sparse import SparseTensor
from torch_cluster import grid_cluster
print('✅ All PyG extensions imported successfully!')
"
```

### 4. Build FRNN with RTX 5090 Support

FRNN requires explicit sm_120 architecture flags.

**Update FRNN setup files** (already done if you're using this modified repo):

The following files have been modified to include sm_120:
- `src/dependencies/FRNN/setup.py`
- `src/dependencies/FRNN/external/prefix_sum/setup.py`

**Build FRNN**:

```bash
cd /path/to/superpoint_transformer

export CUDA_HOME=$CONDA_PREFIX
export TORCH_CUDA_ARCH_LIST="9.0;12.0"

# Build prefix_sum
cd src/dependencies/FRNN/external/prefix_sum
python setup.py install

# Build FRNN
cd ../../
python setup.py install

cd ../../../
```

**Note**: FRNN compilation takes 10-15 minutes.

**Verify FRNN**:
```bash
python -c "
from src.dependencies.FRNN import frnn
import torch
pts = torch.randn(10, 100, 3, device='cuda')
dists, idxs, _, _ = frnn.frnn_grid_points(pts, pts, K=5, r=0.1)
print('✅ FRNN works on RTX 5090!')
print('Neighbors shape:', idxs.shape)
"
```

---

## ✅ Verification

Test the full training pipeline:

```bash
conda activate spt
python src/train.py experiment=semantic/s3dis datamodule.fold=5
```

**Expected**: Training should start without CUDA errors. You'll see config output and the model will begin training.

> [!NOTE]
> If you get "dataset not found" errors, you need to set up the S3DIS dataset following `docs/datasets.md`.

---

## 🔧 Troubleshooting

### Out of Memory Error

**Symptom**: `torch.OutOfMemoryError: CUDA out of memory`

**Solution**: Reduce batch size or enable memory optimizations:

```bash
# Option 1: Smaller batch size
python src/train.py experiment=semantic/s3dis datamodule.fold=5 datamodule.batch_size=4

# Option 2: Enable PyTorch memory management
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python src/train.py experiment=semantic/s3dis datamodule.fold=5

# Option 3: Both
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python src/train.py experiment=semantic/s3dis datamodule.fold=5 datamodule.batch_size=4
```

### "No kernel image available" Error

**Symptom**: `RuntimeError: CUDA error: no kernel image is available for execution on the device`

**Cause**: Wrong PyTorch CUDA version (not cu128)

**Solution**: Reinstall PyTorch cu128:
```bash
pip uninstall torch torchvision torchaudio -y
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

Verify CUDA version:
```bash
python -c "import torch; print('CUDA:', torch.version.cuda)"
```
Should output: `CUDA: 12.8`

### FRNN Import Error

**Symptom**: `ModuleNotFoundError: No module named 'frnn'`

**Solution**: FRNN not installed properly. Rebuild:
```bash
cd src/dependencies/FRNN
rm -rf build *.egg-info
export CUDA_HOME=$CONDA_PREFIX
export TORCH_CUDA_ARCH_LIST="9.0;12.0"
python setup.py install
```

### PyG Extension Errors

**Symptom**: `OSError: undefined symbol` or `ImportError` for torch_scatter/sparse/cluster

**Solution**: Rebuild PyG extensions:
```bash
pip uninstall torch-scatter torch-sparse torch-cluster -y

export CUDA_HOME=$CONDA_PREFIX
export FORCE_CUDA=1
export TORCH_CUDA_ARCH_LIST="12.0"

pip install --no-build-isolation git+https://github.com/rusty1s/pytorch_scatter.git
pip install --no-build-isolation git+https://github.com/rusty1s/pytorch_sparse.git
pip install --no-build-isolation git+https://github.com/rusty1s/pytorch_cluster.git
```

---

## 📊 Performance Notes

### RTX 5090 Specifications
- **VRAM**: 32 GB GDDR7
- **Compute Capability**: 12.0 (sm_120)
- **Architecture**: Blackwell

### Recommended Settings

For optimal performance on RTX 5090:

```bash
# Enable TF32 for faster mixed-precision training
export NVIDIA_TF32_OVERRIDE=1

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Training command
python src/train.py \
  experiment=semantic/s3dis \
  datamodule.fold=5 \
  datamodule.batch_size=8 \
  trainer.precision=16-mixed
```

---

## 🔍 Key Technical Details

### Why cu128 Specifically?

PyTorch's CUDA wheel naming convention:
- `cu124` = CUDA 12.4 (supports up to sm_90 / H100)
- `cu126` = CUDA 12.6 (supports up to sm_90 / H100)
- `cu128` = CUDA 12.8 (includes sm_120 / RTX 5090) ✅

The RTX 5090 requires **sm_120 kernels** which only started shipping in CUDA 12.8 wheels.

### Modified Files

The following files have been modified in this repository to support RTX 5090:

1. **`src/dependencies/FRNN/setup.py`**
   - Added `-gencode=arch=compute_120,code=sm_120`
   - Simplified architecture list to sm_90 and sm_120

2. **`src/dependencies/FRNN/external/prefix_sum/setup.py`**
   - Added `-gencode=arch=compute_120,code=sm_120`
   - Included necessary compiler flags

### Build Time Summary

| Component | Time | sm_120 Support |
|-----------|------|----------------|
| PyTorch cu128 (pre-built) | ~10 min | ✅ Included |
| torch-scatter | ~10 min | ✅ Built |
| torch-sparse | ~10 min | ✅ Built |
| torch-cluster | ~10 min | ✅ Built |
| prefix_sum | ~2 min | ✅ Built |
| FRNN | ~15 min | ✅ Built |
| **Total** | **~60 min** | |

---

## 📚 References

- **NVIDIA Forums Solution**: [RTX 5090 PyTorch Fix](https://forums.developer.nvidia.com/t/rtx-5090-not-working-with-pytorch-and-stable-diffusion-sm-120-unsupported/338015/7)
- **PyTorch Nightly**: https://pytorch.org/get-started/locally/
- **CUDA Compute Capabilities**: https://developer.nvidia.com/cuda-gpus

---

## 🆘 Getting Help

If you encounter issues:

1. **Check CUDA version**: `python -c "import torch; print(torch.version.cuda)"`
   - Must be `12.8` or higher

2. **Check GPU detection**: `python -c "import torch; print(torch.cuda.get_device_name(0))"`
   - Should show `NVIDIA GeForce RTX 5090`

3. **Check compute capability**: `python -c "import torch; print(torch.cuda.get_device_capability(0))"`
   - Should be `(12, 0)`

4. **Test CUDA operation**:
   ```bash
   python -c "import torch; x = torch.randn(10, 10, device='cuda'); print(x.sum())"
   ```
   - Should complete without errors

If all checks pass but training fails, the issue is likely configuration (batch size, dataset) not GPU support.

---

## ✨ Success Criteria

You'll know everything is working when:

✅ PyTorch imports and detects RTX 5090  
✅ CUDA operations execute without "kernel image" errors  
✅ PyG extensions import successfully  
✅ FRNN operations work on GPU  
✅ Training script starts and runs for multiple epochs  

---

## 📝 License

This setup guide is part of the Superpoint Transformer project. Refer to the main [LICENSE](LICENSE) file.

---

**Last Updated**: December 3, 2025  
**Tested On**: NVIDIA RTX 5090, PyTorch 2.10.0.dev (cu128), Ubuntu Linux



# Option 1: Smaller batch
python src/train.py experiment=semantic/s3dis datamodule.fold=5 datamodule.batch_size=4

# Option 2: Use PyTorch's memory management
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python src/train.py experiment=semantic/s3dis datamodule.fold=5
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python src/train.py experiment=semantic/s3dis datamodule.fold=5
