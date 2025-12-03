#!/bin/bash

# Installation script for Superpoint Transformer on RTX 5090
# This uses newer versions compatible with CUDA compute capability sm_120

# Project configuration
PROJECT_NAME=spt
PYTHON=3.10
TORCH=2.5.1
CUDA_VERSION=12.6

# Recover the project's directory
HERE=`dirname $0`
HERE=`realpath $HERE`
cd $HERE

echo "_____________________________________________"
echo ""
echo "         🧩 Superpoint Transformer 🤖        "
echo "         RTX 5090 Compatible Installer       "
echo ""
echo "_____________________________________________"
echo ""

# Recover conda path
CONDA_DIR=`realpath ~/miniconda3`
if (test -z $CONDA_DIR) || [ ! -d $CONDA_DIR ]
then
  CONDA_DIR=`realpath ~/anaconda3`
fi

while (test -z $CONDA_DIR) || [ ! -d $CONDA_DIR ]
do
    echo "Could not find conda at: "$CONDA_DIR
    read -p "Please provide your conda install directory: " CONDA_DIR
    CONDA_DIR=`realpath $CONDA_DIR`
done

echo "Using conda found at: ${CONDA_DIR}/etc/profile.d/conda.sh"
source ${CONDA_DIR}/etc/profile.d/conda.sh

echo ""
echo "⭐ Creating conda environment '${PROJECT_NAME}'"
echo ""
conda create --name ${PROJECT_NAME} python=${PYTHON} -y

# Activate the env
source ${CONDA_DIR}/etc/profile.d/conda.sh  
conda activate ${PROJECT_NAME}

echo ""
echo "⭐ Installing conda and pip dependencies"
echo ""
conda install pip nb_conda_kernels -y
pip install matplotlib
pip install plotly==5.9.0
pip install "jupyterlab>=3" "ipywidgets>=7.6" jupyter-dash
pip install "notebook>=5.3" "ipywidgets>=7.5"
pip install ipykernel

# Install PyTorch 2.5.1 with CUDA 12.6 support
echo ""
echo "⭐ Installing PyTorch ${TORCH} with CUDA ${CUDA_VERSION}"
echo ""
pip3 install torch==${TORCH} torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

pip install torchmetrics==0.11.4

# Install PyG with CUDA 12.6
echo ""
echo "⭐ Installing PyTorch Geometric"
echo ""
pip install pyg_lib torch_scatter torch_cluster torch_sparse -f https://data.pyg.org/whl/torch-${TORCH}+cu126.html
pip install torch_geometric==2.3.0

# Install other dependencies
pip install plyfile
pip install h5py
pip install colorhash
pip install seaborn
pip install numba
pip install pytorch-lightning
pip install pyrootutils
pip install hydra-core --upgrade
pip install hydra-colorlog
pip install hydra-submitit-launcher
pip install "rich<=14.0"
pip install torch_tb_profiler
pip install wandb
pip install open3d
pip install gdown
pip install ipyfilechooser

echo ""
echo "⭐ Installing CUDA toolkit for compilation"
echo ""
# Install CUDA toolkit via conda for FRNN compilation
conda install -c "nvidia/label/cuda-12.6.0" cuda-toolkit -y
# Also need GCC compatible with CUDA
conda install -c conda-forge gcc_linux-64=12 gxx_linux-64=12 -y
conda install -c conda-forge libxcrypt -y

echo ""
echo "⭐ Installing FRNN"
echo ""
# Clone FRNN if not already present
if [ ! -d "src/dependencies/FRNN" ]; then
    git clone --recursive https://github.com/lxxue/FRNN.git src/dependencies/FRNN
fi

# Set CUDA_HOME for compilation
export CUDA_HOME=$CONDA_PREFIX

# Install prefix_sum
cd src/dependencies/FRNN/external/prefix_sum
python setup.py install

# Install FRNN
cd ../../
python setup.py install
cd ../../../

echo ""
echo "⭐ Installing Point Geometric Features"
echo ""
conda install -c conda-forge libstdcxx-ng -y
pip install git+https://github.com/drprojects/point_geometric_features.git

echo ""
echo "⭐ Installing Parallel Cut-Pursuit"
echo ""
# Clone if not already present
if [ ! -d "src/dependencies/parallel_cut_pursuit" ]; then
    git clone https://gitlab.com/1a7r0ch3/parallel-cut-pursuit.git src/dependencies/parallel_cut_pursuit
fi
if [ ! -d "src/dependencies/grid_graph" ]; then
    git clone https://gitlab.com/1a7r0ch3/grid-graph.git src/dependencies/grid_graph
fi

# Compile the projects
python scripts/setup_dependencies.py build_ext

echo ""
echo "🚀 Successfully installed SPT for RTX 5090!"
echo ""
echo "To use: conda activate ${PROJECT_NAME}"
echo ""
