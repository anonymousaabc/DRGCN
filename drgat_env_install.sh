#!/usr/bin/env bash
# make sure command is : source drgcn_env_install.sh

source ~/.bashrc
export TORCH_CUDA_ARCH_LIST="7.0;7.5"   # v100: 7.0; 2080ti: 7.5; titan xp: 6.1

# make sure system cuda version is the same with pytorch cuda
export PATH=/usr/local/cuda-10.2/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64:$LD_LIBRARY_PATH

conda create -n drgcnenv
conda activate drgcnenv
# make sure pytorch version >=1.4.0
conda install -y pytorch=1.9.0 torchvision cudatoolkit=10.2 python=3.7 -c pytorch
pip install tensorboard

# command to install pytorch geometric, please refer to the official website for latest installation.
#  https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html
CUDA=cu102
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.9.0+${CUDA}.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.9.0+${CUDA}.html
pip install torch-geometric

pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.9.0+${CUDA}.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.9.0+${CUDA}.html

pip install requests

# install useful modules
pip install tqdm

# additional package required for ogb experiments
pip install ogb==1.3.2

# additional package required for dgl implementation
pip install dgl-cu102
