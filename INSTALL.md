# Installation

## Requirements

* Python 3.7
* Pytorch 1.2
* cuda 11.0

## Setup with Conda


```sh
# create a new environment
conda create --name CLEVER python=3.7
conda activate CLEVER

# install pytorch
conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=11.0 -c pytorch

export INSTALL_DIR=$PWD

# install apex
cd $INSTALL_DIR
git clone https://github.com/NVIDIA/apex.git
cd apex
python setup.py install --cuda_ext --cpp_ext

# install oscar
cd $INSTALL_DIR
git clone --recursive https://github.com/thunlp/CLEVER.git
cd CLEVER/src/coco_caption
./get_stanford_models.sh
cd ..
python setup.py build develop

# install requirements
pip install -r requirements.txt

unset INSTALL_DIR
```