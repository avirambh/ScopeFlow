#!/bin/bash


echo "This is a full installation script, use with caution!"

pushd "$(dirname "$0")/../"

echo "Creating venv.."
virtualenv venv --python=python3.6
. venv/bin/activate
pip3 install -r requirements_cuda8.txt

echo "Installing correlation.. (tested for Titan-X, Pytorch 0.4.1 and Cuda 8)"
sudo apt-get install gcc-5 g++-5 -y
bash -x scripts/install_correlation.sh

echo "Download models.."
bash -x scripts/download_models.sh

echo "Done!"

popd

