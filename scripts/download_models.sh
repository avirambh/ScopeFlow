#!/bin/bash

echo "Downloading models.."
pushd "$(dirname "$0")/../checkpoints/"

# Download
wget https://www.dropbox.com/s/fjfzp215rpulwqn/scopeflow_sintel_models.tar
wget https://www.dropbox.com/s/eozzcpqcj94y1ai/scopeflow_kitti_models.tar
wget https://www.dropbox.com/s/xygsof2267km04q/scopeflow_pre_models.tar

# Extract
for fname in *.tar; do tar xvf $fname; done
rm *.tar

# Move dir
for d in $(ls checkpoints); do echo "Setting $d.."; mv checkpoints/$d .;done
rm -r checkpoints

popd
