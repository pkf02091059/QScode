#!/usr/bin/env sh
# Compute the mean image from the imagenet training leveldb
# N.B. this is available in data/ilsvrc12

./build/tools/compute_image_mean /media/F/train_data/face/train_lmdb \
  data/MyData/my_image_mean.binaryproto

echo "Done."
