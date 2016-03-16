#!/usr/bin/env sh
TOOLS=build/tools
EXAMPLE=data/MyData
MODEL=models
GLOG_logtostderr=1 $TOOLS/caffe train -solver=$EXAMPLE/finetune_solver.prototxt -weights=$EXAMPLE/bvlc_reference_caffenet.caffemodel #-weights=$MODEL/clothes/finetune_clothes_iter_1000.caffemodel #-weights=$MODEL/face_iter_70000.caffemodel #-weights=$MODEL/finetune_dog_iter_1000.caffemodel #-weights=$EXAMPLE/bvlc_reference_caffenet.caffemodel

echo "Done."
