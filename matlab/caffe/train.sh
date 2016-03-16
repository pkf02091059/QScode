TOOLS=build/tools
EXAMPLE=models/haonet
MODEL=models
GLOG_logtostderr=1 $TOOLS/caffe train -solver=$EXAMPLE/solver.prototxt
