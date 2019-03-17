from https://github.com/TencentYoutuResearch/ObjectDetection-OneStageDet/tree/master/yolo

env: python3.6

use:

cd utils/test

make # make gpu nms and cpu nms

1. python3.6 labels.py # create dataset
2. sh train.sh # tain
3. python3.6 predict_cv.py # detect local img
