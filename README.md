from https://github.com/TencentYoutuResearch/ObjectDetection-OneStageDet/tree/master/yolo
env: python3.6
cd utils/test and make # make gpu nms or cpu nms
1. python3.6 labels.py # create dataset
2. sh train.sh # tain
3. python3.6 predict_cv.py # detect local img
