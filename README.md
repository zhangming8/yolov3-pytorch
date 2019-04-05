# Implementation of yolov3 and yolov3-tiny by pytorch.

most of codes are copied from here: https://github.com/TencentYoutuResearch/ObjectDetection-OneStageDet/tree/master/yolo

The original code was hard to understand, so I reorganized the code structure and added new features.

# Additional functions
    1. use sigmoid predict(replace softmax) class prob
    2. support train 1 class dataset
    #3. Multi GPU training

# Usage method
    Environment: python3.6, pytroch 0.4

Prepare dataset:

    1. donwnload dataset:
        wget https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar
        wget https://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar
        wget https://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar
        tar xf VOCtrainval_11-May-2012.tar
        tar xf VOCtrainval_06-Nov-2007.tar
        tar xf VOCtest_06-Nov-2007.tar
    2. modify "ROOT" to VOCdevkit folder in labels_voc.py, such as: ROOT = '/media/data/VOCdevkit' 
    3. python3.6 labels_voc.py
    4. the new dataset annotations will be saved in ROOT+"/onedet_cache" folder
    
How to train:

    train yolov3-tiny:
        1. modify "data_root_dir" to your dataset folder in "cfg/tiny_yolov3.yml", such as: '/media/data/VOCdevkit/onedet_cache' 
        2. modify "model_name" to "TinyYolov3" in train_my.py, such as: model_name = "TinyYolov3"
        2. python3.6 train_my.py  # train from scratch
    train yolov3:
        1. download init weight: wget https://pjreddie.com/media/files/darknet53.conv.74
        2. modify "data_root_dir" to your dataset folder in "cfg/yolov3.yml", such as: '/media/data/VOCdevkit/onedet_cache' 
        2. modify "model_name" to "Yolov3" in train_my.py, such as: model_name = "Yolov3"
        2. python3.6 train_my.py
     the trained model will be saved in folder "output"
        
Detect a image:

    python3.6 predict_cv.py
    the images in folder "test_img" will be used, and the results are saved in folder "result" 
