#
#   Lightnet dataset that works with brambox annotations
#   Copyright EAVISE
#

import numpy as np
import os
import sys
import copy
import logging as log
from PIL import Image
from PIL import ImageDraw
import cv2
import random

import brambox.boxes as bbb
try:
    from ._dataloading import Dataset
except:
    sys.path.append("../../")
    from data import Dataset
    import data as mydata

__all__ = ['BramboxDataset']


class BramboxDataset(Dataset):
    """ Dataset for any brambox parsable annotation format.

    Args:
        anno_format (brambox.boxes.formats): Annotation format
        anno_filename (list or str): Annotation filename, list of filenames or expandable sequence
        input_dimension (tuple): (width,height) tuple with default dimensions of the network
        class_label_map (list): List of class_labels
        identify (function, optional): Lambda/function to get image based of annotation filename or image id; Default **replace/add .png extension to filename/id**
        img_transform (torchvision.transforms.Compose): Transforms to perform on the images
        anno_transform (torchvision.transforms.Compose): Transforms to perform on the annotations
        kwargs (dict): Keyword arguments that are passed to the brambox parser
    """
    def __init__(self, anno_format, anno_filename, input_dimension, class_label_map=None, identify=None, img_transform=None, anno_transform=None, **kwargs):
        super().__init__(input_dimension)
        self.img_tf = img_transform
        self.anno_tf = anno_transform
        if callable(identify):
            self.id = identify
        else:
            self.id = lambda name: os.path.splitext(name)[0] + '.png'

        # Get annotations
        self.annos = bbb.parse(anno_format, anno_filename, identify=lambda f: f, class_label_map=class_label_map, **kwargs)
        self.keys = list(self.annos)

        # Add class_ids
        if class_label_map is None:
            log.warn(f'No class_label_map given, annotations wont have a class_id values for eg. loss function')
        for k, annos in self.annos.items():
            for a in annos:
                if class_label_map is not None:
                    try:
                        a.class_id = class_label_map.index(a.class_label)
                    except ValueError as err:
                        raise ValueError(f'{a.class_label} is not found in the class_label_map') from err
                else:
                    a.class_id = 0

        log.info(f'Dataset loaded: {len(self.keys)} images')

    def __len__(self):
        return len(self.keys)

    @Dataset.resize_getitem
    def __getitem__(self, index):
        """ Get transformed image and annotations based of the index of ``self.keys``

        Args:
            index (int): index of the ``self.keys`` list containing all the image identifiers of the dataset.

        Returns:
            tuple: (transformed image, list of transformed brambox boxes)
        """
        if index >= len(self):
            raise IndexError(f'list index out of range [{index}/{len(self)-1}]')

        # use PIL.Image.open
        # img = Image.open(self.id(self.keys[index]))
        # use cv2.imread
        img = cv2.imread(self.id(self.keys[index]))
        #print("load image: {}".format(self.id(self.keys[index])))
        if type(img) == type(None):
            print("image damaged +++++++++++++++++++++++++++++++++++++++++++++++++++++")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        anno = copy.deepcopy(self.annos[self.keys[index]])
        random.shuffle(anno)

        # Transform
        if self.img_tf is not None:
            img = self.img_tf(img)
        if self.anno_tf is not None:
            anno = self.anno_tf(anno)
        return img, anno


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    def identify(img_id):
        # return f'{root}/VOCdevkit/{img_id}.jpg'
        return f'{img_id}'
    class unreal_dataset():
        def __init__(self):
            self.input_dim = [416, 416]
    labels = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

    rf = mydata.transform.RandomFlip(0.5)  # 水平翻转
    lb = mydata.transform.Letterbox(dataset=unreal_dataset())  # 长边变为416，短边等比例缩放使用127填充
    rc = mydata.transform.RandomCrop(0.3)  # 在原图上随机裁剪，裁剪0.3，保留原图0.7
    rc2 = mydata.transform.RandomCropLetterbox(dataset=unreal_dataset(), jitter=0.3)  # 等比例缩放后再随机裁剪
    hsv = mydata.transform.HSVShift(0.1, 1.5, 1.5)  # 随机色度饱和度等
    rot = mydata.transform.RandomRotate(jitter_min=-20, jitter_max=20)

    img_tf = mydata.transform.Compose([rf, hsv, rot, rc2])
    anno_tf = mydata.transform.Compose([rf, rot, rc2])

    data = BramboxDataset('anno_pickle', '/Users/ming/Desktop/tmp/VOCdevkit/onedet_cache/test.pkl', [416, 416], labels, identify, img_tf, anno_tf)
    for img, label in data:
        imgdraw = ImageDraw.Draw(img)
        print("----------------")
        for l in label:
            cls, x1, y1, w, h = l.class_label, l.x_top_left, l.y_top_left, l.width, l.height
            print("label: {}, x: {}, y: {}, w: {}, h: {}".format(cls, x1, y1, w, h))
            x2, y2 = x1+w, y1+h
            imgdraw.rectangle((x1, y1, x2, y2), outline='red')
        plt.imshow(img)
        plt.show()


