import os
import cv2
import glob
import shutil
import time
import logging as log
import torch
from torchvision import transforms as tf
from pprint import pformat
from PIL import Image, ImageOps
import numpy as np
import sys
sys.path.insert(0, '.')

from utils.envs import initEnv
import models


def py_cpu_nms(dets, thresh=0.5):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def detect(net, img_path, use_cuda, network_size, nms_thresh):
    data = Image.open(img_path)
    #data = cv2.imread(img_path)
    #data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
    st = time.time()
    orig_width, orig_height = data.size  # use Image.open
    #orig_height, orig_width, _ = data.shape
    netw, neth = network_size
    scale = min(float(netw) / orig_width, float(neth) / orig_height)
    new_width = orig_width * scale
    new_height = orig_height * scale
    pad_w = (netw - new_width) / 2.0
    pad_h = (neth - new_height) / 2.0

    #data = process_cv(data, network_size)
    data = process_data(data, network_size)  # use Image.open
    data = tf.ToTensor()(data)
    data = data.unsqueeze(0)

    if use_cuda:
        data = data.cuda()
    s1 = time.time()
    with torch.no_grad():
        output, _ = net(data)
    res, res_label = [], []
    # conver x,y,w,h to x1,y1,x2,y2
    for o in output[0]:
        xmin = o.x_top_left
        ymin = o.y_top_left
        xmax = xmin + o.width
        ymax = ymin + o.height
        conf = o.confidence
        class_label = o.class_label

        x1 = max(0, float(xmin - pad_w) / scale)
        x2 = min(orig_width - 1, float(xmax - pad_w) / scale)
        y1 = max(0, float(ymin - pad_h) / scale)
        y2 = min(orig_height - 1, float(ymax - pad_h) / scale)
        res.append([x1, y1, x2, y2, conf])
        res_label.append([int(x1), int(y1), int(x2), int(y2), conf, class_label])
    if len(res) == 0:
        return []
    # do nms
    nms_keep = py_cpu_nms(np.array(res), nms_thresh)
    final_res = []
    for index in nms_keep:
        x1, y1, x2, y2 = res_label[index][0], res_label[index][1], res_label[index][2], res_label[index][3]
        conf = res_label[index][4]
        class_label = res_label[index][5]
        final_res.append((class_label, conf, [x1, y1, x2, y2]))
    end = time.time()
    print("detect time: ", end-st)
    return final_res


def process_cv(img, dimension):
    """ Letterbox and image to fit in the network """
    fill_color = 127
    net_w, net_h = dimension
    im_h, im_w = img.shape[:2]

    if im_w == net_w and im_h == net_h:
        scale = None
        pad = None
        return img

    # Rescaling
    if im_w / net_w >= im_h / net_h:
        scale = net_w / im_w
    else:
        scale = net_h / im_h
    if scale != 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        im_h, im_w = img.shape[:2]

    if im_w == net_w and im_h == net_h:
        pad = None
        return img

    # Padding
    channels = img.shape[2] if len(img.shape) > 2 else 1
    pad_w = (net_w - im_w) / 2
    pad_h = (net_h - im_h) / 2
    pad = (int(pad_w), int(pad_h), int(pad_w + .5), int(pad_h + .5))
    img = cv2.copyMakeBorder(img, pad[1], pad[3], pad[0], pad[2], cv2.BORDER_CONSTANT,
                             value=(fill_color,) * channels)
    return img


def process_data(img, dimension):
    fill_color = 127
    net_w, net_h = dimension
    im_w, im_h = img.size
    if im_w == net_w and im_h == net_h:
        return img

    # Rescaling
    if im_w / net_w >= im_h / net_h:
        scale = net_w / im_w
    else:
        scale = net_h / im_h
    if scale != 1:
        resample_mode = Image.NEAREST  # Image.BILINEAR if self.scale > 1 else Image.ANTIALIAS
        img = img.resize((int(scale * im_w), int(scale * im_h)), resample_mode)
        im_w, im_h = img.size

    if im_w == net_w and im_h == net_h:
        return img

    # Padding
    img_np = np.array(img)
    channels = img_np.shape[2] if len(img_np.shape) > 2 else 1
    pad_w = (net_w - im_w) / 2
    pad_h = (net_h - im_h) / 2
    pad = (int(pad_w), int(pad_h), int(pad_w+.5), int(pad_h+.5))
    img = ImageOps.expand(img, border=pad, fill=(fill_color,)*channels)
    return img


class HyperParams(object):
    def __init__(self, config):
        self.cuda = True
        self.labels = config['labels']
        self.classes = len(self.labels)
        self.data_root = config['data_root_dir']
        self.model_name = config['model_name']

        cur_cfg = config
        dataset = cur_cfg['dataset']
        self.testfile = f'{self.data_root}/{dataset}.pkl'
        self.nworkers = cur_cfg['nworkers']
        self.pin_mem = cur_cfg['pin_mem']
        self.network_size = cur_cfg['input_shape']
        self.batch = cur_cfg['batch_size']
        self.weights = cur_cfg['weights']
        self.conf_thresh = cur_cfg['conf_thresh']
        self.nms_thresh = cur_cfg['nms_thresh']
        self.results = cur_cfg['results']

        # cuda check
        if self.cuda:
            if not torch.cuda.is_available():
                log.debug('CUDA not available')
                self.cuda = False
            else:
                log.debug('CUDA enabled')


def voc_test(hyper_params):
    model_name = hyper_params.model_name
    use_cuda = hyper_params.cuda
    weights = hyper_params.weights
    conf_thresh = hyper_params.conf_thresh
    network_size = hyper_params.network_size
    labels = hyper_params.labels
    nms_thresh = hyper_params.nms_thresh
    save_dir = hyper_params.results
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.mkdir(save_dir)

    test_args = {'conf_thresh': conf_thresh, 'network_size': network_size, 'labels': labels}
    net = models.__dict__[model_name](hyper_params.classes, weights, train_flag=2, test_args=test_args)
    net.eval()
    log.info('Net structure\n%s' % net)
    if use_cuda:
        net.cuda()
    
    img_path = "/media/lishundong/DATA2/docker/data/VOCdevkit/VOC2007/JPEGImages"
    #img_path = "/home/lishundong/Desktop/yolov3_pytorch/test_img"
    img_list = glob.glob(img_path + "/*.jpg")
    for idx, img_path in enumerate(img_list):
        print("--------------------------------")
        result = detect(net, img_path, use_cuda, network_size, nms_thresh)
        img = cv2.imread(img_path)
        for res in result:
            class_label, conf, box = res
            x1, y1, x2, y2 = box
            cv2.putText(img, class_label + ":"+str(conf)[:5], (max(0, x1), max(15, y1)), cv2.FONT_ITALIC, 0.6, (0, 0, 255), 2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255))
            cv2.imwrite(os.path.join(save_dir, os.path.basename(img_path)), img)
        print("detect: {}".format(result))
        print("save: {} -> {}".format(img_path, os.path.join(save_dir, os.path.basename(img_path))))


if __name__ == '__main__':
    #model_name = "Yolov3"
    model_name = "TinyYolov3"
    train_flag = 2
    config = initEnv(train_flag=train_flag, model_name=model_name)
    log.info('Config\n\n%s\n' % pformat(config))
    # init env
    hyper_params = HyperParams(config)
    voc_test(hyper_params)
