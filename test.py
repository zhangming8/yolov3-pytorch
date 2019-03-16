import os
import numpy as np
import argparse
import logging as log
import torch
from torchvision import transforms as tf
from pprint import pformat
import sys
sys.path.insert(0, '.')

from utils.envs import initEnv
import data as mydata
import models
from utils.test import voc_wrapper


class HyperParams(object):
    def __init__(self, config, train_flag=1):

        self.cuda = True
        self.labels = config['labels']
        self.classes = len(self.labels)
        self.data_root = config['data_root_dir']
        self.model_name = config['model_name']

        # cuda check
        if self.cuda:
            if not torch.cuda.is_available():
                log.debug('CUDA not available')
                self.cuda = False
            else:
                log.debug('CUDA enabled')

        if train_flag == 1:
            cur_cfg = config

            self.nworkers = cur_cfg['nworkers']
            self.pin_mem = cur_cfg['pin_mem']
            dataset = cur_cfg['dataset']
            self.trainfile = f'{self.data_root}/{dataset}.pkl'

            self.network_size = cur_cfg['input_shape']

            self.batch = cur_cfg['batch_size']
            self.mini_batch = cur_cfg['mini_batch_size']
            self.max_batches = cur_cfg['max_batches']

            self.jitter = 0.3
            self.flip = 0.5
            self.hue = 0.1
            self.sat = 1.5
            self.val = 1.5

            self.learning_rate = cur_cfg['warmup_lr']
            self.momentum = cur_cfg['momentum']
            self.decay = cur_cfg['decay']
            self.lr_steps = cur_cfg['lr_steps']
            self.lr_rates = cur_cfg['lr_rates']

            self.backup = cur_cfg['backup_interval']
            self.bp_steps = cur_cfg['backup_steps']
            self.bp_rates = cur_cfg['backup_rates']
            self.backup_dir = cur_cfg['backup_dir']

            self.resize = cur_cfg['resize_interval']
            self.rs_steps = []
            self.rs_rates = []

            self.weights = cur_cfg['weights']
            self.clear = cur_cfg['clear']
        elif train_flag == 2:
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

        else:
            cur_cfg = config

            self.network_size = cur_cfg['input_shape']
            self.batch = cur_cfg['batch_size']
            self.max_iters = cur_cfg['max_iters']


class CustomDataset(mydata.BramboxDataset):
    def __init__(self, hyper_params):
        anno = hyper_params.testfile
        root = hyper_params.data_root
        network_size = hyper_params.network_size
        labels = hyper_params.labels

        lb = mydata.transform.Letterbox(network_size)
        it = tf.ToTensor()
        img_tf = mydata.transform.Compose([lb, it])
        anno_tf = mydata.transform.Compose([lb])

        def identify(img_id):
            return f'{img_id}'

        super(CustomDataset, self).__init__('anno_pickle', anno, network_size, labels, identify, img_tf, anno_tf)

    def __getitem__(self, index):
        img, anno = super(CustomDataset, self).__getitem__(index)
        for a in anno:
            a.ignore = a.difficult  # Mark difficult annotations as ignore for pr metric
        return img, anno


def VOCTest(hyper_params):
    log.debug('Creating network')

    model_name = hyper_params.model_name
    batch = hyper_params.batch
    use_cuda = hyper_params.cuda
    weights = hyper_params.weights
    conf_thresh = hyper_params.conf_thresh
    network_size = hyper_params.network_size
    labels = hyper_params.labels
    nworkers = hyper_params.nworkers
    pin_mem = hyper_params.pin_mem
    nms_thresh = hyper_params.nms_thresh
    # prefix = hyper_params.prefix
    results = hyper_params.results

    test_args = {'conf_thresh': conf_thresh, 'network_size': network_size, 'labels': labels}
    net = models.__dict__[model_name](hyper_params.classes, weights, train_flag=2, test_args=test_args)
    net.eval()
    log.info('Net structure\n%s' % net)
    # import pdb
    # pdb.set_trace()
    if use_cuda:
        net.cuda()

    log.debug('Creating dataset')
    loader = torch.utils.data.DataLoader(
        CustomDataset(hyper_params),
        batch_size=batch,
        shuffle=False,
        drop_last=False,
        num_workers=nworkers if use_cuda else 0,
        pin_memory=pin_mem if use_cuda else False,
        collate_fn=mydata.list_collate,
    )

    log.debug('Running network')
    tot_loss = []
    coord_loss = []
    conf_loss = []
    cls_loss = []
    anno, det = {}, {}
    num_det = 0

    for idx, (data, box) in enumerate(loader):
        print("sssssssssssssize {}".format(np.shape(data)))
        if (idx + 1) % 20 == 0:
            log.info('%d/%d' % (idx + 1, len(loader)))
        if use_cuda:
            data = data.cuda()
        with torch.no_grad():
            output, loss = net(data, box)
        print("output:::::::::::::::::::::::::::",output)
        key_val = len(anno)
        anno.update({loader.dataset.keys[key_val + k]: v for k, v in enumerate(box)})
        det.update({loader.dataset.keys[key_val + k]: v for k, v in enumerate(output)})
        # print("++++++++++++++++++++++++++")
        print("predict img:", idx + 1)
        # print(box)
        # print(det)
        netw, neth = network_size
        reorg_dets = voc_wrapper.reorgDetection(det, netw, neth)  # , prefix)
        result = voc_wrapper.genResults(reorg_dets, results, nms_thresh)
        print("------------------>>> detect result:", result)
        det = {}
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OneDet: an one stage framework based on PyTorch')
    parser.add_argument('model_name', help='model name: TinyYolov3, Yolov3', default="TinyYolov3")
    args = parser.parse_args()

    train_flag = 2
    config = initEnv(train_flag=train_flag, model_name=args.model_name)

    log.info('Config\n\n%s\n' % pformat(config))

    # init env
    hyper_params = HyperParams(config, train_flag=train_flag)

    # init and run eng
    result = VOCTest(hyper_params)
