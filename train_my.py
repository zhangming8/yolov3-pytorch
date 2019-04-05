import os
import argparse
import time
from pprint import pformat
import logging as log
import torch
from torchvision import transforms as tf
from statistics import mean
import sys
sys.path.append(".")

from utils.envs import initEnv
import data as mydata
import models
from engine import engine


class HyperParams(object):
    def __init__(self, config, train_flag=1):

        self.cuda = True
        self.labels = config['labels']
        self.classes = len(self.labels)
        self.data_root = config['data_root_dir']
        self.model_name = config['model_name']
        self.anchors = config["anchor"]

        # cuda check
        if self.cuda:
            if not torch.cuda.is_available():
                log.debug('CUDA not available')
                self.cuda = False
            else:
                log.debug('CUDA enabled')

        if train_flag == 1:
            cur_cfg = config

            self.gpus = cur_cfg["gpus"]
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


class VOCDataset(mydata.BramboxDataset):
    def __init__(self, hyper_params):
        anno = hyper_params.trainfile
        root = hyper_params.data_root
        flip = hyper_params.flip
        jitter = hyper_params.jitter
        hue, sat, val = hyper_params.hue, hyper_params.sat, hyper_params.val
        network_size = hyper_params.network_size
        labels = hyper_params.labels

        rf = mydata.transform.RandomFlip(flip)
        rc = mydata.transform.RandomCropLetterbox(self, jitter)
        hsv = mydata.transform.HSVShift(hue, sat, val)
        rot = mydata.transform.RandomRotate(jitter_min=-20, jitter_max=20)
        it = tf.ToTensor()

        img_tf = mydata.transform.Compose([rot, rc, rf, hsv, it])
        anno_tf = mydata.transform.Compose([rot, rc, rf])

        def identify(img_id):
            # return f'{root}/VOCdevkit/{img_id}.jpg'
            return f'{img_id}'

        super(VOCDataset, self).__init__('anno_pickle', anno, network_size, labels, identify, img_tf, anno_tf)


class VOCTrainingEngine(engine.Engine):
    """ This is a custom engine for this training cycle """

    def __init__(self, hyper_params):
        self.hyper_params = hyper_params
        # all in args
        self.batch_size = hyper_params.batch
        self.mini_batch_size = hyper_params.mini_batch
        self.max_batches = hyper_params.max_batches

        self.classes = hyper_params.classes

        self.cuda = hyper_params.cuda
        self.gpus = hyper_params.gpus
        self.backup_dir = hyper_params.backup_dir

        log.debug('Creating network')
        model_name = hyper_params.model_name
        # net = models.__dict__[model_name](hyper_params.classes, hyper_params.weights, train_flag=1,
        #                                   clear=hyper_params.clear)
        if model_name == "TinyYolov3":
            net = models.TinyYolov3(hyper_params.classes, hyper_params.weights,anchors=hyper_params.anchors, train_flag=1, clear=hyper_params.clear)
        elif model_name == "Yolov3":
            net = models.Yolov3(hyper_params.classes, hyper_params.weights, train_flag=1, clear=hyper_params.clear)
        else:
            print("model name should be 'TinyYolov3' or 'Yolov3', your input {}".format(model_name))
            exit()
        log.info('Net structure\n\n%s\n' % net)
        if self.cuda:
            net.cuda()
            net = torch.nn.DataParallel(net, device_ids=[i for i in range(len(self.gpus.split(",")))])  # multi-GPU

        log.debug('Creating optimizer')
        learning_rate = hyper_params.learning_rate
        momentum = hyper_params.momentum
        decay = hyper_params.decay
        batch = hyper_params.batch
        log.info(f'Adjusting learning rate to [{learning_rate}]')
        optim = torch.optim.SGD(net.parameters(), lr=learning_rate / batch, momentum=momentum, dampening=0,
                                weight_decay=decay * batch)

        log.debug('Creating dataloader')
        dataset = VOCDataset(hyper_params)
        dataloader = mydata.DataLoader(
            dataset,
            batch_size=self.mini_batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=hyper_params.nworkers if self.cuda else 0,
            pin_memory=hyper_params.pin_mem if self.cuda else False,
            collate_fn=mydata.list_collate,
        )

        super(VOCTrainingEngine, self).__init__(net, optim, dataloader)

        self.nloss = self.network.nloss

        self.train_loss = [{'tot': [], 'coord': [], 'conf': [], 'cls': []} for _ in range(self.nloss)]

    def start(self):
        log.debug('Creating additional logging objects')
        hyper_params = self.hyper_params

        lr_steps = hyper_params.lr_steps
        lr_rates = hyper_params.lr_rates

        bp_steps = hyper_params.bp_steps
        bp_rates = hyper_params.bp_rates
        backup = hyper_params.backup

        rs_steps = hyper_params.rs_steps
        rs_rates = hyper_params.rs_rates
        resize = hyper_params.resize

        self.add_rate('learning_rate', lr_steps, [lr / self.batch_size for lr in lr_rates])
        self.add_rate('backup_rate', bp_steps, bp_rates, backup)
        self.add_rate('resize_rate', rs_steps, rs_rates, resize)

        self.dataloader.change_input_dim()

    def process_batch(self, data):
        data, target = data
        # to(device)
        if self.cuda:
            data = data.cuda()
        # mydata = torch.autograd.Variable(mydata, requires_grad=True)

        loss = self.network(data, target)
        loss.backward()

        for ii in range(self.nloss):
            self.train_loss[ii]['tot'].append(self.network.loss[ii].loss_tot.item() / self.mini_batch_size)
            self.train_loss[ii]['coord'].append(self.network.loss[ii].loss_coord.item() / self.mini_batch_size)
            self.train_loss[ii]['conf'].append(self.network.loss[ii].loss_conf.item() / self.mini_batch_size)
            if self.network.loss[ii].loss_cls is not None:
                self.train_loss[ii]['cls'].append(self.network.loss[ii].loss_cls.item() / self.mini_batch_size)

    def train_batch(self):
        self.optimizer.step()
        self.optimizer.zero_grad()

        all_tot = 0.0
        all_coord = 0.0
        all_conf = 0.0
        all_cls = 0.0
        for ii in range(self.nloss):
            tot = mean(self.train_loss[ii]['tot'])
            coord = mean(self.train_loss[ii]['coord'])
            conf = mean(self.train_loss[ii]['conf'])
            all_tot += tot
            all_coord += coord
            all_conf += conf
            #if self.classes > 1:
            if True:
                cls = mean(self.train_loss[ii]['cls'])
                all_cls += cls

            #if self.classes > 1:
            if True:
                log.info(
                    f'{self.batch} # {ii}: Loss:{round(tot, 5)} (Coord:{round(coord, 2)} Conf:{round(conf, 2)} Cls:{round(cls, 2)})')
            else:
                log.info(f'{self.batch} # {ii}: Loss:{round(tot, 5)} (Coord:{round(coord, 2)} Conf:{round(conf, 2)})')

        #if self.classes > 1:
        if True:
            log.info(
                f'{self.batch} # All : Loss:{round(all_tot, 5)} (Coord:{round(all_coord, 2)} Conf:{round(all_conf, 2)} Cls:{round(all_cls, 2)})')
        else:
            log.info(
                f'{self.batch} # All : Loss:{round(all_tot, 5)} (Coord:{round(all_coord, 2)} Conf:{round(all_conf, 2)})')
        self.train_loss = [{'tot': [], 'coord': [], 'conf': [], 'cls': []} for _ in range(self.nloss)]
        if self.batch % self.backup_rate == 0:
            self.network.save_weights(os.path.join(self.backup_dir, f'weights_{self.batch}.pt'))

        if self.batch % 100 == 0:
            self.network.save_weights(os.path.join(self.backup_dir, f'backup.pt'))

        if self.batch % self.resize_rate == 0:
            if self.batch + 200 >= self.max_batches:
                finish_flag = True
            else:
                finish_flag = False
            self.dataloader.change_input_dim(finish=finish_flag)

    def quit(self):
        if self.sigint:
            self.network.save_weights(os.path.join(self.backup_dir, f'backup.pt'))
            return True
        elif self.batch >= self.max_batches:
            self.network.save_weights(os.path.join(self.backup_dir, f'final.dw'))
            return True
        else:
            return False


if __name__ == '__main__':

    train_flag = 1  # 1 for train, 2 for test, 3 for test speed
    model_name = "TinyYolov3"
    #model_name = "Yolov3"
    config = initEnv(train_flag=train_flag, model_name=model_name)

    log.info('Config\n\n%s\n' % pformat(config))

    # init env
    hyper_params = HyperParams(config, train_flag=train_flag)

    # int eng
    eng = VOCTrainingEngine(hyper_params)

    # run eng
    b1 = eng.batch
    t1 = time.time()
    eng()
    t2 = time.time()
    b2 = eng.batch

    log.info(f'\nDuration of {b2-b1} batches: {t2-t1} seconds [{round((t2-t1)/(b2-b1), 3)} sec/batch]')
