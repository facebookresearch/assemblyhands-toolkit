# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import os.path as osp
import math
import time
import glob
import abc
from torch.utils.data import DataLoader
import torch.optim
import torchvision.transforms as transforms

# from src import cfg
from dataset import Dataset
from src.common.timer import Timer
from src.common.logger import colorlogger
from torch.nn.parallel.data_parallel import DataParallel
from src.main.model import get_model


class Base(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, cfg, log_name="logs.txt"):
        self.cur_epoch = 0
        self.cfg = cfg
        # timer
        self.tot_timer = Timer()
        self.gpu_timer = Timer()
        self.read_timer = Timer()

        # logger
        self.logger = colorlogger(self.cfg.log_dir, log_name=log_name)

    @abc.abstractmethod
    def _make_batch_generator(self):
        return

    @abc.abstractmethod
    def _make_model(self):
        return


class Trainer(Base):
    def __init__(self, cfg):
        super(Trainer, self).__init__(cfg, log_name="train_logs.txt")

    def get_optimizer(self, model):
        optimizer = torch.optim.Adam(model.parameters(), lr=self.cfg.lr)
        return optimizer

    def set_lr(self, epoch):
        if len(self.cfg.lr_dec_epoch) == 0:
            return self.cfg.lr

        for e in self.cfg.lr_dec_epoch:
            if epoch < e:
                break
        if epoch < self.cfg.lr_dec_epoch[-1]:
            idx = self.cfg.lr_dec_epoch.index(e)
            for g in self.optimizer.param_groups:
                g["lr"] = self.cfg.lr / (self.cfg.lr_dec_factor**idx)
        else:
            for g in self.optimizer.param_groups:
                g["lr"] = self.cfg.lr / (
                    self.cfg.lr_dec_factor ** len(self.cfg.lr_dec_epoch)
                )

    def get_lr(self):
        for g in self.optimizer.param_groups:
            cur_lr = g["lr"]

        return cur_lr

    def _make_batch_generator(self):
        # data load and construct batch generator
        self.logger.info("Creating train dataset...")
        trainset_loader = Dataset(transforms.ToTensor(), self.cfg, "train")
        batch_generator = DataLoader(
            dataset=trainset_loader,
            batch_size=self.cfg.num_gpus * self.cfg.train_batch_size,
            shuffle=True,
            num_workers=self.cfg.num_thread,
            pin_memory=True,
        )

        self.joint_num = trainset_loader.joint_num
        self.itr_per_epoch = math.ceil(
            trainset_loader.__len__() / self.cfg.num_gpus / self.cfg.train_batch_size
        )
        self.batch_generator = batch_generator
        self.trainset = trainset_loader

    def _make_model(self, resume_epoch=None):
        # prepare network
        self.logger.info("Creating graph and optimizer...")
        model = get_model("train", self.joint_num)
        model = DataParallel(model).cuda()
        optimizer = self.get_optimizer(model)
        if self.cfg.continue_train:
            start_epoch, model, optimizer = self.load_model(
                model, optimizer, resume_epoch
            )
        else:
            start_epoch = 0
        model.train()

        self.start_epoch = start_epoch
        self.model = model
        self.optimizer = optimizer

    def save_model(self, state, epoch):
        file_path = osp.join(
            self.cfg.model_dir, "snapshot_{:02d}.pth.tar".format(int(epoch))
        )
        torch.save(state, file_path)
        self.logger.info("Write snapshot into {}".format(file_path))

    def load_model(self, model, optimizer, resume_epoch=None):
        model_file_list = glob.glob(osp.join(self.cfg.model_dir, "*.pth.tar"))
        if resume_epoch is not None:
            cur_epoch = resume_epoch
        else:
            cur_epoch = max(
                [
                    int(
                        file_name[
                            file_name.find("snapshot_") + 9 : file_name.find(".pth.tar")
                        ]
                    )
                    for file_name in model_file_list
                ]
            )
        model_path = osp.join(
            self.cfg.model_dir, "snapshot_{:02d}.pth.tar".format(int(cur_epoch))
        )
        self.logger.info("Load checkpoint from {}".format(model_path))
        ckpt = torch.load(model_path)
        start_epoch = ckpt["epoch"] + 1

        model.load_state_dict(ckpt["network"])
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except:
            pass

        return start_epoch, model, optimizer


class Tester(Base):
    def __init__(self, cfg, test_epoch):
        super(Tester, self).__init__(cfg, log_name="test_logs.txt")
        self.test_epoch = int(test_epoch)

    def _make_batch_generator(self, test_set, shuffle=False):
        # data load and construct batch generator
        self.logger.info("Creating " + test_set + " dataset...")
        testset_loader = Dataset(transforms.ToTensor(), self.cfg, test_set)
        batch_generator = DataLoader(
            dataset=testset_loader,
            batch_size=self.cfg.num_gpus * self.cfg.test_batch_size,
            shuffle=shuffle,
            num_workers=self.cfg.num_thread,
            pin_memory=True,
        )

        self.joint_num = testset_loader.joint_num
        self.batch_generator = batch_generator
        self.testset = testset_loader

    def _make_model(self, checkpoint_dir=None):
        if checkpoint_dir is not None:
            model_path = os.path.join(
                checkpoint_dir, "snapshot_%02d.pth.tar" % self.test_epoch
            )
            if not os.path.exists(model_path):
                model_path = os.path.join(
                    self.cfg.model_dir, "snapshot_%02d.pth.tar" % self.test_epoch
                )
        else:
            model_path = os.path.join(
                self.cfg.model_dir, "snapshot_%02d.pth.tar" % self.test_epoch
            )
        # assert os.path.exists(model_path), 'Cannot find model at ' + model_path
        # prepare network
        self.logger.info("Creating graph...")
        model = get_model("test", self.joint_num)
        model = DataParallel(model).cuda()

        if os.path.exists(model_path):
            self.logger.info("Load checkpoint from {}".format(model_path))
            ckpt = torch.load(model_path)
            model.load_state_dict(ckpt["network"])
        else:
            self.logger.info("Load checkpoint not found {}".format(model_path))

        model.eval()
        self.model = model

    def _evaluate(self, preds):
        self.testset.evaluate(preds)
