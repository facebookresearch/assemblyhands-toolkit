# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import os.path as osp
import numpy as np
from src.common.utils.dir import add_pypath, make_folder


class BaseConfig(object):
    def __init__(self):
        """
        Base config for model and heatmap generation
        """
        ## input, output
        self.input_img_shape = (256, 256)
        self.output_hm_shape = (64, 64, 64)  # (depth, height, width)
        self.sigma = 2.5
        self.bbox_3d_size = 400  # depth axis
        self.bbox_3d_size_root = 400  # depth axis
        self.output_root_hm_shape = 64  # depth axis

        ## model
        self.resnet_type = 50  # 18, 34, 50, 101, 152

    def print_config(self):
        """
        Print configuration
        """
        print(">>> Configuration:")
        for key, value in self.__dict__.items():
            print(f"{key}: {value}")


class Config(BaseConfig):
    def __init__(self, dataset="InterHand2.6M"):
        super(Config, self).__init__()
        ## dataset
        self.dataset = dataset

        ## training config
        if dataset in ("InterHand2.6M", "AssemblyHands-Ego"):
            self.lr_dec_epoch = [15, 17]
        else:
            self.lr_dec_epoch = [45, 47]

        if dataset in ("InterHand2.6M", "AssemblyHands-Ego"):
            self.end_epoch = 20
        else:
            self.end_epoch = 50
        self.lr = 1e-4
        self.lr_dec_factor = 10
        self.train_batch_size = 8  # 16

        ## testing config
        self.test_batch_size = 8  # 32
        if dataset == "InterHand2.6M":
            self.trans_test = "gt"  # gt, rootnet
            self.bbox_scale = 1.25
        elif dataset == "AssemblyHands-Ego":
            self.trans_test = "gt"
            self.bbox_scale = 1.75
        else:
            self.trans_test = "gt"

        ## directory
        self.cur_dir = osp.dirname(os.path.abspath(__file__))
        assert self.cur_dir.endswith("main"), f"cur_dir: {self.cur_dir}"
        self.root_dir = self.cur_dir.rsplit("/src", 1)[0]
        assert self.root_dir.endswith(
            "assemblyhands-toolkit"
        ), f"root_dir: {self.root_dir}"
        self.src_dir = osp.join(self.root_dir, "src")
        self.data_dir = osp.join(
            self.root_dir, "data", dataset.lower().rsplit("-", 1)[0]
        )
        self.dataset_dir = osp.join(self.src_dir, "dataset")
        self.output_dir = osp.join(self.root_dir, f"output/{dataset.lower()}")
        self.model_dir = osp.join(self.output_dir, "model_dump")
        self.vis_dir = osp.join(self.output_dir, "vis")
        self.log_dir = osp.join(self.output_dir, "log")
        self.result_dir = osp.join(self.output_dir, "result")
        print(">>> data_dir: {}".format(self.data_dir))
        print(">>> output_dir: {}".format(self.output_dir))

        ## others
        self.num_thread = 4  # 40
        self.gpu_ids = "0"
        self.num_gpus = 1
        self.continue_train = False
        self.print_freq = 500

        # set pathes
        add_pypath(osp.join(self.dataset_dir))
        print(">>> dataset_dir: {}/{}".format(self.dataset_dir, self.dataset))
        add_pypath(osp.join(self.dataset_dir, self.dataset))
        make_folder(self.model_dir)
        make_folder(self.vis_dir)
        make_folder(self.log_dir)
        make_folder(self.result_dir)

    def set_args(self, gpu_ids, continue_train=False):
        self.gpu_ids = gpu_ids
        self.num_gpus = len(self.gpu_ids.split(","))
        self.continue_train = continue_train
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_ids
        print(">>> Using GPU: {}".format(self.gpu_ids))


# cfg = Config()
base_cfg = BaseConfig()
