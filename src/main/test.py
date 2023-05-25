# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
from tqdm import tqdm
import numpy as np
import cv2
import torch
from src.main.config import Config
import torch.backends.cudnn as cudnn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, dest="gpu_ids", default="0")
    parser.add_argument("--test_epoch", type=str, dest="test_epoch", default="20")
    parser.add_argument("--test_set", type=str, dest="test_set")
    parser.add_argument("--vis", action="store_true")
    parser.add_argument("--vis_freq", type=int, dest="vis_freq", default=10)
    parser.add_argument("--dataset", type=str, dest="dataset", default="InterHand2.6M")
    args = parser.parse_args()

    if "-" in args.gpu_ids:
        gpus = args.gpu_ids.split("-")
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ",".join(map(lambda x: str(x), list(range(*gpus))))

    return args


def main(cfg, args):

    if cfg.dataset == "InterHand2.6M" or cfg.dataset.startswith("AssemblyHands"):
        assert args.test_set, "Test set is required. Select one of test/val"
    else:
        args.test_set = "test"
    checkpoint_dir = "checkpoints"
    tester = Tester(cfg, args.test_epoch)
    tester._make_batch_generator(args.test_set)
    tester._make_model(checkpoint_dir)

    preds = {"joint_coord": [], "rel_root_depth": [], "hand_type": [], "inv_trans": []}
    with torch.no_grad():
        for itr, (inputs, targets, meta_info) in enumerate(
            tqdm(tester.batch_generator)
        ):

            # forward
            out = tester.model(inputs, targets, meta_info, "test")

            joint_coord_out = out["joint_coord"].cpu().numpy()
            rel_root_depth_out = out["rel_root_depth"].cpu().numpy()
            hand_type_out = out["hand_type"].cpu().numpy()
            inv_trans = out["inv_trans"].cpu().numpy()

            preds["joint_coord"].append(joint_coord_out)
            preds["rel_root_depth"].append(rel_root_depth_out)
            preds["hand_type"].append(hand_type_out)
            preds["inv_trans"].append(inv_trans)

            if args.vis and itr % args.vis_freq == 0:
                raise NotImplementedError()

    # evaluate
    preds = {k: np.concatenate(v) for k, v in preds.items()}
    tester._evaluate(preds)


if __name__ == "__main__":
    args = parse_args()
    cfg = Config(dataset=args.dataset)
    cfg.set_args(args.gpu_ids)
    cudnn.benchmark = True

    from src.common.base import Tester
    from src.common.utils.vis import vis_keypoints
    from src.common.utils.transforms import flip

    main(cfg, args)
