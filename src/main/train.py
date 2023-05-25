# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse

# from config import cfg
from src.main.config import Config
import numpy as np
import torch
import torch.backends.cudnn as cudnn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, dest="gpu_ids", default="0")
    parser.add_argument("--resume_epoch", type=int, default=None)
    parser.add_argument("--continue", dest="continue_train", action="store_true")
    parser.add_argument("--dataset", type=str, dest="dataset", default="InterHand2.6M")
    parser.add_argument("--running_eval", dest="running_eval", action="store_true")
    args = parser.parse_args()

    if "-" in args.gpu_ids:
        gpus = args.gpu_ids.split("-")
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ",".join(map(lambda x: str(x), list(range(*gpus))))

    return args


def main(cfg, args):

    trainer = Trainer(cfg)
    trainer._make_batch_generator()
    trainer._make_model(args.resume_epoch)

    # train
    for epoch in range(trainer.start_epoch, cfg.end_epoch):

        trainer.set_lr(epoch)
        trainer.tot_timer.tic()
        trainer.read_timer.tic()
        preds = {
            "joint_coord": [],
            "rel_root_depth": [],
            "hand_type": [],
            "inv_trans": [],
        }
        for itr, (inputs, targets, meta_info) in enumerate(trainer.batch_generator):
            trainer.read_timer.toc()
            trainer.gpu_timer.tic()

            # forward
            trainer.optimizer.zero_grad()
            if args.running_eval:
                loss, out = trainer.model(
                    inputs, targets, meta_info, "train_with_preds"
                )
            else:
                loss = trainer.model(inputs, targets, meta_info, "train")
            loss = {k: loss[k].mean() for k in loss}

            # backward
            sum(loss[k] for k in loss).backward()
            trainer.optimizer.step()
            trainer.gpu_timer.toc()
            if itr % cfg.print_freq == 0:
                screen = [
                    "Epoch %02d/%02d itr %05d/%05d:"
                    % (epoch, cfg.end_epoch, itr, trainer.itr_per_epoch),
                    "lr: %g" % (trainer.get_lr()),
                    "speed: %.2f(%.2fs r%.2f)s/itr"
                    % (
                        trainer.tot_timer.average_time,
                        trainer.gpu_timer.average_time,
                        trainer.read_timer.average_time,
                    ),
                    "%.2fh/epoch"
                    % (trainer.tot_timer.average_time / 3600.0 * trainer.itr_per_epoch),
                ]
                screen += [
                    "%s: %.4f" % ("loss_" + k, v.detach())
                    for k, v in loss.items()
                    if v > 0
                ]
                trainer.logger.info(" ".join(screen))

            trainer.tot_timer.toc()
            trainer.tot_timer.tic()
            trainer.read_timer.tic()

            if args.running_eval:
                # set preds for evaluation
                joint_coord_out = out["joint_coord"].detach().cpu().numpy()
                rel_root_depth_out = out["rel_root_depth"].detach().cpu().numpy()
                hand_type_out = out["hand_type"].detach().cpu().numpy()
                inv_trans = out["inv_trans"].detach().cpu().numpy()

                preds["joint_coord"].append(joint_coord_out)
                preds["rel_root_depth"].append(rel_root_depth_out)
                preds["hand_type"].append(hand_type_out)
                preds["inv_trans"].append(inv_trans)

        # save model
        trainer.save_model(
            {
                "epoch": epoch,
                "network": trainer.model.state_dict(),
                "optimizer": trainer.optimizer.state_dict(),
            },
            epoch,
        )

        # evaluate
        if args.running_eval:
            preds = {k: np.concatenate(v) for k, v in preds.items()}
            trainer.trainset.evaluate(preds, cfg)


if __name__ == "__main__":
    # argument parse and create log
    args = parse_args()
    cfg = Config(dataset=args.dataset)
    cfg.set_args(args.gpu_ids, args.continue_train)
    cudnn.benchmark = True

    from src.common.base import Trainer

    main(cfg, args)
