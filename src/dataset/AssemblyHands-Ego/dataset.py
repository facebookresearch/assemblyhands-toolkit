# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import torch
import torch.utils.data
import cv2
import os
import os.path as osp
import random
from copy import deepcopy
from src.common.utils.preprocessing import (
    load_img,
    load_crop_img,
    update_params_after_crop,
    load_skeleton,
    get_bbox,
    process_bbox,
    augmentation,
    transform_input_to_output_space,
    trans_point2d,
)
from src.common.utils.transforms import cam2pixel, pixel2cam, Camera
from src.common.utils.transforms import world2cam_assemblyhands as world2cam
from src.common.utils.transforms import cam2world_assemblyhands as cam2world
from src.common.utils.vis import vis_keypoints, vis_3d_keypoints
from copy import deepcopy
import json
from pycocotools.coco import COCO

ANNOT_VERSION = "v1-1"
IS_DEBUG = True
# IS_DEBUG = False
N_DEBUG_SAMPLES = 200


class Dataset(torch.utils.data.Dataset):
    def __init__(self, transform, cfg, mode):
        self.mode = mode  # train, test, val
        self.img_path = "data/assemblyhands/images"
        self.annot_path = "data/assemblyhands/annotations"
        self.modality = "ego"
        self.transform = transform
        self.cfg = cfg
        self.joint_num = 21  # single hand
        self.root_joint_idx = {"right": 20, "left": 41}
        self.joint_type = {
            "right": np.arange(0, self.joint_num),
            "left": np.arange(self.joint_num, self.joint_num * 2),
        }
        self.skeleton = load_skeleton(
            osp.join(self.annot_path, "skeleton.txt"), self.joint_num * 2
        )

        self.datalist = []
        self.datalist_sh = []
        self.datalist_ih = []
        self.sequence_names = []
        n_skip = 0
        # load annotation
        print(f"Load annotation from  {self.annot_path}, mode: {self.mode}")
        data_mode = self.mode
        if IS_DEBUG and self.mode.startswith("train"):
            print(">>> DEBUG MODE: Loading val data during training")
            data_mode = "val"
        self.invalid_data_file = os.path.join(
            self.annot_path, data_mode, f"invalid_{data_mode}_{self.modality}.txt"
        )
        db = COCO(
            osp.join(
                self.annot_path,
                data_mode,
                "assemblyhands_"
                + data_mode
                + f"_{self.modality}_data_{ANNOT_VERSION}.json",
            )
        )
        with open(
            osp.join(
                self.annot_path,
                data_mode,
                "assemblyhands_"
                + data_mode
                + f"_{self.modality}_calib_{ANNOT_VERSION}.json",
            )
        ) as f:
            cameras = json.load(f)["calibration"]
        with open(
            osp.join(
                self.annot_path,
                data_mode,
                "assemblyhands_" + data_mode + f"_joint_3d_{ANNOT_VERSION}.json",
            )
        ) as f:
            joints = json.load(f)["annotations"]

        print("Get bbox and root depth from groundtruth annotation")
        invalid_data_list = None
        if osp.exists(self.invalid_data_file):
            with open(self.invalid_data_file) as f:
                lines = f.readlines()
            if len(lines) > 0:
                invalid_data_list = [line.strip() for line in lines]
        else:
            print(
                "Invalid data file does not exist. Checking the validity of generated crops"
            )
            f = open(self.invalid_data_file, "w")

        annot_list = db.anns.keys()
        for i, aid in enumerate(annot_list):
            ann = db.anns[aid]
            image_id = ann["image_id"]
            img = db.loadImgs(image_id)[0]

            seq_name = str(img["seq_name"])
            camera_name = img["camera"]
            frame_idx = img["frame_idx"]
            file_name = img["file_name"]
            img_path = osp.join(self.img_path, file_name)
            assert osp.exists(img_path), f"Image path {img_path} does not exist"

            K = np.array(
                cameras[seq_name]["intrinsics"][camera_name + "_mono10bit"],
                dtype=np.float32,
            )
            Rt = np.array(
                cameras[seq_name]["extrinsics"][f"{frame_idx:06d}"][
                    camera_name + "_mono10bit"
                ],
                dtype=np.float32,
            )
            retval_camera = Camera(K, Rt, dist=None, name=camera_name)
            campos, camrot, focal, princpt = retval_camera.get_params()

            joint_world = np.array(
                joints[seq_name][f"{frame_idx:06d}"]["world_coord"], dtype=np.float32
            )
            joint_cam = world2cam(joint_world, camrot, campos)
            joint_img = cam2pixel(joint_cam, focal, princpt)[:, :2]

            joint_valid = np.array(ann["joint_valid"], dtype=np.float32).reshape(
                self.joint_num * 2
            )
            # if root is not valid -> root-relative 3D pose is also not valid. Therefore, mark all joints as invalid
            # joint_valid[self.joint_type['right']] *= joint_valid[self.root_joint_idx['right']]
            # joint_valid[self.joint_type['left']] *= joint_valid[self.root_joint_idx['left']]

            abs_depth = {
                "right": joint_cam[self.root_joint_idx["right"], 2],
                "left": joint_cam[self.root_joint_idx["left"], 2],
            }
            cam_param = {"focal": focal, "princpt": princpt}
            for hand_id, hand_type in enumerate(["right", "left"]):
                if ann["bbox"][hand_type] is None:
                    continue
                hand_type_valid = np.ones(1, dtype=np.float32)

                img_width, img_height = img["width"], img["height"]
                bbox = np.array(ann["bbox"][hand_type], dtype=np.float32)  # x,y,x,y
                x0, y0, x1, y1 = bbox
                original_bbox = [x0, y0, x1 - x0, y1 - y0]  # x,y,w,h
                bbox = process_bbox(
                    original_bbox, (img_height, img_width), scale=1.75
                )  # bbox = original_bbox

                joint_valid_single_hand = deepcopy(joint_valid)
                inv_hand_id = abs(1 - hand_id)
                # make invlid for the other hand
                joint_valid_single_hand[
                    inv_hand_id * self.joint_num : (inv_hand_id + 1) * self.joint_num
                ] = 0
                if invalid_data_list is not None:
                    crop_name = f"{file_name},{hand_id}"
                    if crop_name in invalid_data_list:  # skip registred invalid samples
                        n_skip += 1
                        continue
                else:  # first run to check the validity of generated crops
                    if sum(joint_valid_single_hand) < 10:
                        n_skip += 1
                        f.write(f"{file_name},{hand_id}\n")
                        continue
                    try:
                        load_crop_img(
                            img_path,
                            bbox,
                            joint_img.copy(),
                            joint_world.copy(),
                            joint_valid_single_hand.copy(),
                            deepcopy(retval_camera),
                        )
                    except:
                        n_skip += 1
                        f.write(f"{file_name},{hand_id}\n")
                        continue

                joint = {
                    "cam_coord": joint_cam,
                    "img_coord": joint_img,
                    "world_coord": joint_world,
                    "valid": joint_valid_single_hand,
                }  # joint_valid}
                data = {
                    "img_path": img_path,
                    "seq_name": seq_name,
                    "cam_param": cam_param,
                    "bbox": bbox,
                    "original_bbox": original_bbox,
                    "joint": joint,
                    "hand_type": hand_type,
                    "hand_type_valid": hand_type_valid,
                    "abs_depth": abs_depth,
                    "file_name": img["file_name"],
                    "seq_name": seq_name,
                    "cam": camera_name,
                    "frame": frame_idx,
                    "retval_camera": retval_camera,
                }
                if hand_type == "right" or hand_type == "left":
                    self.datalist_sh.append(data)
                else:
                    self.datalist_ih.append(data)
                if seq_name not in self.sequence_names:
                    self.sequence_names.append(seq_name)

            if IS_DEBUG and i >= N_DEBUG_SAMPLES - 1:
                print(">>> DEBUG MODE: Loaded %d samples" % N_DEBUG_SAMPLES)
                break
        self.datalist = self.datalist_sh + self.datalist_ih
        assert len(self.datalist) > 0, "No data found."
        if not osp.exists(self.invalid_data_file):
            f.close()
        print(
            "Number of annotations in single hand sequences: "
            + str(len(self.datalist_sh))
        )
        print(
            "Number of annotations in interacting hand sequences: "
            + str(len(self.datalist_ih))
        )
        print("Number of skipped annotations: " + str(n_skip))

    def handtype_str2array(self, hand_type):
        if hand_type == "right":
            return np.array([1, 0], dtype=np.float32)
        elif hand_type == "left":
            return np.array([0, 1], dtype=np.float32)
        elif hand_type == "interacting":
            return np.array([1, 1], dtype=np.float32)
        else:
            assert 0, print("Not supported hand type: " + hand_type)

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        data = self.datalist[idx]
        img_path, bbox, joint, hand_type, hand_type_valid = (
            data["img_path"],
            data["bbox"],
            data["joint"],
            data["hand_type"],
            data["hand_type_valid"],
        )
        joint_world = joint["world_coord"]
        joint_img = joint["img_coord"].copy()
        joint_valid = joint["valid"].copy()

        hand_type = self.handtype_str2array(hand_type)
        # image load # img = load_img(img_path, bbox)
        img, bbox, joint_img, joint_cam, joint_valid, retval_camera = load_crop_img(
            img_path,
            bbox,
            joint_img,
            joint_world,
            joint_valid,
            deepcopy(data["retval_camera"]),
        )
        joint_coord = np.concatenate((joint_img, joint_cam[:, 2, None]), 1)

        # augmentation
        img, joint_coord, joint_valid, hand_type, inv_trans = augmentation(
            img,
            bbox,
            joint_coord,
            joint_valid,
            hand_type,
            self.mode,
            self.joint_type,
            no_aug=True,
        )
        rel_root_depth = np.array(
            [
                joint_coord[self.root_joint_idx["left"], 2]
                - joint_coord[self.root_joint_idx["right"], 2]
            ],
            dtype=np.float32,
        ).reshape(1)
        root_valid = (
            np.array(
                [
                    joint_valid[self.root_joint_idx["right"]]
                    * joint_valid[self.root_joint_idx["left"]]
                ],
                dtype=np.float32,
            ).reshape(1)
            if hand_type[0] * hand_type[1] == 1
            else np.zeros((1), dtype=np.float32)
        )
        # transform to output heatmap space
        (
            joint_coord_hm,
            joint_valid,
            rel_root_depth,
            root_valid,
        ) = transform_input_to_output_space(
            joint_coord.copy(),
            joint_valid,
            rel_root_depth,
            root_valid,
            self.root_joint_idx,
            self.joint_type,
        )
        img = self.transform(img.astype(np.float32)) / 255.0

        # update camera parameters after resize to cfg.input_img_shape
        retval_camera.update_after_resize((bbox[3], bbox[2]), self.cfg.input_img_shape)
        campos, camrot, focal, princpt = retval_camera.get_params()
        cam_param = {"focal": focal, "princpt": princpt}

        inputs = {"img": img, "idx": idx}
        targets = {
            "joint_coord": joint_coord_hm,
            "_joint_coord": joint_coord,
            "rel_root_depth": rel_root_depth,
            "hand_type": hand_type,
        }
        meta_info = {
            "joint_valid": joint_valid,
            "root_valid": root_valid,
            "hand_type_valid": hand_type_valid,
            "inv_trans": inv_trans,
            "seq_name": data["seq_name"],
            "cam": data["cam"],
            "frame": int(data["frame"]),
            "cam_param_updated": cam_param,
        }
        return inputs, targets, meta_info

    def view_samples(self, max_n_vis=10, is_crop=True):
        print("\nVisualize GT...")
        gts = self.datalist
        sample_num = len(gts)
        random.seed(0)
        random_samples = random.sample(range(sample_num), max_n_vis)
        for i in random_samples:
            data = gts[i]
            img_path, joint, gt_hand_type = (
                data["img_path"],
                data["joint"],
                data["hand_type"],
            )
            frame_id = int(img_path.split("/")[-1].split(".")[0])
            gt_joint_coord_img = joint["img_coord"].copy()  # original image
            gt_joint_coord_cam = joint["cam_coord"].copy()
            joint_valid = joint["valid"]

            if is_crop:
                # read cropped image
                inputs, targets, meta_info = self.__getitem__(i)
                img = inputs["img"]
                _img = img.numpy() * 255.0

                gt_joint_coord_img = targets["joint_coord"].copy()  # heatmap coord
                gt_joint_coord_img[:, 0] = (
                    gt_joint_coord_img[:, 0]
                    / self.cfg.output_hm_shape[2]
                    * self.cfg.input_img_shape[1]
                )
                gt_joint_coord_img[:, 1] = (
                    gt_joint_coord_img[:, 1]
                    / self.cfg.output_hm_shape[1]
                    * self.cfg.input_img_shape[0]
                )
                # restore depth to original camera space
                gt_joint_coord_img[:, 2] = (
                    gt_joint_coord_img[:, 2] / self.cfg.output_hm_shape[0] * 2 - 1
                ) * (self.cfg.bbox_3d_size / 2)
                gt_joint_coord_img[self.joint_type["right"], 2] += data["abs_depth"][
                    "right"
                ]
                gt_joint_coord_img[self.joint_type["left"], 2] += data["abs_depth"][
                    "left"
                ]
            else:
                # read full image
                _img = cv2.imread(img_path)
                _img = _img.transpose(2, 0, 1)
            _img = _img.astype(np.uint8)
            vis_kps = gt_joint_coord_img.copy()
            vis_valid = joint_valid.copy()
            filename = f"vis_{frame_id:04d}_{gt_hand_type}_2d.jpg"
            vis_keypoints(
                _img,
                vis_kps,
                vis_valid,
                self.skeleton,
                filename,
                self.cfg,
                is_print=True,
            )
            filename = f"vis_{frame_id:04d}_{gt_hand_type}_3d.jpg"
            vis_kps = gt_joint_coord_cam.copy()
            vis_3d_keypoints(
                vis_kps, joint_valid, self.skeleton, filename, self.cfg, is_print=True
            )

    def evaluate(self, preds):

        print("\nEvaluation start...")
        # set False to view full images
        IS_CROP = True  # False

        gts = self.datalist
        preds_joint_coord, preds_rel_root_depth, preds_hand_type, inv_trans = (
            preds["joint_coord"],
            preds["rel_root_depth"],
            preds["hand_type"],
            preds["inv_trans"],
        )
        assert len(gts) == len(preds_joint_coord)
        sample_num = len(gts)

        mpjpe_sh = [[] for _ in range(self.joint_num * 2)]
        mpjpe_ih = [[] for _ in range(self.joint_num * 2)]
        mrrpe = []
        acc_hand_cls = 0
        hand_cls_cnt = 0
        img_size = None
        for n in range(sample_num):
            data = gts[n]
            img_path, bbox, cam_param, joint, gt_hand_type, hand_type_valid = (
                data["img_path"],
                data["bbox"],
                data["cam_param"],
                data["joint"],
                data["hand_type"],
                data["hand_type_valid"],
            )
            if img_size is None:
                img_size = cv2.imread(img_path).shape[:2]
            gt_joint_coord_img = joint["img_coord"].copy()  # original image
            gt_joint_coord_cam = joint["cam_coord"].copy()
            joint_valid = joint["valid"]
            if IS_CROP:
                inputs, targets, meta_info = self.__getitem__(n)
                cam_param = meta_info["cam_param_updated"]

            focal = cam_param["focal"]
            princpt = cam_param["princpt"]

            # restore xy coordinates to original image space
            pred_joint_coord_img = preds_joint_coord[n].copy()
            pred_joint_coord_img[:, 0] = (
                pred_joint_coord_img[:, 0]
                / self.cfg.output_hm_shape[2]
                * self.cfg.input_img_shape[1]
            )
            pred_joint_coord_img[:, 1] = (
                pred_joint_coord_img[:, 1]
                / self.cfg.output_hm_shape[1]
                * self.cfg.input_img_shape[0]
            )
            # for j in range(self.joint_num*2):
            #     pred_joint_coord_img[j, :2] = trans_point2d(pred_joint_coord_img[j, :2], inv_trans[n])
            # restore depth to original camera space
            pred_joint_coord_img[:, 2] = (
                pred_joint_coord_img[:, 2] / self.cfg.output_hm_shape[0] * 2 - 1
            ) * (self.cfg.bbox_3d_size / 2)

            # mrrpe
            if (
                gt_hand_type == "interacting"
                and joint_valid[self.root_joint_idx["left"]]
                and joint_valid[self.root_joint_idx["right"]]
            ):
                pred_rel_root_depth = (
                    preds_rel_root_depth[n] / self.cfg.output_root_hm_shape * 2 - 1
                ) * (self.cfg.bbox_3d_size_root / 2)

                pred_left_root_img = pred_joint_coord_img[
                    self.root_joint_idx["left"]
                ].copy()
                pred_left_root_img[2] += (
                    data["abs_depth"]["right"] + pred_rel_root_depth
                )
                pred_left_root_cam = pixel2cam(
                    pred_left_root_img[None, :], focal, princpt
                )[0]

                pred_right_root_img = pred_joint_coord_img[
                    self.root_joint_idx["right"]
                ].copy()
                pred_right_root_img[2] += data["abs_depth"]["right"]
                pred_right_root_cam = pixel2cam(
                    pred_right_root_img[None, :], focal, princpt
                )[0]

                pred_rel_root = pred_left_root_cam - pred_right_root_cam
                gt_rel_root = (
                    gt_joint_coord_cam[self.root_joint_idx["left"]]
                    - gt_joint_coord_cam[self.root_joint_idx["right"]]
                )
                mrrpe.append(float(np.sqrt(np.sum((pred_rel_root - gt_rel_root) ** 2))))

            # add root joint depth
            pred_joint_coord_img[self.joint_type["right"], 2] += data["abs_depth"][
                "right"
            ]
            pred_joint_coord_img[self.joint_type["left"], 2] += data["abs_depth"][
                "left"
            ]

            # back project to camera coordinate system
            pred_joint_coord_cam = pixel2cam(pred_joint_coord_img, focal, princpt)

            # root joint alignment
            for h in ("right", "left"):
                pred_joint_coord_cam[self.joint_type[h]] = (
                    pred_joint_coord_cam[self.joint_type[h]]
                    - pred_joint_coord_cam[self.root_joint_idx[h], None, :]
                )
                gt_joint_coord_cam[self.joint_type[h]] = (
                    gt_joint_coord_cam[self.joint_type[h]]
                    - gt_joint_coord_cam[self.root_joint_idx[h], None, :]
                )

            # mpjpe
            for j in range(self.joint_num * 2):
                if joint_valid[j]:
                    if gt_hand_type == "right" or gt_hand_type == "left":
                        mpjpe_sh[j].append(
                            np.sqrt(
                                np.sum(
                                    (pred_joint_coord_cam[j] - gt_joint_coord_cam[j])
                                    ** 2
                                )
                            )
                        )
                    else:
                        mpjpe_ih[j].append(
                            np.sqrt(
                                np.sum(
                                    (pred_joint_coord_cam[j] - gt_joint_coord_cam[j])
                                    ** 2
                                )
                            )
                        )

            # handedness accuray
            if hand_type_valid:
                if (
                    gt_hand_type == "right"
                    and preds_hand_type[n][0] > 0.5
                    and preds_hand_type[n][1] < 0.5
                ):
                    acc_hand_cls += 1
                elif (
                    gt_hand_type == "left"
                    and preds_hand_type[n][0] < 0.5
                    and preds_hand_type[n][1] > 0.5
                ):
                    acc_hand_cls += 1
                elif (
                    gt_hand_type == "interacting"
                    and preds_hand_type[n][0] > 0.5
                    and preds_hand_type[n][1] > 0.5
                ):
                    acc_hand_cls += 1
                hand_cls_cnt += 1

        if hand_cls_cnt > 0:
            print("Handedness accuracy: " + str(acc_hand_cls / hand_cls_cnt))
        if len(mrrpe) > 0:
            print("MRRPE: " + str(sum(mrrpe) / len(mrrpe)))
        print()

        if len(mpjpe_ih[j]) > 0:
            tot_err = []
            eval_summary = "MPJPE for each joint: \n"
            for j in range(self.joint_num * 2):
                tot_err_j = np.mean(
                    np.concatenate((np.stack(mpjpe_sh[j]), np.stack(mpjpe_ih[j])))
                )
                joint_name = self.skeleton[j]["name"]
                eval_summary += joint_name + ": %.2f, " % tot_err_j
                tot_err.append(tot_err_j)
            print(eval_summary)
            print("MPJPE for all hand sequences: %.2f\n" % (np.nanmean(tot_err)))

            eval_summary = "MPJPE for each joint: \n"
            for j in range(self.joint_num * 2):
                mpjpe_ih[j] = np.mean(np.stack(mpjpe_ih[j]))
                joint_name = self.skeleton[j]["name"]
                eval_summary += joint_name + ": %.2f, " % mpjpe_ih[j]
            print(eval_summary)
            print(
                "MPJPE for interacting hand sequences: %.2f\n" % (np.nanmean(mpjpe_ih))
            )

        eval_summary = "MPJPE for each joint: \n"
        for j in range(self.joint_num * 2):
            if len(mpjpe_sh[j]) > 0:
                mpjpe_sh[j] = np.mean(np.stack(mpjpe_sh[j]))
            else:
                mpjpe_sh[j] = np.nan
            joint_name = self.skeleton[j]["name"]
            eval_summary += joint_name + ": %.2f, " % mpjpe_sh[j]
        print(eval_summary)
        print("MPJPE for single hand sequences: %.2f\n" % (np.nanmean(mpjpe_sh)))


if __name__ == "__main__":
    from tqdm import tqdm
    from src.main.config import Config

    args_gpu_ids = "0"
    args_dataset = "AssemblyHands-Ego"
    args_test_set = "val"
    args_test_epoch = "20"
    cfg = Config(dataset=args_dataset)
    cfg.set_args(args_gpu_ids)
    cfg.print_config()

    from src.common.base import Tester

    tester = Tester(cfg, args_test_epoch)
    tester._make_batch_generator(args_test_set, shuffle=True)
    checkpoint_dir = "checkpoints"
    tester._make_model(checkpoint_dir)

    # visualize samples
    tester.testset.view_samples()

    for itr, (inputs, targets, meta_info) in enumerate(tqdm(tester.batch_generator)):
        print(f"OK: data get {itr}")
        break
