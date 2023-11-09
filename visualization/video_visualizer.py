# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import sys
import cv2
import numpy as np
import subprocess
import argparse
import json
import glob
import pickle
from tqdm import tqdm

from src.common.utils.preprocessing import load_skeleton, get_calib
from src.common.utils.transforms import world2cam_assemblyhands as world2cam
from src.common.utils.transforms import world2pixel
from src.common.utils.vis import vis_keypoints, get_keypoint_rgb
from src.dataset import SEQ2SPLIT, SPLIT2SEQ, CAM2ID, ID2CAM, JMAP_INTERHAND_TO_PANOPTIC

ANNOT_FPS = 60
RECTIFIED_VIDEO_FPS = 30


def visualize_pose_on_video(vis_items, joints_3d_dict, skeleton_file, save_path, calib, args=None):
    frame_id_list = sorted(list(joints_3d_dict.keys()))
    start_kpt_frame_id = int(frame_id_list[0])
    end_kpt_frame_id = int(frame_id_list[-1])
    print(
        f"start_kpt_frame_id: {start_kpt_frame_id}, end_kpt_frame_id: {end_kpt_frame_id}")
    vis_img_dict = {}
    skeleton = load_skeleton(skeleton_file, 42)
    if len(vis_items) < 3:
        vis_height = 512
    else:
        vis_height = 300
    for i, vis_item in enumerate(vis_items):
        video_path, camera_name = vis_item
        stream = cv2.VideoCapture(video_path)
        video_fps, num_frames = stream.get(
            cv2.CAP_PROP_FPS), stream.get(cv2.CAP_PROP_FRAME_COUNT)
        print(
            f"video path: {video_path}, video fps {video_fps}, num_frames {num_frames}")
        assert video_fps == RECTIFIED_VIDEO_FPS, f"video fps {video_fps} is not {RECTIFIED_VIDEO_FPS}"
        if i == 0:  # set vis start id
            start_frame_id = int(frame_id_list[0])
            if args.start_margin > 0:
                print(f"start after {args.start_margin}s")
                start_frame_id = start_frame_id + args.start_margin * video_fps
        max_n_frame = int(args.max_dur * video_fps)
        end_frame_id = min(end_kpt_frame_id + 1, start_frame_id + max_n_frame)
        # NOTE: rectified video starts from synced point
        video_frame_cnt = start_frame_id
        print(
            f"process the video: start frame: {video_frame_cnt}, end frame: {end_frame_id}")
        frame_interval = int(ANNOT_FPS//RECTIFIED_VIDEO_FPS)
        for j, frame_id in enumerate(tqdm(range(start_frame_id, end_frame_id, frame_interval))):
            assert video_frame_cnt == frame_id, f"video_frame_cnt: {video_frame_cnt}, frame_id: {frame_id}"
            frame_id = f"{frame_id:06d}"
            ret, img = stream.read()
            if not ret:
                break
            h, w = img.shape[:2]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            video_frame_cnt += frame_interval

            # set vis image
            resize_ratio = 1
            if h > 1000:
                resize_ratio = 0.5
                img = cv2.resize(
                    img, (int(w*resize_ratio), int(h*resize_ratio)))
            vis_img = img.copy()
            h1, w1 = vis_img.shape[:2]

            # get joints
            if frame_id in joints_3d_dict:  # show joints
                joints_3d = np.asarray(joints_3d_dict[frame_id]["world_coord"])
                try:
                    joint_valid = np.asarray(
                        joints_3d_dict[frame_id]["joint_valid"])
                except:
                    joint_valid = np.ones([joints_3d.shape[0]])
                K = np.asarray(calib["intrinsics"][camera_name])
                RT = np.asarray(calib["extrinsics"][camera_name])
                joints_2d = world2pixel(joints_3d, np.dot(K, RT))[:, :2]

                if resize_ratio != 1:
                    joints_2d = joints_2d * resize_ratio
                # set joint valid for vis
                joint_valid = np.logical_and.reduce([
                    joint_valid,
                    joints_2d.min(axis=1) >= 0,
                    joints_2d[:, 0] < w1,
                    joints_2d[:, 1] < h1])
                vis_img = vis_keypoints(
                    vis_img, joints_2d, joint_valid, skeleton, is_print=True)
                vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
            else:
                raise NotImplementedError()

            vis_img_2d_resized = cv2.resize(
                vis_img, (int(w1 * (vis_height/h1)), vis_height))
            if frame_id not in vis_img_dict:
                vis_img_dict[frame_id] = []
            vis_img_dict[frame_id].append(vis_img_2d_resized.astype(np.uint8))

    # merge vis images and save as a video
    print(f"start saving video to {os.path.dirname(save_path)}")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    tmp_save_path = save_path.replace("vis_", "tmp_vis_")
    assert len(vis_img_dict) > 0, "no vis image found"
    for k, frame_id in enumerate(tqdm(vis_img_dict)):
        vis_img_list = vis_img_dict[frame_id]
        vis_img = np.concatenate(vis_img_list, axis=1)
        H, W = vis_img.shape[:2]
        if k == 0:
            videoWriter = cv2.VideoWriter(
                tmp_save_path, fourcc, RECTIFIED_VIDEO_FPS, (W, H))
        videoWriter.write(vis_img)
    videoWriter.release()
    stream.release()
    # reformat the video
    ret = os.system(
        f"ffmpeg -y -i {tmp_save_path} -vcodec libx264 {save_path}")
    assert ret == 0, "check ffmpeg processing"
    os.system(f"rm {tmp_save_path}")
    print(f"Saved visualization video to\n{save_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root", default="./data/assemblyhands", type=str)
    parser.add_argument("--annot-root", default="data/assemblyhands", type=str)
    parser.add_argument("--vis-split", default="val", type=str)
    parser.add_argument("--vis-seq-id", default=1, type=int)
    parser.add_argument("--vis-camera-id", default='exo2,ego3', type=str)
    parser.add_argument("--kpt-file", default=None, type=str)
    parser.add_argument("--kpt-type", default='gt', type=str)
    parser.add_argument('--start-margin', default=0,
                        type=int, help="unit: second")
    parser.add_argument('--max-dur', default=10, type=int, help="unit: second")
    parser.add_argument("--save-dir", default='vis-results/videos', type=str)
    args = parser.parse_args()

    def print_message(message): return print(
        f"""\n\n###################################\t{message}\t###################################""")

    # get seq & camera name & skeleton file
    seq_list = SPLIT2SEQ[args.vis_split]
    seq_name = seq_list[args.vis_seq_id]
    vis_seq_list = [(args.vis_seq_id, seq_name)]
    skeleton_file = f"{args.annot_root}/annotations/skeleton.txt"
    assert os.path.exists(
        skeleton_file), f"skeleton_file: {skeleton_file} not found"
    # set save path
    os.makedirs(args.save_dir, exist_ok=True)

    # per-seq process
    for (seq_id, seq_name) in vis_seq_list:
        print_message(f"vis seq: {args.vis_split}{seq_id}: {seq_name}")
        save_filename = f"vis_{args.kpt_type}_view_{args.vis_camera_id}"

        if "," in args.vis_camera_id:
            camera_ids = args.vis_camera_id.split(",")
        else:
            camera_ids = [args.vis_camera_id]

        vis_items = []
        for camera_id in camera_ids:
            camera_name = ID2CAM[camera_id]
            # set video path
            # find the heaset type (HMC_2* or HMC_8*)
            if camera_id.startswith("ego") and isinstance(camera_name, list):
                for _camera_name in camera_name:
                    video_path = os.path.join(
                        args.data_root, "videos", seq_name, f"{_camera_name}.mp4")
                    if os.path.exists(video_path):
                        camera_name = _camera_name
                        break
                raise NotImplementedError(
                    "ego video processing is not implemented yet")
            elif camera_id.startswith("exo"):
                video_path = os.path.join(
                    args.data_root, "videos/exo_videos_rectified", seq_name, f"{camera_name}.mp4")
            assert os.path.exists(
                video_path), f"video path: {video_path} not found"
            vis_items.append((video_path, camera_name))

        # get calib dict
        calib = None
        if "ego" in args.vis_camera_id:
            version = "v1-1"
            calib_path = f"{args.annot_root}/annotations/{args.vis_split}/assemblyhands_{args.vis_split}_ego_calib_{version}.json"
            calib = get_calib(calib_path, seq_name, view_type="ego")
        else:
            calib_path = os.path.join(
                args.data_root, "videos/exo_videos_rectified", seq_name, "calib.txt")
            calib = get_calib(calib_path, view_type="exo")

        # get 3d keypoints
        if args.kpt_file is None or args.kpt_type == "gt":
            print("kpt file is not specified, use GT by default")
            keypoint_path = f"{args.annot_root}/annotations/{args.vis_split}/assemblyhands_{args.vis_split}_joint_3d_v1-1.json"
        else:
            keypoint_path = args.kpt_file
        assert keypoint_path is not None, "keypoint path is not found"
        with open(keypoint_path) as f:
            joints_3d_dict = json.load(f)["annotations"][seq_name]

        save_path = os.path.join(
            args.save_dir, f"{save_filename}_{seq_name}.mp4")
        visualize_pose_on_video(vis_items, joints_3d_dict,
                                skeleton_file, save_path, calib=calib, args=args)
