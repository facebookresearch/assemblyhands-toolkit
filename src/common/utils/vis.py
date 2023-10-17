# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import os.path as osp
import cv2
import numpy as np
import matplotlib

# matplotlib.use('tkagg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
from src import cfg
from PIL import Image, ImageDraw
import io

hand_colors = [
    [0, 0, 255],
    [255, 0, 0],
]


def get_keypoint_rgb(skeleton):
    rgb_dict = {}
    for joint_id in range(len(skeleton)):
        joint_name = skeleton[joint_id]["name"]

        if joint_name.endswith("thumb_null"):
            rgb_dict[joint_name] = (255, 0, 0)
        elif joint_name.endswith("thumb3"):
            rgb_dict[joint_name] = (255, 51, 51)
        elif joint_name.endswith("thumb2"):
            rgb_dict[joint_name] = (255, 102, 102)
        elif joint_name.endswith("thumb1"):
            rgb_dict[joint_name] = (255, 153, 153)
        elif joint_name.endswith("thumb0"):
            rgb_dict[joint_name] = (255, 204, 204)
        elif joint_name.endswith("index_null"):
            rgb_dict[joint_name] = (0, 255, 0)
        elif joint_name.endswith("index3"):
            rgb_dict[joint_name] = (51, 255, 51)
        elif joint_name.endswith("index2"):
            rgb_dict[joint_name] = (102, 255, 102)
        elif joint_name.endswith("index1"):
            rgb_dict[joint_name] = (153, 255, 153)
        elif joint_name.endswith("middle_null"):
            rgb_dict[joint_name] = (255, 128, 0)
        elif joint_name.endswith("middle3"):
            rgb_dict[joint_name] = (255, 153, 51)
        elif joint_name.endswith("middle2"):
            rgb_dict[joint_name] = (255, 178, 102)
        elif joint_name.endswith("middle1"):
            rgb_dict[joint_name] = (255, 204, 153)
        elif joint_name.endswith("ring_null"):
            rgb_dict[joint_name] = (0, 128, 255)
        elif joint_name.endswith("ring3"):
            rgb_dict[joint_name] = (51, 153, 255)
        elif joint_name.endswith("ring2"):
            rgb_dict[joint_name] = (102, 178, 255)
        elif joint_name.endswith("ring1"):
            rgb_dict[joint_name] = (153, 204, 255)
        elif joint_name.endswith("pinky_null"):
            rgb_dict[joint_name] = (255, 0, 255)
        elif joint_name.endswith("pinky3"):
            rgb_dict[joint_name] = (255, 51, 255)
        elif joint_name.endswith("pinky2"):
            rgb_dict[joint_name] = (255, 102, 255)
        elif joint_name.endswith("pinky1"):
            rgb_dict[joint_name] = (255, 153, 255)
        else:
            rgb_dict[joint_name] = (230, 230, 0)

    return rgb_dict


def draw_bbox_on_image(img, bbox):
    if not isinstance(bbox, list):
        bbox = [bbox]
    idx = 0
    for idx in range(len(bbox)):
        box = list(bbox[idx])
        x0, y0, w, h = box
        x1, y1 = x0 + w, y0 + h
        cv2.rectangle(
            img,
            (int(x0), int(y0)),
            (int(x1), int(y1)),
            hand_colors[idx % len(hand_colors)],
            2,
        )
    return img


def vis_keypoints(
    img,
    kps,
    score,
    skeleton,
    filename=None,
    vis_dir=None,
    score_thr=0.4,
    line_width=3,
    circle_rad=3,
    bbox=None,
    is_print=False,
):

    rgb_dict = get_keypoint_rgb(skeleton)
    if img.shape[0] == 3:
        img = img.transpose(1, 2, 0)
    _img = Image.fromarray(img.astype("uint8"))
    draw = ImageDraw.Draw(_img)
    for i in range(len(skeleton)):
        joint_name = skeleton[i]["name"]
        pid = skeleton[i]["parent_id"]
        parent_joint_name = skeleton[pid]["name"]

        kps_i = (kps[i][0].astype(np.int32), kps[i][1].astype(np.int32))
        kps_pid = (kps[pid][0].astype(np.int32), kps[pid][1].astype(np.int32))

        if score[i] > score_thr and score[pid] > score_thr and pid != -1:
            draw.line(
                [(kps[i][0], kps[i][1]), (kps[pid][0], kps[pid][1])],
                fill=rgb_dict[parent_joint_name],
                width=line_width,
            )
        if score[i] > score_thr:
            draw.ellipse(
                (
                    kps[i][0] - circle_rad,
                    kps[i][1] - circle_rad,
                    kps[i][0] + circle_rad,
                    kps[i][1] + circle_rad,
                ),
                fill=rgb_dict[joint_name],
            )
        if score[pid] > score_thr and pid != -1:
            draw.ellipse(
                (
                    kps[pid][0] - circle_rad,
                    kps[pid][1] - circle_rad,
                    kps[pid][0] + circle_rad,
                    kps[pid][1] + circle_rad,
                ),
                fill=rgb_dict[parent_joint_name],
            )

    if bbox is not None:
        _img = draw_bbox_on_image(np.asarray(_img), bbox)
        _img = Image.fromarray(_img)

    if vis_dir is not None:
        _img.save(osp.join(vis_dir, filename))
        if is_print:
            print(f"save vis image to {osp.join(vis_dir, filename)}")

    return np.asarray(_img)


def fig2img(fig):
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data


def vis_3d_keypoints(
    kps_3d,
    score,
    skeleton,
    filename=None,
    vis_dir=None,
    score_thr=0.4,
    line_width=3,
    circle_rad=3,
    is_print=False,
):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    white = (1.0, 1.0, 1.0, 0.0)
    ax.xaxis.set_pane_color(white)
    ax.yaxis.set_pane_color(white)
    ax.zaxis.set_pane_color(white)

    ax.tick_params("x", labelbottom=False)
    ax.tick_params("y", labelleft=False)
    ax.tick_params("z", labelleft=False)

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    rgb_dict = get_keypoint_rgb(skeleton)

    for i in range(len(skeleton)):
        joint_name = skeleton[i]["name"]
        pid = skeleton[i]["parent_id"]
        parent_joint_name = skeleton[pid]["name"]

        x = np.array([kps_3d[i, 0], kps_3d[pid, 0]])
        y = np.array([kps_3d[i, 1], kps_3d[pid, 1]])
        z = np.array([kps_3d[i, 2], kps_3d[pid, 2]])

        if score[i] > score_thr and score[pid] > score_thr and pid != -1:
            ax.plot(
                x,
                z,
                -y,
                c=np.array(rgb_dict[parent_joint_name]) / 255.0,
                linewidth=line_width,
            )
        if score[i] > score_thr:
            ax.scatter(
                kps_3d[i, 0],
                kps_3d[i, 2],
                -kps_3d[i, 1],
                c=np.array(rgb_dict[joint_name]).reshape(1, 3) / 255.0,
                marker="o",
            )
        if score[pid] > score_thr and pid != -1:
            ax.scatter(
                kps_3d[pid, 0],
                kps_3d[pid, 2],
                -kps_3d[pid, 1],
                c=np.array(rgb_dict[parent_joint_name]).reshape(1, 3) / 255.0,
                marker="o",
            )

    img_array = None
    if vis_dir is not None:
        fig.savefig(osp.join(vis_dir, filename), dpi=fig.dpi)
        if is_print:
            print(f"save vis image to {osp.join(vis_dir, filename)}")
    else:
        img_array = fig2img(fig)
        img_array = img_array[:, :, :3]
        img_array = img_array[:, :, ::-1]
    plt.close()
    return img_array
