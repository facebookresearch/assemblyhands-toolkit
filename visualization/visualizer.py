import os
import cv2
import json
from tqdm import tqdm
import numpy as np
import argparse
import glob

from src.common.utils.preprocessing import load_skeleton
from src.common.utils.vis import vis_keypoints
from src.common.utils.transforms import world2pixel


def visualize_pose_on_frame(frame_path, camera_name, joints_3d_dict, calib, skeleton):
    """
    Overlay 3D keypoints on the given frame and return the visualized frame.
    """
    img = cv2.imread(frame_path)
    assert len(img.shape) == 3, "Image must be a 3-channel BGR image."
    frame_id = os.path.basename(frame_path).split('.')[0]

    K = calib["intrinsics"][camera_name]
    Rt = calib["extrinsics"][frame_id][camera_name]
    KRT = np.dot(K, Rt)

    joints_3d = np.asarray(joints_3d_dict[frame_id]["world_coord"])
    joint_valid = np.asarray(joints_3d_dict[frame_id]["joint_valid"])
    joints_2d = world2pixel(joints_3d, KRT)[:, :2]

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    vis_img = vis_keypoints(img, joints_2d, joint_valid, skeleton)

    return vis_img


def main(args):
    os.makedirs(args.save_path, exist_ok=True)
    
    # set image path
    ego_image_dir = os.path.join(args.ego_image_root, args.vis_set, args.vis_seq_name, args.vis_camera_name)
    assert os.path.exists(ego_image_dir), f"Path {ego_image_dir} does not exist."
    frame_list = sorted(glob.glob(f"{ego_image_dir}/*"))        
    
    # read annotation
    with open(args.calib_file) as f:
        ego_calib = json.load(f)["calibration"][args.vis_seq_name]
    with open(args.kpt_file) as f:
        joints_3d_dict = json.load(f)["annotations"][args.vis_seq_name]    
    skeleton = load_skeleton(args.skeleton_file, 42)
    
    # set video writer
    output_video_path = os.path.join(args.save_path, f'vis_video_{args.vis_seq_name}_{args.vis_camera_name.replace("_mono10bit", "")}.mp4')
    tmp_output_video_path = output_video_path.replace("vis_video", "tmp_vis_video")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vis_img = cv2.imread(frame_list[0])
    H, W = vis_img.shape[:2]
    videoWriter = cv2.VideoWriter(tmp_output_video_path, fourcc, args.vis_fps, (W, H))
    
    for frame_path in tqdm(frame_list[:args.vis_limit]):
        vis_img = visualize_pose_on_frame(frame_path, args.vis_camera_name, joints_3d_dict, ego_calib, skeleton)
        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
        videoWriter.write(vis_img)
                
    videoWriter.release()    
    # reformat the video
    ret = os.system(f"ffmpeg -y -i {tmp_output_video_path} -vcodec libx264 {output_video_path}")
    assert ret == 0, "check ffmpeg processing"
    os.system(f"rm {tmp_output_video_path}")
    print(f"Saved visualization video to {output_video_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize hand pose on video frames and create a video.")
    parser.add_argument("--ego_image_root", default="./data/assemblyhands/images/ego_images_rectified", type=str)
    parser.add_argument("--vis_set", default="val", type=str, help="val or test")
    parser.add_argument("--vis_camera_name", default="HMC_21179183_mono10bit", type=str)
    parser.add_argument("--vis_seq_name", default="nusar-2021_action_both_9081-c11b_9081_user_id_2021-02-12_161433", type=str)
    parser.add_argument("--calib_file", default="./data/assemblyhands/annotations/val/assemblyhands_val_ego_calib_v1-1.json", type=str)
    parser.add_argument("--data_file", default="./data/assemblyhands/annotations/val/assemblyhands_val_ego_data_v1-1.json", type=str)
    parser.add_argument("--kpt_file", default="./data/assemblyhands/annotations/val/assemblyhands_val_joint_3d_v1-1.json", type=str)
    parser.add_argument("--skeleton_file", default="./data/assemblyhands/annotations/skeleton.txt", type=str)
    parser.add_argument("--save_path", default="vis-results/vis-gt", type=str)
    parser.add_argument("--vis_limit", default=50, type=int, help="Limit number of frames for visualization.")
    parser.add_argument("--vis_fps", default=10, type=int, help="Frames per second for the output video.")
    args = parser.parse_args()        
    
    main(args)
