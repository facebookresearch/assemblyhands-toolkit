# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
 
import json
import numpy as np
from camera import from_json   

if __name__ == "__main__":
    # mapping from points in the world coordinates to the image coordinates of original images (before undistortion)
    
    seq_name = "nusar-2021_action_both_9081-c11b_9081_user_id_2021-02-12_161433"
    camera_name = "HMC_21179183_mono10bit"
    # camera_name = "C10095_rgb"
    
    json_file = f"calib/nimble_json_calib/{seq_name}.json"
    with open(json_file, "r") as f:
        camera_jsons = json.load(f)
    
    # get camera model
    for js in camera_jsons:
        cam = from_json(js)
        if cam.serial == camera_name:
            break
        
    print(cam.serial, cam)
    origin = np.array([10, 10, 10])
    print(f"origin {origin} projects to", cam.world_to_window(origin), "\n")
