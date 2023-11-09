# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

JMAP_INTERHAND_TO_PANOPTIC =  [
                        20,             # wrist
                        3, 2, 1, 0,     # thumb                        
                        7, 6, 5, 4,      # index
                        11, 10, 9, 8,    # middle                        
                        15, 14, 13, 12,  # ring  
                        19, 18, 17, 16,  # pinky                                                                  
                    ]

SPLIT2SEQ = {
    "train": [
        "nusar-2021_action_both_9014-a23_9014_user_id_2021-02-02_142800",
        "nusar-2021_action_both_9015-b05b_9015_user_id_2021-02-02_161800",
        "nusar-2021_action_both_9026-c12b_9026_user_id_2021-02-03_171236",
        "nusar-2021_action_both_9034-c08b_9034_user_id_2021-02-23_175357",
        "nusar-2021_action_both_9041-c07c_9041_user_id_2021-02-05_104114",
        "nusar-2021_action_both_9051-c10a_9051_user_id_2021-02-22_120421",
        "nusar-2021_action_both_9053-c08b_9053_user_id_2021-02-08_140757",
        "nusar-2021_action_both_9054-c01a_9054_user_id_2021-02-08_160424",
        "nusar-2021_action_both_9062-a21_9062_user_id_2021-02-09_151231",
        "nusar-2021_action_both_9071-c14a_9071_user_id_2021-02-11_092353",
        "nusar-2021_action_both_9072-c10a_9072_user_id_2021-02-11_112415",
        "nusar-2021_action_both_9075-c08b_9075_user_id_2021-02-12_101609",
        "nusar-2021_action_both_9076-c13d_9076_user_id_2021-02-12_115510",
        "nusar-2021_action_both_9085-c10a_9085_user_id_2021-02-22_174720",
    ],
    "val": [
        "nusar-2021_action_both_9012-c07c_9012_user_id_2021-02-01_164345",
        "nusar-2021_action_both_9081-c11b_9081_user_id_2021-02-12_161433",
    ],
    "test": [
        "nusar-2021_action_both_9021-c10a_9021_user_id_2021-02-23_100458",
        "nusar-2021_action_both_9034-c08b_9034_user_id_2021-02-04_161726",
        "nusar-2021_action_both_9061-c13d_9061_user_id_2021-02-09_143830",
        "nusar-2021_action_both_9074-a29_9074_user_id_2021-02-11_154856",
    ]
}

SEQ2SPLIT = {}
for split, seq_list in SPLIT2SEQ.items():
    for seq in seq_list:
        SEQ2SPLIT[seq] = split
        

"""
please see camera arrangement from Assembly101
""" 
CAM2ID = {
    "C10095_rgb": "exo1", # v1
    "C10115_rgb": "exo2", # v2
    "C10118_rgb": "exo3", # v3
    "C10119_rgb": "exo4", # v4
    "C10379_rgb": "exo5", # v5
    "C10390_rgb": "exo6", # v6
    "C10395_rgb": "exo7", # v7
    "C10404_rgb": "exo8", # v8
    "HMC_84346135_mono10bit": "ego1", # e1
    "HMC_21176875_mono10bit": "ego1", # e1
    "HMC_84347414_mono10bit": "ego2", # e2
    "HMC_21176623_mono10bit": "ego2", # e2
    "HMC_84355350_mono10bit": "ego3", # e3
    "HMC_21110305_mono10bit": "ego3", # e3
    "HMC_84358933_mono10bit": "ego4", # e4
    "HMC_21179183_mono10bit": "ego4", # e4
}
ID2CAM = {}
for cam, view in CAM2ID.items():
    if view.startswith("ego"):
        if view not in ID2CAM:
            ID2CAM[view] = []
        ID2CAM[view].append(cam)
    else:
        ID2CAM[view] = cam
