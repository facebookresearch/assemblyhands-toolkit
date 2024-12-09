# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""
Camera Models

A camera model (as used here) defines a mapping between 3D coordinates
in the world and 2D coordinates in some sensor image, for use in
computer vision.

This generally consists of three steps:

1. Project the 3D coordinates `v` down to 2D coordinates `p` via some
   fixed projection function.

2. Distort the 2D coordinates using fitted coefficients of some general
   function (e.g. a polynomial), producing distorted coordinates `q`.

3. Scale the distorted coordinates by the focal length and offset to the
   focal center, to get window coordinates `w`.


## Coordinate convention

Right-handed coords following the OpenCV convention:
`+X` and `+U` are to the right, `+Y` and `+V` are down. `+Z` is in
front of camera plane, `-Z` is behind.

## Projections

There are many possible projection functions to choose from.
All of the camera models are defined so they produce the same results in
neighborhood of the image center: near the point [0,0,1], `u ~= x` and
`v ~= y`.

The interface to projection functions is in :class:`CameraProjection`.


## Distortion functions

The distortion function applies a fitted polynomial to adjust the
`(u,v)` coordinates so they closely match some actual physical sensor.

Distortion coefficients are usually broken into "radial" and "tangential"
terms, named "k1, k2, ..." and "p1, p2, ..." respectively.


## Window coordinates

Finally, the window coordinates are found by scaling the distorted `u,v`
by the focal length `f` (or `fx, fy` for non-square pixels), and adding
the window center `cx, cy`.

Note: There is an important subtlety here in how the continuous 2D range
of coordinates then maps to 2D array indices. Two conventions are
possible:

1. Coordinate (0.0, 0.0) maps to the *corner* of the pixel at
   `image[0,0]`.

2. Coordinate (0.0, 0.0) maps to the *center* of the pixel at
   `image[0,0]`.

   This means that the full range coordinates within a window goes from
   `(-0.5, -0.5)` to `(width-0.5, height-0.5)`. If an image is scaled by
   a factor `s`, coordinates are *not* simply scaled by the same factor:
   `[cx,cy]` maps to `[(cx + 0.5) * s - 0.5, (cy + 0.5) * s - 0.5]`.

The former convention tends to produce simpler code that is easier to
get correct, and therefore is dominant in computer graphics.

Unfortunately the latter convention is dominant in computer vision, and
is what OpenCV does (though this isn't clearly documented), so that is
what this library assumes.


# ---------------------------------------------------------------------
# API Conventions and naming
#
# Points have the xyz or uv components in the last axis, and may have
# arbitrary batch shapes. ([...,2] for 2d and [...,3] for 3d).
#
# v
#    3D xyz position in eye space, usually unit-length.
# p
#    projected uv coordinates: `p = project(v)`
# q
#    distorted uv coordinates: `q = distort(p)`
# w
#    window coordinates: `q = q * f + [cx, cy]`
#
# A trailing underscore (e.g. `p_`, `q_`) should be read as "hat", and
# generally indicates an approximation to another value.
# ---------------------------------------------------------------------
"""
import os
import abc
import json
import math
from typing import Tuple, Type

import numpy as np

try:
    import affine
except:
    from calib import affine
try:
    import camera_distortion as dis
except:
    from calib import camera_distortion as dis


class CameraModel(dis.CameraProjection, abc.ABC):
    """
    Parameters
    ----------
    width, height : int
        Size of the sensor window

    f : float or tuple(float, float)
        Focal length

    c : tuple(float, float)
        Optical center in window coordinates

    distort_coeffs
        Forward distortion coefficients (eye -> window).

        If this is an instance of DistortionModel, it will be used as-is
        (even if it's a different polynomial than this camera model
        would normally use.) If it's a simple tuple or array, it will
        used as coefficients for `self.distortion_model`.

    extrinsics : np.ndarray
        Camera's position and orientation in world space, represented as
        a 3x4 or 4x4 matrix.

        The matrix be a rigid transform (only rotation and translation).

        You can change a camera's extrinsics after construction by
        assigning to or modifying this matrix.

    serial : string
        Aribtrary string identifying the specific camera.

    """

    width: int
    height: int

    f: Tuple[float, float]
    c: Tuple[float, float]

    extrinsics: np.ndarray

    distortion_model: Type[dis.DistortionModel]
    distort: dis.DistortionModel

    def __init__(
        self,
        width,
        height,
        f,
        c,
        distort_coeffs,
        extrinsics=None,
        serial="",
    ):
        self.width = width
        self.height = height
        self.serial = serial

        # f can be either a scalar or (fx,fy) pair. We only fit scalars,
        # but may load (fx, fy) from a stored file.
        self.f = tuple(np.broadcast_to(f, 2))
        self.c = tuple(c)

        if extrinsics is None:
            self.extrinsics = np.eye(4)
        else:
            self.extrinsics = affine.as_4x4(extrinsics, copy=True)
            if (
                np.abs((self.extrinsics.T @ self.extrinsics)[:3, :3] - np.eye(3)).max()
                >= 1.0e-5
            ):
                info_str = "camera extrinsics must be a rigid transform\n"
                info_str = info_str + "T\n{}\n".format(self.extrinsics.T)
                info_str = info_str + "(T*T_t - I).max()\n{}\n".format(
                    np.abs(
                        (self.extrinsics.T @ self.extrinsics)[:3, :3] - np.eye(3)
                    ).max()
                )
                raise ValueError(info_str)

        if isinstance(distort_coeffs, dis.DistortionModel):
            self.distort = distort_coeffs
        else:
            self.distort = self.distortion_model(*distort_coeffs)

    def __repr__(self):
        return (
            f"{type(self).__name__}({self.serial}, {self.width}x{self.height}, f={self.f} c={self.c})"
        )
    
    def update_extrinsics(self, extrinsics):        
        """Update the value of extrinsics for moving camera"""        
        self.extrinsics = np.linalg.inv(np.array(extrinsics))

    def world_to_window(self, v):
        """Project world space points to 2D window coordinates"""
        return self.eye_to_window(self.world_to_eye(v))

    def world_to_window3(self, v):
        """Project world space points to 3D window coordinates (uv + depth)"""
        return self.eye_to_window3(self.world_to_eye(v))

    def world_to_eye(self, v):
        """
        Apply camera extrinsics to points `v` to get eye coords
        """
        return affine.transform_vec3(self.extrinsics.T, v - self.extrinsics[:3, 3])

    def eye_to_world(self, v):
        """
        Apply inverse camera extrinsics to eye points `v` to get world coords
        """
        return affine.transform3(self.extrinsics, v)

    def eye_to_window(self, v):
        """Project eye coordinates to 2d window coordinates"""
        p = self.project(v)
        q = self.distort.evaluate(p)
        return q * self.f + self.c

    def eye_to_window3(self, v):
        """Project eye coordinates to 3d window coordinates (uv + depth)"""
        p = self.project3(v)
        q = self.distort.evaluate(p[..., :2])
        p[..., :2] = q * self.f + self.c
        return p


# =============
# Camera models
# =============


class OpenCVCameraModel(dis.PerspectiveProjection, CameraModel):
    distortion_model = dis.OpenCVDistortion
    model_fov_limit = 50 * (math.pi / 180)


class OV62CameraModel(dis.ArctanProjection, CameraModel):
    distortion_model = dis.OV62Distortion
    model_fov_limit = math.pi / 2


model_by_name = {
    "OpenCV": OpenCVCameraModel,
    "OVFishEye62": OV62CameraModel,
}


def from_json(js):
    """
    File format example::

    {
        'DistortionModel': 'FishEye',
        'ImageSizeX': 640,
        'ImageSizeY': 480,
        'ModelViewMatrix': [
            [-0.7446, 0.4606, -0.4830, -14.3485],
            [0.1369, 0.8137, 0.5648, 23.6725],
            [0.6532, 0.3543, -0.6690, -130.5179],
            [0.0, 0.0, 0.0, 1.0]
        ],
        'SerialNo': '0072510f171e0702010000031b0a00010:mono',
        'cx': 311.2256,
        'cy': 234.9109,
        'fx': 189.5322,
        'fy': 189.5141,
        'k1': 0.3050,
        'k2': -0.05414,
        'k3': -0.04977,
        'k4': 0.01936,
        'k5': 0.0,
        'k6': 0.0,
        'p1': -0.0001268,
        'p2': -0.0001938,
        'p3': 0.7929,
        'p4': -0.2831
    }
    """
    if isinstance(js, str):
        js = json.loads(js)
    js = js.get("Camera", js)

    width = js["ImageSizeX"]
    height = js["ImageSizeY"]
    model = js["DistortionModel"]
    fx = js["fx"]
    fy = js["fy"]
    cx = js["cx"]
    cy = js["cy"]
    extrinsics = np.linalg.inv(np.array(js["ModelViewMatrix"]))
    serial=js["SerialNo"]

    cls = model_by_name[model]
    distort_params = cls.distortion_model._fields
    coeffs = [js[name] for name in distort_params]

    js_inverse = js.get("inverse", None)
    if js_inverse is not None:
        uncoeffs = [js_inverse[name] for name in distort_params]
    else:
        uncoeffs = None

    # return cls(width, height, (fx, fy), (cx, cy), coeffs, uncoeffs, extrinsics)
    return cls(width, height, (fx, fy), (cx, cy), distort_coeffs=coeffs, extrinsics=extrinsics, serial=serial)