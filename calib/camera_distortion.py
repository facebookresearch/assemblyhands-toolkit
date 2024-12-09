# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""
Various implementations of Distortion and CameraProjection types.
"""
import abc
from typing import NamedTuple, Protocol, Sequence, runtime_checkable

import numpy as np


class CameraProjection(Protocol):
    """
    Defines a projection from a 3D `xyz` direction or point to 2D.
    """

    @classmethod
    @abc.abstractmethod
    def project(cls, v):
        """
        Project a 3d vector in eye space down to 2d.
        """
        ...

    @staticmethod
    @abc.abstractmethod
    def project3(v):
        """
        Same as project(), but return a 3d vector.

        The third component represents the distance to the point (either
        from the eye or Z plane, depending on the projection).
        """
        ...


class PerspectiveProjection(CameraProjection):
    """
    Perspective projection: `uv = xy / z`

    This has some nice properties:

    - straight lines map to straight lines
    - math is very nice, especially when working with homogeneous
      coordinates (4x4 matrices).
    - graphics rasterisers generally produce this by default

    However it produces extreme distortion at wide fields of view, and
    cannot represent FOV >= 180°.

    `project_perspective3()` returns the Z component unchanged.
    """

    @staticmethod
    def project(v):
        # map to [x/z, y/z]
        assert v.shape[-1] == 3
        return v[..., :2] / v[..., 2, None]

    @staticmethod
    def project3(v):
        # map to [x/z, y/z, z]
        x, y, z = np.moveaxis(v, -1, 0)
        return np.stack([x / z, y / z, z], axis=-1)


class ArctanProjection(CameraProjection):
    """
    Arctan projection: uv = xy / |xy| * atan2(|xy|, z)

    This is what the Oculus fisheye models use: the length of the 2D
    u vector is the angle in radians between +Z and the target ray
    direction.

    Unlike perspective projection, this can represent a FOV over 180
    degrees. However, distortion increases as the FOV gets wider. Near
    180 degrees, the aspect ratio of a pixel is about 1:1.57, which
    must then be corrected for in the distortion function. (This is why
    the OVRFisheye model needed to move to 6 radial terms: it has more
    distortion to correct for.)

    It's also relatively expensive to compute on since there is a trig
    function involved.
    """

    @staticmethod
    # pyre-ignore
    def project(p, eps: float = 2.0**-128):
        assert p.shape[-1] == 3
        x, y, z = np.moveaxis(p, -1, 0)
        r = np.sqrt(x * x + y * y)
        s = np.arctan2(r, z) / np.maximum(r, eps)
        return np.stack((x * s, y * s), axis=-1)

    @staticmethod
    def unproject(uv):
        assert uv.shape[-1] == 2
        u, v = np.moveaxis(uv, -1, 0)
        r = np.sqrt(u * u + v * v)
        c = np.cos(r)
        s = np.sinc(r / np.pi)
        return np.stack([u * s, v * s, c], axis=-1)

    @staticmethod
    # pyre-ignore
    def project3(p, eps: float = 2.0**-128):
        assert p.shape[-1] == 3
        xy = p[..., :2]
        z = p[..., 2]
        r2 = np.linalg.norm(xy, axis=-1)
        r3 = np.linalg.norm(p, axis=-1)
        s = np.arctan2(r2, z) / np.maximum(r2, eps)
        return np.stack([xy[..., 0] * s, xy[..., 1] * s, r3], axis=-1)

    @staticmethod
    def unproject3(uvd):
        assert uvd.shape[-1] == 3
        u, v, d = np.moveaxis(uvd, -1, 0)
        r = np.sqrt(u * u + v * v)
        c = np.cos(r)
        s = np.sinc(r / np.pi)
        return np.moveaxis(np.stack([u * s, v * s, c]) * d, 0, -1)


# =================
# Distortion models
# =================


@runtime_checkable
class DistortionModel(Protocol):
    """
    Defines the polynomial used for distortion.

    Implementations are NamedTuples, so coefficients are accessible as
    either named values or a single vector. The order of the
    coefficients in the vector is chosen for compatibility with C++
    implementations.
    """

    @abc.abstractmethod
    def evaluate(self: Sequence[float], p: np.ndarray) -> np.ndarray:
        """
        Evaluate this polynomial at the given points.

        This can also be called as a standalone function with a tuple or
        array of coefficients, without creating a NamedTuple instance.

        Arguments
        ---------
        p: ndarray[..., 2]
            Array of 2D points, of arbitrary batch shape.

        Returns
        -------
        q: ndarray[..., 2]
            Distorted points with same shape as input
        """
        ...


class OpenCVDistortion(NamedTuple):
    """
    Default OpenCV distortion model, also cf.known as Brown-Conrady.

    http://docs.opencv.org/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html

    This model only uses radial distortion up to k3, and tangential
    distortion up to p2.
    """

    k1: float
    k2: float
    k3: float
    p1: float
    p2: float

    def evaluate(self: Sequence[float], p: np.ndarray) -> np.ndarray:
        assert p.shape[-1] == 2
        k1, k2, k3, p1, p2 = self
        x, y = p[..., 0], p[..., 1]
        x2 = x * x
        y2 = y * y
        xy = x * y
        r2 = x2 + y2
        r4 = r2 * r2
        r6 = r2 * r4
        radial = 1 + k1 * r2 + k2 * r4 + k3 * r6
        x_out = x * radial + 2 * p1 * xy + p2 * (r2 + 2 * x2)
        y_out = y * radial + 2 * p2 * xy + p1 * (r2 + 2 * y2)
        return np.stack((x_out, y_out), axis=-1)


class OV62Distortion(NamedTuple):
    """
    OVRFisheye62 model, with 6 radial and 2 tangential coeffs.
    """

    k1: float
    k2: float
    k3: float
    k4: float
    p1: float
    p2: float
    k5: float
    k6: float

    def evaluate(self: Sequence[float], p: np.ndarray) -> np.ndarray:
        k1, k2, k3, k4, p1, p2, k5, k6 = self
        # radial component
        r2 = (p * p).sum(axis=-1, keepdims=True)
        r2 = np.clip(r2, -np.pi**2, np.pi**2)
        # we compute r^n using previous terms
        # instead of r^n = r ** n
        r4 = r2 * r2
        r6 = r2 * r4
        r8 = r4 * r4
        r10 = r4 * r6
        r12 = r6 * r6
        radial = 1 + k1 * r2 + k2 * r4 + k3 * r6 + k4 * r8 + k5 * r10 + k6 * r12
        uv = p * radial

        # tangential component
        x, y = uv[..., 0], uv[..., 1]
        x2 = x * x
        y2 = y * y
        xy = x * y
        r2 = x2 + y2
        x += 2 * p2 * xy + p1 * (r2 + 2 * x2)
        y += 2 * p1 * xy + p2 * (r2 + 2 * y2)
        return np.stack((x, y), axis=-1)
