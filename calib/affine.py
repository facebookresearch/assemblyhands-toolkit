# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""
Helpers for affine transformation matrix operations.
"""
import numpy as np


def as_4x4(a: np.ndarray, *, copy: bool = False) -> np.ndarray:
    """
    Append [0,0,0,1] to convert 3x4 matrices to a 4x4 homogeneous matrices

    If the matrices are already 4x4 they will be returned unchanged.
    """
    if a.shape[-2:] == (4, 4):
        if copy:
            a = np.array(a)
        return a
    if a.shape[-2:] == (3, 4):
        return np.concatenate(
            (
                a,
                np.broadcast_to(
                    np.array([0, 0, 0, 1], dtype=a.dtype), a.shape[:-2] + (1, 4)
                ),
            ),
            axis=-2,
        )
    raise ValueError("expected 3x4 or 4x4 affine transform")


def transform_vec3(m, v):
    """
    Transform an array of 3D vectors with an affine transform.

    This ignores the translation in `m`; to transform 3D *points*, use
    `tranform3()` instead.

    Note that we specifically optimize for ndim=2, which is a frequent
    use case, for better performance. See n388920 for the comparison.

    Parameters
    ----------
    m
        affine transform(s) as 3x4 or 4x4 matrix
    v
        3d vectors

    m or v can be batched as long as the batch shapes are broadcastable.
    """
    v = np.asarray(v)
    if m.ndim == 2:
        return (v.reshape(-1, 3) @ m[:3, :3].T).reshape(v.shape)
    else:
        return (m[..., :3, :3] @ v[..., None]).squeeze(-1)


def transform3(m, v):
    """
    Transform an array of 3D points with an affine transform.

    Parameters
    ----------
    m
        affine transform(s) as 3x4 or 4x4 matrix
    v
        Array of 3d points

    m or v can be batched as long as the batch shapes are broadcastable.
    """
    v = np.asarray(v)
    return transform_vec3(m, v) + m[..., :3, 3]


def normalized(v: np.ndarray, axis: int = -1, eps: float = 5.43e-20) -> np.ndarray:
    """
    Return a unit-length copy of vector(s) v

    Parameters
    ----------
    axis : int = -1
        Which axis to normalize on

    eps
        Epsilon to avoid division by zero. Vectors with length below
        eps will not be normalized. The default is 2^-64, which is
        where squared single-precision floats will start to lose
        precision.
    """
    d = np.maximum(eps, (v * v).sum(axis=axis, keepdims=True) ** 0.5)
    return v / d
