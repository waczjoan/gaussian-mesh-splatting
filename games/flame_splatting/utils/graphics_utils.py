#
# Copyright (C) 2024, Gmum
# Group of Machine Learning Research. https://gmum.net/
# All rights reserved.
#
# The Gaussian-splatting software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
# For inquiries contact  george.drettakis@inria.fr
#
# The Gaussian-mesh-splatting is software based on Gaussian-splatting, used on research.
# This Games software is free for non-commercial, research and evaluation use
#

import numpy as np
import torch
from typing import NamedTuple


class FLAMEPointCloud(NamedTuple):
    alpha: torch.Tensor
    points: torch.Tensor
    colors: np.array
    normals: np.array
    faces: torch.Tensor
    vertices_init: torch.Tensor
    flame_model: object
    transform_vertices_function: object
    flame_model_shape_init: torch.Tensor
    flame_model_expression_init: torch.Tensor
    flame_model_pose_init: torch.Tensor
    flame_model_neck_pose_init: torch.Tensor
    flame_model_transl_init: torch.Tensor
    vertices_enlargement_init: float


