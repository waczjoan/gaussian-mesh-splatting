import numpy as np
import torch
from typing import NamedTuple


class FLAMEPointCloud(NamedTuple):
    alpha: torch.Tensor
    points: torch.Tensor
    colors: np.array
    normals: np.array
    faces: torch.Tensor
    flame_model: object
    transform_vertices_function: object
    flame_model_shape_init: torch.Tensor
    flame_model_expression_init: torch.Tensor
    flame_model_pose_init: torch.Tensor
    flame_model_neck_pose_init: torch.Tensor
    flame_model_transl_init: torch.Tensor


