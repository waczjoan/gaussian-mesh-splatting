import numpy as np
import torch
from typing import NamedTuple


class FLAMEPointCloud(NamedTuple):
    alpha: torch.Tensor
    points: torch.Tensor
    colors: np.array
    normals: np.array
    flame_model: object
    transform_vertices_function: object

