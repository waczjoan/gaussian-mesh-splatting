import numpy as np
import torch
from typing import NamedTuple


class MeshPointCloud(NamedTuple):
    alpha: torch.Tensor
    points: torch.Tensor
    colors: np.array
    normals: np.array
    vertices: np.array
    faces: np.array
    triangles: torch.Tensor

