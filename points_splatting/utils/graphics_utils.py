import numpy as np
import torch
from typing import NamedTuple


class PointCloud(NamedTuple):
    alpha: torch.Tensor
    points: torch.Tensor
    colors: np.array
    normals: np.array
    selected_points: torch.Tensor
    referents_idx: torch.Tensor
    referents_points: torch.Tensor

