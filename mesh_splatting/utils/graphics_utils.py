import numpy as np
from typing import NamedTuple


class MeshPointCloud(NamedTuple):
    alpha: np.array
    colors: np.array
    normals: np.array
    vertices: np.array
    faces: np.array

