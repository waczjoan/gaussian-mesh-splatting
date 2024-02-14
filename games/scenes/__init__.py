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

from scene.dataset_readers import (
    readColmapSceneInfo,
    readNerfSyntheticInfo,
)
from games.mesh_splatting.scene.dataset_readers import (
    readNerfSyntheticMeshInfo
)
from games.multi_mesh_splatting.scene.dataset_readers import (
    readColmapMeshSceneInfo
)
from games.flame_splatting.scene.dataset_readers import (
    readNerfSyntheticFlameInfo
)

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Colmap_Mesh": readColmapMeshSceneInfo,
    "Blender": readNerfSyntheticInfo,
    "Blender_Mesh": readNerfSyntheticMeshInfo,
    "Blender_FLAME": readNerfSyntheticFlameInfo
}
