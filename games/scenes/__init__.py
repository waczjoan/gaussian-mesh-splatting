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
