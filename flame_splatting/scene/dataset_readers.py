import os
import numpy as np
import trimesh
import torch

from flame_splatting.utils.graphics_utils import FLAMEPointCloud
from scene.dataset_readers import (
    readColmapSceneInfo,
    readNerfSyntheticInfo,
    readCamerasFromTransforms,
    getNerfppNorm,
    SceneInfo,
    storePly,
    fetchPly
)
from utils.sh_utils import SH2RGB
from flame_splatting.FLAME import FLAME
from flame_splatting.FLAME.config import FlameConfig

softmax = torch.nn.Softmax(dim=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def transform_vertices_function(vertices):
    vertices = vertices[:, [0, 2, 1]]
    vertices[:, 1] = -vertices[:, 1]
    vertices *= 3
    return vertices


def readNerfSyntheticFlameInfo(
        path, white_background, eval, extension=".png"
):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    print("Reading Mesh object")

    flame_config = FlameConfig()
    model_flame = FLAME(flame_config).to(device)

    vertices = mesh_scene.vertices
    faces = mesh_scene.faces

    triangles = torch.tensor(mesh_scene.triangles).float()  # equal vertices[faces]

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    # if not os.path.exists(ply_path):
    if True:
        # Since this data set has no colmap data, we start with random points
        num_pts_each_triangle = 50
        num_pts = num_pts_each_triangle * triangles.shape[0]
        print(
            f"Generating random point cloud ({num_pts})..."
        )

        # We create random points inside the bounds traingles
        alpha = torch.rand(
            triangles.shape[0],
            num_pts_each_triangle,
            3
        )

        xyz = torch.matmul(
            alpha,
            triangles
        )
        xyz = xyz.reshape(num_pts, 3)

        shs = np.random.random((num_pts, 3)) / 255.0

        pcd = FLAMEPointCloud(
            alpha=alpha,
            points=xyz,
            colors=SH2RGB(shs),
            normals=np.zeros((num_pts, 3)),
            flame_model=flame_model,
            transform_vertices_function=transform_vertices_function,
        )

        storePly(ply_path, pcd.points, SH2RGB(shs) * 255)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender": readNerfSyntheticInfo,
    "Blender_Mesh": readNerfSyntheticMeshInfo
}
