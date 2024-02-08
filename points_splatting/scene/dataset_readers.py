#
# Copyright (C) 2024, Krakow
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
#

import os
import numpy as np
import trimesh
import torch
from plyfile import PlyData
from sklearn.neighbors import NearestNeighbors

from points_splatting.utils.graphics_utils import PointCloud
from scene.dataset_readers import (
    readColmapSceneInfo,
    readNerfSyntheticInfo,
    readCamerasFromTransforms,
    getNerfppNorm,
    SceneInfo,
    storePly,
)
from mesh_splatting.scene.dataset_readers import (
    readNerfSyntheticMeshInfo
)
from flame_splatting.scene.dataset_readers import (
    readNerfSyntheticFlameInfo
)
from utils.sh_utils import SH2RGB

softmax = torch.nn.Softmax(dim=2)

def transform_vertices_function(vertices, c=1):
    vertices = vertices[:, [0, 2, 1]]
    vertices[:, 1] = -vertices[:, 1]
    vertices *= c
    return vertices

def load_points_cloud(path):
    plydata = PlyData.read(path)

    xyz_opacity = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"]),
                    np.asarray(plydata.elements[0]["opacity"])
                    ), axis=1)
    xyz_selected = xyz_opacity[xyz_opacity[:, -1] > 15]
    return torch.tensor(xyz_selected[:, :-1]).cuda()


def find_referents_points(points):
    pts = points.cpu().numpy()
    knn = NearestNeighbors(n_neighbors=10)
    knn.fit(pts)
    distance_mat, neighbours_mat = knn.kneighbors(pts)
    return torch.tensor(neighbours_mat).long().cuda()


def readNerfSyntheticPointsInfo(
        path, white_background, eval, num_splats, extension=".png"
):
    print("Reading Point cloud")
    """
    pc_path = "/media/joanna/DANE/uj/gs/fork/gaussian-splatting/output/hotdog/point_cloud/iteration_30000/point_cloud.ply"
    points = load_points_cloud(pc_path)
    k_nearest_idx = find_referents_points(points)
    torch.save(points, 'vertices_v2')
    torch.save(k_nearest_idx, 'faces_v2')
    """
    mesh_scene = trimesh.load(f'{path}/mesh.obj', force='mesh')
    vertices = mesh_scene.vertices
    vertices = transform_vertices_function(
        torch.tensor(vertices),
    )
    faces = mesh_scene.faces
    torch.save(vertices, 'vertices_org.pt')
    torch.save(faces, 'faces_org.pt')

    k_nearest_idx = find_referents_points(vertices)
    torch.save(vertices, 'vertices_from_org_v3.pt')
    torch.save(k_nearest_idx, 'faces_knn_from_org_v3.pt')
    points = vertices


    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)

    referents_points = points[k_nearest_idx].float().cpu()

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    # if not os.path.exists(ply_path):
    if True:
        # Since this data set has no colmap data, we start with random points
        num_pts_each_triangle = num_splats
        num_pts = num_pts_each_triangle * referents_points.shape[0]
        print(
            f"Generating random point cloud ({num_pts})..."
        )

        # We create random points inside the bounds traingles
        alpha = torch.rand(
            referents_points.shape[0],
            num_pts_each_triangle,
            3
        )

        xyz = torch.matmul(
            alpha,
            referents_points
        )
        xyz = xyz.reshape(num_pts, 3)

        shs = np.random.random((num_pts, 3)) / 255.0

        pcd = PointCloud(
            alpha=alpha,
            points=xyz,
            colors=SH2RGB(shs),
            normals=np.zeros((num_pts, 3)),
            selected_points=points,
            referents_idx=k_nearest_idx,
            referents_points=referents_points.cuda()
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
    "Blender_Mesh": readNerfSyntheticMeshInfo,
    "Blender_FLAME": readNerfSyntheticFlameInfo,
    "Blender_Points": readNerfSyntheticPointsInfo
}
