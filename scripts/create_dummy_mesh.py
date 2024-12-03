import os
import trimesh
import numpy as np
import open3d as o3d
import torch
from argparse import ArgumentParser

def reconstruction(xyz, save_dir, alpha=0.003):
    """Given a 3D point cloud, get unstructured mesh using Alpha shapes
    
    Based on this stack overflow code snippet:
    https://www.open3d.org/docs/latest/tutorial/Advanced/surface_reconstruction.html
    https://stackoverflow.com/questions/56965268/how-do-i-convert-a-3d-point-cloud-ply-into-a-mesh-with-faces-and-vertices
    
    Parameters
    ----------
    xyz: [n_points, 3]
        input point cloud, numpy array
    
    Returns
    -------
    mesh: trimesh Mesh object with verts, faces and normals of the mesh
    
    """
    
    # estimate normals first
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

        
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    tri_mesh = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles),
                                vertex_normals=np.asarray(mesh.vertex_normals))

    trimesh.exchange.export.export_mesh(tri_mesh, f'{save_dir}/mesh_alpha_0_003.obj')
    
    return mesh


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    parser.add_argument("--pseudomesh_path",  type=str)
    parser.add_argument("--scale", default=2, type=int)
    parser.add_argument("--alpha", default=0.003, type=float)
    args = parser.parse_args()

    p = torch.load(f'{args.pseudomesh_path}')
    save_dir = os.path.dirname(args.pseudomesh_path)
    faces = torch.range(0, p.shape[0] * 3 - 1).reshape(p.shape[0],3)
    vertice = p.reshape(p.shape[0] * 3, 3) * args.scale

    xyz = vertice.detach().cpu().numpy()
    reconstruction(xyz, save_dir, args.alpha)



