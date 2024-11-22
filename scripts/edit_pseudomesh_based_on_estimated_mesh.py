import torch
import trimesh
from sklearn.neighbors import KDTree
from argparse import ArgumentParser
from scripts.save_pseudomesh import write_simple_obj


def calc_new_vertices_position(alpha, normal, vec_1, vec_2, vertice_1):
    vertices = torch.bmm(
        alpha.permute(0, 2, 1),torch.stack((normal, vec_1, vec_2), dim=1)
    ).reshape(-1, 3)  + vertice_1
    return vertices

def transform_pseudomesh_based_on_mesh(
    psuedomesh, mesh, mesh_edited, save_dir, scale,
    save_psuedomesh_edited_triangles = True
):

    # pseudomesh transformation based on mesh use triangle
    psuedomesh_triangles = torch.tensor(psuedomesh.triangles).cuda().float()
    mesh_triangles = torch.tensor(mesh.triangles).cuda().float()
    mesh_edited_triangles = torch.tensor(mesh_edited.triangles).cuda().float()

    # find the closest face (triangle)
    tree = KDTree(torch.mean(mesh_triangles, dim = 1).cpu())
    index_of_closest = tree.query(
        torch.mean(psuedomesh_triangles, dim = 1).cpu(), k = 1, return_distance = False
    )
    closest_triangle = mesh_triangles[index_of_closest.flatten()]

    # vertices of the closest face from references mesh to init psuedomesh
    v1 = closest_triangle[:, 0,:]
    v2 = closest_triangle[:, 1,:]
    v3 = closest_triangle[:, 2,:]

    v2_v1 = v2 - v1
    v3_v1 = v3 - v1

    normal = torch.cross(v2_v1,v3_v1)
    v2_v1 = v2_v1 / torch.linalg.vector_norm(v2_v1, dim=-1, keepdim=True)
    v3_v1 = v3_v1 / torch.linalg.vector_norm(v3_v1, dim=-1, keepdim=True)
    normal = normal / torch.linalg.vector_norm(normal, dim=-1, keepdim=True)
    A_T = torch.stack([normal, v2_v1, v3_v1]).permute(1, 2, 0)


    # vertices psuedomesh
    w1 = psuedomesh_triangles[:, 0,:]
    w2 = psuedomesh_triangles[:, 1,:]
    w3 = psuedomesh_triangles[:, 2,:]

    # calculate alpha
    alpha_w1 = torch.linalg.solve(A_T, w1 - v1).reshape(A_T.shape[0],3,1)
    alpha_w2 = torch.linalg.solve(A_T, w2 - v1).reshape(A_T.shape[0],3,1)
    alpha_w3 = torch.linalg.solve(A_T, w3 - v1).reshape(A_T.shape[0],3,1)
    alpha_w3.permute(0, 2, 1)

    # find referenced triangle based on edited mesh and init mesh
    referenced_triangle = mesh_edited_triangles[index_of_closest.flatten()]

    v1_referenced = referenced_triangle[:, 0,:]
    v2_referenced = referenced_triangle[:, 1,:]
    v3_referenced = referenced_triangle[:, 2,:]

    referenced_v2_v1 = v2_referenced - v1_referenced
    referenced_v3_v1 = v3_referenced - v1_referenced
    normal = torch.cross(referenced_v2_v1, referenced_v3_v1)

    # norm
    referenced_v2_v1 = referenced_v2_v1 / torch.linalg.vector_norm(referenced_v2_v1, dim=-1, keepdim=True)
    referenced_v3_v1 = referenced_v3_v1 / torch.linalg.vector_norm(referenced_v3_v1, dim=-1, keepdim=True)
    normal = normal / torch.linalg.vector_norm(normal, dim=-1, keepdim=True)


    # calculate new vertices of edited psuedomesh
    w1_edited = calc_new_vertices_position(alpha_w1, normal, referenced_v2_v1, referenced_v3_v1, v1_referenced)
    w2_edited = calc_new_vertices_position(alpha_w2, normal, referenced_v2_v1, referenced_v3_v1, v1_referenced)
    w3_edited = calc_new_vertices_position(alpha_w3, normal, referenced_v2_v1, referenced_v3_v1, v1_referenced)

    # create edited psuedomesh
    psuedomesh_edited_triangles = torch.stack(
        [w1_edited, w2_edited, w3_edited]
    ).permute(1,0,2)

    if save_psuedomesh_edited_triangles:
        torch.save(psuedomesh_edited_triangles, f'{save_dir}/edited_triangles.pt')

    faces = torch.range(
        0, psuedomesh_edited_triangles.shape[0] * 3 - 1
    ).reshape(psuedomesh_edited_triangles.shape[0], 3)

    vertices = psuedomesh_edited_triangles.reshape(psuedomesh_edited_triangles.shape[0] * 3, 3)

    filename = f'{save_dir}/scale_{scale}_edited_cylinder.obj'
    write_simple_obj(mesh_v=(vertices * scale).detach().cpu().numpy(), mesh_f=faces, filepath=filename)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    parser.add_argument("--triangle_soup_path", type=str)
    parser.add_argument("--mesh_path", type=str)
    parser.add_argument("--edited_mesh_path", type=str)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--scale", default=1, type=int)
    args = parser.parse_args()

    # triangle_soup is interpreted as psuedomesh
    psuedomesh = trimesh.load(args.triangle_soup_path, force='mesh')

    # load referenced mesh and edited mesh
    mesh = trimesh.load(args.mesh_path, force='mesh')
    mesh_edited = trimesh.load(args.edited_mesh_path,  force='mesh')

    transform_pseudomesh_based_on_mesh(
        psuedomesh, mesh, mesh_edited, args.save_dir, args.scale
    )