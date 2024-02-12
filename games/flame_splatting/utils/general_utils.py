import torch


def write_mesh_obj(
        vertices: torch.tensor,
        faces: torch.tensor,
        filepath,
        verbose=False
):
    """Simple save vertices and face as an obj file."""
    vertices = vertices.detach().cpu().numpy()
    with open(filepath, 'w') as fp:
        for v in vertices:
            fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
        for f in faces + 1:  # Faces are 1-based, not 0-based in obj files
            fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))
    if verbose:
        print('mesh saved to: ', f'{filepath}.obj')
