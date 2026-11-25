import struct

import numpy as np
import open3d as o3d


def write_android_mesh(ply_path, f32_path):
    mesh = o3d.io.read_triangle_mesh(str(ply_path))
    if not mesh.has_triangles():
        raise ValueError("Expected a mesh with triangles")

    faces = np.asarray(mesh.triangles)
    vertices = np.empty((len(faces), 3, 8), dtype="<f4")
    vertices[:, :, :3] = np.asarray(mesh.vertices)[faces]
    vertices[:, :, 3:6] = (
        np.asarray(mesh.vertex_colors)[faces]
        if mesh.has_vertex_colors() else 0.75
    )
    vertices[:, :, 6:] = ((1, 0), (0, 1), (0, 0))

    with f32_path.open("wb") as out:
        out.write(struct.pack("<i", vertices.size))
        vertices.tofile(out)