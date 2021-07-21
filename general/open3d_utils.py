from typing import Union, List, Callable

import open3d as o3d
# from open3d.web_visualizer import draw
import numpy as np


def numpy_to_o3d_pcd(points: "np.ndarray", **kwargs) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    for attr, value in kwargs.items():
        type_fn = o3d.utility.Vector3dVector
        setattr(pcd, attr, type_fn(value))

    return pcd


def get_o3d_pcd_colored(points: "np.ndarray", color: Union[List, str] = 'g', **kwargs):
    cur_pcd = numpy_to_o3d_pcd(points, **kwargs)

    if color is not None:
        if isinstance(color, str):
            if color in ['r', 'red']:
                color = [1, 0, 0]
            elif color in ['g', 'green']:
                color = [0, 1, 0]
            elif color in ['b', 'blue']:
                color = [0, 0, 1]
        cur_pcd.paint_uniform_color(color)

    return cur_pcd


def numpy_to_o3d_mesh(**kwargs) -> o3d.geometry.TriangleMesh:
    msh = o3d.geometry.TriangleMesh()
    # msh.vertices = o3d.utility.Vector3dVector(verts)
    # msh.triangles = o3d.utility.Vector3iVector(faces)
    # if vertex_normals is not None:
    #     msh.vertex_normals = o3d.utility.Vector3dVector(vertex_normals)
    for attr, value in kwargs.items():
        if attr == "triangles":
            type_fn = o3d.utility.Vector3iVector
        else:
            type_fn = o3d.utility.Vector3dVector
        setattr(msh, attr, type_fn(value))
    return msh


def plot_mesh(vertices: "np.ndarray", triangles: "np.ndarray", draw_fn: Callable, **kwargs) -> None:
    """ We give argument 'draw_fn' instead of the draw from open3d.web_visualizer because other it crashes.
    In a Notebook, give the function open3d.web_visualizer.draw
    """
    msh = numpy_to_o3d_mesh(vertices=vertices, triangles=triangles, **kwargs)
    # msh.compute_triangle_normals()
    msh.compute_vertex_normals()
    draw_fn(msh)


def ball_pivoting_mesh(pcd: o3d.geometry.PointCloud, scalar: float = 3) -> o3d.geometry.TriangleMesh:
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = scalar * avg_dist
    if not pcd.has_normals():
        pcd.estimate_normals()

    bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector([radius, radius]))
    bpa_mesh.compute_vertex_normals()
    return bpa_mesh
