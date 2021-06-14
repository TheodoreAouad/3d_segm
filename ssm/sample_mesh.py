from typing import List

import numpy as np
from tqdm import tqdm

from .dijkstra import Graph, dijkstra


def create_mesh_graph(verts: np.ndarray, faces: np.ndarray) -> Graph:
    gmesh = Graph()
    for tri in faces:
        for i in range(2):
            for j in range(i + 1, 3):
                gmesh.add_node(tri[i])
                gmesh.add_node(tri[j])
                value = np.linalg.norm(verts[tri[i]] - verts[tri[j]])
                gmesh.add_edge(tri[i], tri[j], value)
                gmesh.add_edge(tri[j], tri[i], value)
    return gmesh


def dijkstra_sampling(
    verts: np.ndarray, faces: np.ndarray, n_points: int, verbose=True
) -> (np.ndarray, np.ndarray, np.ndarray):

    gmesh = create_mesh_graph(verts, faces)

    all_points = []
    cur_point = 0

    iterator = range(n_points)
    if verbose:
        iterator = tqdm(iterator)
    for _ in tqdm(iterator):
        all_points.append(cur_point)
        visited, path, closest = dijkstra(gmesh, initial_set=all_points)
        ar_dist = np.zeros(len(gmesh.nodes))
        for node, dist in visited.items():
            ar_dist[node] = dist
        cur_point = ar_dist.argmax()

    ar_clos = np.zeros(len(gmesh.nodes))
    for node, origin in closest.items():
        ar_clos[node] = origin

    return all_points, ar_dist, ar_clos


def dijkstra_mesh(verts: np.ndarray, faces: np.ndarray, initial_set: List[int]) -> np.ndarray:
    """
    Compute minimal distances for each point of the verts to the initial set.
    Args:
        verts (np.ndarray): size (Nxd), vertexes of the mesh
        faces (np.ndarray): size (Mx3), triangles linking indexes of the verts
        initial_set (list): indexes of the verts for initial points

    Returns:
        np.ndarray: size (N,), distance of each vertex to initial_set
    """
    gmesh = create_mesh_graph(verts, faces)
    visited, path, closest = dijkstra(gmesh, initial_set=initial_set)
    ar_dist = np.zeros(len(gmesh.nodes))
    for key, value in visited.items():
        ar_dist[key] = value

    ar_clos = np.zeros(len(gmesh.nodes))
    for node, origin in closest.items():
        ar_clos[node] = origin


    return ar_dist, ar_clos
