import numpy as np
from tqdm import tqdm

from .dijkstra import Graph, dijsktra


def dijkstra_sampling(verts: np.ndarray, faces: np.ndarray, n_points: int, verbose=True) -> (np.ndarray, np.ndarray):
    gmesh = Graph()
    for tri in faces:
        for i in range(2):
            for j in range(i + 1, 3):
                gmesh.add_node(tri[i])
                gmesh.add_node(tri[j])
                value = np.linalg.norm(verts[tri[i]] - verts[tri[j]])
                gmesh.add_edge(tri[i], tri[j], value)
                gmesh.add_edge(tri[j], tri[i], value)


    all_points = []
    cur_point = 0

    iterator = range(n_points)
    if verbose:
        iterator = tqdm(iterator)
    for _ in tqdm(iterator):
        all_points.append(cur_point)
        visited, path = dijsktra(gmesh, initial_set=all_points)
        ar_dist = np.zeros(len(gmesh.nodes))
        for key, value in visited.items():
            ar_dist[key] = value
        cur_point = ar_dist.argmax()

    return verts[all_points], ar_dist
