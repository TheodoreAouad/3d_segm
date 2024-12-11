import networkx as nx
import numpy as np


def get_n_cliques(graph: nx.Graph, n: int) -> np.ndarray:
    """
    Get the n cliques in the graph.

    Args:
        graph: write your description
        nx: write your description
        Graph: write your description
        n: write your description
    """
    return np.array([clique for clique in nx.enumerate_all_cliques(graph) if len(clique) == n])


def get_nx_mesh(faces: np.ndarray) -> nx.Graph:
    """
    Returns a networkx graph representation of the mesh.

    Args:
        faces: write your description
        np: write your description
        ndarray: write your description
    """
    graph = nx.Graph()
    for tri in faces:
        for i in range(2):
            for j in range(i + 1, 3):
                # value = np.linalg.norm(self.vertices[tri[i]] - self.vertices[tri[j]])
                graph.add_edge(tri[i], tri[j])
                graph.add_edge(tri[j], tri[i])
    return graph
