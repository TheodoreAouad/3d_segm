""" Taken from https://gist.github.com/econchick/4666413
"""
from collections import defaultdict


class Graph:
    def __init__(self):
        """
        Initialize the graph.

        Args:
            self: write your description
        """
        self.nodes = set()
        self.edges = defaultdict(set)
        self.distances = {}

    def add_node(self, value):
        """
        Add a node to the graph.

        Args:
            self: write your description
            value: write your description
        """
        self.nodes.add(value)

    def add_edge(self, from_node, to_node, distance):
        """
        Add an edge between two nodes.

        Args:
            self: write your description
            from_node: write your description
            to_node: write your description
            distance: write your description
        """
        self.edges[from_node].add(to_node)
        self.edges[to_node].add(from_node)
        self.distances[(from_node, to_node)] = distance


def dijkstra(graph, initial_set):
    """
    Dijkstra s algorithm for finding shortest paths.

    Args:
        graph: write your description
        initial_set: write your description
    """
    visited = {}
    closest = {}
    for initial in initial_set:
        visited[initial] = 0
        closest[initial] = initial
    path = {}

    nodes = set(graph.nodes)

    while nodes:
        min_node = None
        for node in nodes:
            if node in visited:
                if min_node is None:
                    min_node = node
                elif visited[node] < visited[min_node]:
                    min_node = node

        if min_node is None:
            break

        nodes.remove(min_node)
        current_weight = visited[min_node]

        for dest_node in graph.edges[min_node]:
            weight = current_weight + graph.distances[(min_node, dest_node)]
            if dest_node not in visited or weight < visited[dest_node]:
                visited[dest_node] = weight
                path[dest_node] = min_node
                closest[dest_node] = closest[min_node]

    return visited, path, closest
