""" Taken from https://gist.github.com/econchick/4666413
"""
from collections import defaultdict


class Graph:
    def __init__(self):
        self.nodes = set()
        self.edges = defaultdict(set)
        self.distances = {}

    def add_node(self, value):
        self.nodes.add(value)

    def add_edge(self, from_node, to_node, distance):
        self.edges[from_node].add(to_node)
        self.edges[to_node].add(from_node)
        self.distances[(from_node, to_node)] = distance


def dijkstra(graph, initial_set):
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
