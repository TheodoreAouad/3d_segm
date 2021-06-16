""" Taken from https://gist.github.com/econchick/4666413
"""
from typing import Set, Dict
from collections import defaultdict

from numpy import infty


class Graph:
    def __init__(self):
        self.nodes = set()
        self.edges = defaultdict(set)
        self.distances = {}

    def add_node(self, value):
        self.nodes.add(value)

    def add_edge(self, from_node, to_node, distance=1):
        self.edges[from_node].add(to_node)
        self.edges[to_node].add(from_node)
        self.distances[(from_node, to_node)] = distance


def dijkstra_old(graph, initial_set):
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


def dijkstra(
    graph: Graph,
    initial_set: Set,
    initial_dist: Dict = None,
    initial_closest: Dict = None,
    initial_path: Dict = None
) -> (Dict, Dict, Dict):
    front = set()
    closest = initial_closest if initial_closest is not None else dict()
    path = initial_path if initial_path is not None else dict()
    if initial_dist is None:
        dists = {node: infty for node in graph.nodes}
    else:
        dists = initial_dist

    for node in initial_set:
        dists[node] = 0
        closest[node] = node
        front.add(node)


    while len(front) != 0:
        min_node, min_dist = None, infty
        for node in front:
            if dists[node] <= min_dist:
                min_node, min_dist = node, dists[node]

        front.remove(min_node)
        for neigh in graph.edges[min_node]:
            new_dist = dists[min_node] + graph.distances[(min_node, neigh)]
            if dists[neigh] > new_dist:
                dists[neigh] = new_dist
                path[neigh] = min_node
                closest[neigh] = closest[min_node]
                front.add(neigh)

    return dists, path, closest
