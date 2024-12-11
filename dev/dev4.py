# %%
from importlib import reload
import os

import numpy as np
from tqdm import tqdm
import networkx as nx


import ssm.utils as su
import ssm.sample_mesh as sm
import ssm.shape as sh
import ssm.pyqtgraph_mesh as pm


def reload_modules():
    """
    Reload all modules

    Args:
    """
    for modl in [su, sh, sm, pm]:
        reload(modl)


# %%
# Using shape class

reload_modules()
mesh_dir = "/hdd/datasets/CT-ORG/meshes/labels/"
step_size = 4
n_samples = 100

all_meshes = su.sort_by_regex([
    os.path.join(mesh_dir, filename, f"step_size_{step_size}") for filename in os.listdir(mesh_dir)
])
ref_mesh = all_meshes[2]
print("Reference:", ref_mesh)
ref_verts = np.load(os.path.join(ref_mesh, "vertexes.npy"))
ref_faces = np.load(os.path.join(ref_mesh, "faces.npy"))

ref_shape = sh.Shape(label=ref_mesh, vertexes=ref_verts, faces=ref_faces)
ref_shape.perform_sampling(n_samples)

# %%

n_shapes = 10
all_shapes = []
all_errs = []
all_n_iters = []

iterator = su.sort_by_regex(list(set(all_meshes).difference([ref_mesh])))[:n_shapes]
for idx in tqdm(range(len(iterator))):
    mesh_path = iterator[idx]
    verts = np.load(os.path.join(mesh_path, 'vertexes.npy'))
    faces = np.load(os.path.join(mesh_path, 'faces.npy'))
    cur_shape = sh.Shape(label=mesh_path, vertexes=verts, faces=faces)
    cur_shape.set_reference(ref_shape)
    _, errs, n_iters = cur_shape.register_icp_to_reference()
    cur_shape.match_samples()

    all_shapes.append(cur_shape)
    all_errs.append(errs)
    all_n_iters.append(n_iters)


# %%

shape_to_plots = [all_shapes[4]]
color_vertices = []
closest_samples = []
verts_mean = []


def min_max_norm(ar):
    """
    Minimum and maximum norm of an array.

    Args:
        ar: write your description
    """
    return (ar - ar.min()) / (ar.max() - ar.min())


for idx in range(len(shape_to_plots)):
    shape_to_plots[idx].dijkstra_to_sample()
    colors = min_max_norm(shape_to_plots[idx].dist_to_sample)
    color_vertices.append(np.stack((colors, np.zeros_like(colors), 1 - colors, np.ones_like(colors)), axis=-1))

    col_closest = min_max_norm(shape_to_plots[idx].closest_sample_point)
    closest_samples.append(np.stack((col_closest, np.zeros_like(col_closest), 1 - col_closest, np.ones_like(col_closest)), axis=-1))

    verts_mean.append(shape_to_plots[idx].vertexes.mean(0))


# %%

# gsample = nx.Graph()
# sample_order = {shape_to_plots[0].sample_idx[k]: k for k in range(len(shape_to_plots[0].sample_idx))}

# gmesh = sm.create_mesh_graph(shape_to_plots[0].vertexes, shape_to_plots[0].faces)
# closest = shape_to_plots[0].closest_sample_point
# fronts = set()
# visited_edges = set()
# for node1 in gmesh.nodes:
#     for node2 in gmesh.edges[node1]:
#         if (node2, node1) in visited_edges:
#             continue
#         visited_edges.add((node1, node2))
#         c1 = closest[node1]
#         c2 = closest[node2]
#         if c1 != c2:
#             fronts.add(node1)
#             gsample.add_edge(sample_order[c1], sample_order[c2])

# fronts = list(fronts)
# faces_sample = np.array([clique for clique in nx.enumerate_all_cliques(gsample) if len(clique) == 3])

shape_to_plots[0].compute_sample_faces()


# %%
reload_modules()
# pm.plot_mesh(shape_to_plot.vertexes - verts_mean, shape_to_plot.faces, color_vertices, points=shape_to_plot.sample - verts_mean)
plotter = pm.PlotMesh(title='SSM')

translation = - verts_mean[0] - 50
plotter.add_mesh(
    vertexes=shape_to_plots[0].sample + translation, faces=shape_to_plots[0].faces_sample,
    # edges=[[a, b] for a in range(n_samples) for b in range(n_samples)],
    # vertexColors=su.random_color_generator(len(shape_to_plots[0].sample)),
    faceColors=su.random_color_generator(len(shape_to_plots[0].faces_sample)),
    drawEdges=True,
)
plotter.add_point_cloud(shape_to_plots[0].sample + translation)

translation = - verts_mean[0] + 50
plotter.add_mesh(
    vertexes=shape_to_plots[0].vertexes  + translation,
    edges=plotter.meshes[0].edges,
    faces=shape_to_plots[0].faces,
    vertexColors=color_vertices[0]
)
plotter.add_point_cloud(shape_to_plots[0].sample + translation)
# plotter.add_point_cloud(shape_to_plots[0].vertexes[fronts] + translation, size_point=5, point_color=[.5, .5, 0, 1])



# plotter.add_mesh(shape_to_plots[0].vertexes - verts_mean[0] + translation, shape_to_plots[0].faces, color_vertexes=closest_samples[0])
# plotter.add_point_cloud(shape_to_plots[0].sample - verts_mean[0] + translation)
# plotter.add_point_cloud(shape_to_plots[0].vertexes[fronts] - verts_mean + translation, point_color=[1, 0, 0, 1])

# plotter.add_mesh(shape_to_plots[1].vertexes - verts_mean[1] + 100 * np.ones(3), shape_to_plots[1].faces, color_vertexes=color_vertices[1])
# plotter.add_point_cloud(shape_to_plots[1].sample - verts_mean[1]  + 100 * np.ones(3))


plotter.show()



