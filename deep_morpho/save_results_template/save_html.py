
from typing import List, Dict

import webbrowser
import os
from os.path import join

import torch
import matplotlib.pyplot as plt

from .utils import detect_identical_values, plot_to_html
from general.utils import load_yaml


def html_template():
    return """
    <!DOCTYPE html>
    <html>
      <head>
        <title>{title}</title>
      </head>
      <body>
        <h2>Tensorboard paths</h2>
        <p>{tb_paths}</p>
        <h2>Global Args</h2>
        <p>{global_args}</p>
        <h2>Changing args</h2>
        <p>{changing_args}</p>
        <h2>Results</h2>
        <span>{results}</span>
      </body>
    </html>
    """


def get_results_from_tensorboard(tb_path: str):

    weights = []
    normalized_weights = []

    for file_ in os.listdir(join(tb_path, "observables", "PlotWeightsBiSE")):
        fig_path = join(tb_path, "observables", "PlotWeightsBiSE", file_)
        if "normalized" in file_:
            normalized_weights.append(plt.imread(fig_path))
        else:
            weights.append(plt.imread(fig_path))


    return {
        "args": None,
        "tb_path": tb_path,
        "weights": weights,
        "normalized_weights": normalized_weights,
        "bias": None,
        "dice": None,
        "baseline_dice": None,
        "convergence_dice": None,
        "activation_P": None,
        "learned_selem": None,
        "convergence_layer": None,
    }


def write_html_deep_morpho(results_dict: List[Dict], save_path: str, title: str = "",):
    html = html_template()

    tb_paths = [res["tb_path"] for res in results_dict]

    global_args, changing_args = detect_identical_values(results_dict)

    results_html = ""

    for i, results in enumerate(results_dict):
        results_html += (
            f"<div>"
            f"<h3>{results['tb_path']}</h3>"  # tb
            f"<p>{dict({k: results['tb_path'][k] for k in changing_args})}</p>"  # args
        )

        results_html += ''.join([f"<span>{plot_to_html(fig)}</span>" for fig in results['normalized_weights']])
        results_html += (
            f"<p>dice={results['dice']}  baseline={results['baseline_dice']}  step until convergence (dice)={results['convergence_dice']}</p>"
            "<p>learned selems: "
        )

        results_html += ' '.join([
            f"{layer_idx} <span>{plot_to_html(fig)}</span> cvg={results['convergence_layer'][layer_idx]}"
            for layer_idx, fig in results['learned_selem'].items()])

        results_html += "</p></div>"


    html = html.format(
        title=title,
        tb_paths=tb_paths,
        global_args=global_args,
        changing_args=changing_args,
        results=results_html,
    )

    with open(save_path, "w") as f:
        f.write(html)
    return html


def save_html(results_dict: List[Dict]):
    def filter_params(params):
        filtered = []
        for k, p in params.items():
            if torch.is_tensor(p):
                clean_p = f"Tensor {p.shape}"
            # elif callable(p):
            #     clean_p = p.__qualname__
            else:
                clean_p = p
            filtered.append(f"{k}: {clean_p}")
        return filtered

    html = html_template()
    # Put results in an html file to keep
    results_html = ""
    for i, (algo, params) in enumerate(run_params.items()):
        results_html += f"<div><h3>{algo}</h3><p>{str(filter_params(params))}</p><span>{denoising_figures[algo]}</span></div>"
    results_html += f"<div><h3>Recap</h3>{''.join([f'<span>{fig}</span>' for fig in recap_figures])}</div>"
    html = html.format(title="Denoising with tseng Algorithm",
                       parameters=f"Gaussian Nosise N({parameters.NOISE_MEAN}, {parameters.NOISE_STD}) Gaussian Blur sigma={parameters.BLUR_STD} Size={parameters.KERNEL_SIZE} Data fidelity p={parameters.DATA_FIDELITY_P}",
                       results=results_html)

    save_path = parameters.RESULTS_PATH / f"results_{len(list(parameters.RESULTS_PATH.glob('*')))}.html"
    print(f"Saving html {save_path}")
    with open(save_path, "w") as f:
        f.write(html)
    webbrowser.open(str(save_path))
