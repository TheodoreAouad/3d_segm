
import re
from typing import List, Dict
import pathlib

import webbrowser
import os
from os.path import join

import torch

from .utils import detect_identical_values, plot_to_html, load_png_as_fig
from .load_args import load_args
from general.utils import load_json


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
    res = {
        "args": None,
        "tb_path": None,
        "weights": None,
        "normalized_weights": None,
        "bias": None,
        "dice": None,
        "baseline_dice": None,
        "convergence_dice": None,
        "activation_P": None,
        "learned_selem": None,
        "convergence_layer": None,
    }
    obs_path = join(tb_path, "observables")

    res['tb_path'] = tb_path
    if os.path.exists(join(tb_path, 'args.yaml')):
        res['args'] = load_args(join(tb_path, 'args.yaml'))

    weights = []
    normalized_weights = []

    for file_ in os.listdir(join(obs_path, "PlotWeightsBiSE")):
        fig_path = join(obs_path, "PlotWeightsBiSE", file_)
        if "normalized" in file_:
            normalized_weights.append(load_png_as_fig(fig_path))
        else:
            weights.append(load_png_as_fig(fig_path))
    res['weights'] = weights
    res['normalized_weights'] = normalized_weights

    file_parameters = join(obs_path, "PlotParametersBiSE", "parameters.json")
    if os.path.exists(file_parameters):
        parameters = load_json(file_parameters)
        for key in ['bias', 'activation_P']:
            res[key] = [parameters[layer_idx].get(key, None) for layer_idx in sorted(parameters.keys(), key=int)]

    file_convergence_binary = join(obs_path, "ConvergenceBinary", "convergence_step.json")
    if os.path.exists(file_convergence_binary):
        convergence_steps = load_json(file_convergence_binary)
        res['convergence_layer'] = [convergence_steps[layer_idx] for layer_idx in sorted(convergence_steps.keys(), key=int)]

    file_baseline = join(obs_path, "InputAsPredMetric", "baseline_metrics.json")
    if os.path.exists(file_baseline):
        res['baseline_dice'] = load_json(file_baseline)["dice"]

    file_metrics = join(obs_path, "CalculateAndLogMetrics", "metrics.json")
    if os.path.exists(file_metrics):
        res['dice'] = load_json(file_metrics)["dice"]

    file_convergence_metrics = join(obs_path, "ConvergenceMetrics", "convergence_step.json")
    if os.path.exists(file_convergence_metrics):
        res['convergence_dice'] = load_json(file_convergence_metrics)['train']['dice']

    file_learned_selem = join(obs_path, "ShowSelemBinary")
    if os.path.exists(file_learned_selem):
        learned_selem = {}
        for file_ in os.listdir(file_learned_selem):
            layer_idx = re.findall(r'layer_(\d+)}', file_)
            learned_selem[layer_idx] = load_png_as_fig(join(file_learned_selem, file_))
        res['learned_selem'] = learned_selem


    return res


def write_html_from_dict_deep_morpho(results_dict: List[Dict], save_path: str, title: str = "",):
    html = html_template()

    tb_paths = [res["tb_path"] for res in results_dict]

    global_args, changing_args = detect_identical_values([results['args'] for results in results_dict])

    results_html = ""

    for i, results in enumerate(results_dict):
        results_html += (
            f"<div>"
            f"<h3>{results['tb_path']}</h3>"  # tb
            f"<p>{dict({k: results[k] for k in changing_args})}</p>"  # args
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

    pathlib.Path(save_path).parent.mkdir(exist_ok=True, parents=True)
    with open(save_path, "w") as f:
        f.write(html)
    return html


def write_html_deep_morpho(tb_paths: List[str], save_path: str, title: str = ""):
    results_dict = [get_results_from_tensorboard(tb_path) for tb_path in tb_paths]
    return write_html_from_dict_deep_morpho(results_dict, save_path, title)


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
