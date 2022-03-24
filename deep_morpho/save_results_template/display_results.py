import pathlib
from typing import List, Dict
import os
from os.path import join
import re

from general.utils import load_json
from .html_template import html_template
from .utils import detect_identical_values, plot_to_html, load_png_as_fig
from .load_args import load_args


class DisplayResults:


    def __init__(self):
        pass

    def write_results(self, results, changing_args):
        results_html = ''

        # Arguments
        results_html += (
            f"<div>"
            f"<h3>{results['tb_path']}</h3>"  # tb
            f"<p>{dict({k: results['args'][k] for k in changing_args})}</p>"  # args
        )

        # Weights
        # if "normalized_weights" in results.keys():
        #     results_html += ''.join([f"<span>{plot_to_html(fig)}</span>" for fig in results['normalized_weights']])

        # if "target_selem" in results.keys():
        #     results_html += ''.join([f"<span>{plot_to_html(fig)}</span>" for fig in results['target_selem']])

        # Target Operation
        results_html += "Target Operation"
        results_html += f"<span>{plot_to_html(results['target_operation'])}</span>"


        # Learned model
        results_html += "Learned Selems"
        results_html += f"<span>{plot_to_html(results['model_learned_viz'])}</span>"

        results_html += "Closest Selems"
        results_html += f"<span>{plot_to_html(results['model_closest_viz'])}</span>"

        results_html += "Learned Weights"
        results_html += f"<span>{plot_to_html(results['model_weights_viz'])}</span>"


        # Metrics
        results_html += f"<p>dice={results['dice']} || baseline={results['baseline_dice']}"
        if "binary_mode_dice" in results.keys():
            results_html += f" || binary mode={results['binary_mode_dice']}"
        results_html += "</p>"

        results_html += f"<p>step until convergence (dice)={results['convergence_dice']}</p>"


        # Zoom on learned selems
        results_html += "<p>learned selems: "
        results_html += ' '.join([
            f"layer {layer_idx} chin {chin} chout {chout} <span>{plot_to_html(fig)}</span> cvg={results['convergence_layer'][layer_idx, chin, chout]}"
            for (layer_idx, chin, chout), fig in results['learned_selem'].items()])

        results_html += "</p>"

        # Zoom on weights
        results_html += "<p>weights: "
        results_html += ' '.join([
            f"layer {layer_idx} chin {chin} chout {chout} <span>{plot_to_html(fig)}</span>"
            for (layer_idx, chin, chout), fig in results['normalized_weights'].items()])

        results_html += "</p>"


        results_html += "</div>"
        return results_html

    def write_all_results(self, results_dict, changing_args):
        results_html = ""

        for i, results in enumerate(results_dict):
            results_html += self.write_results(results, changing_args)

        return results_html

    def write_table(self, results, changing_args):
        table_html = ""

        table_html += f"<td>{results['tb_path']}</td>"
        for arg in changing_args:
            table_html += f"<td>{results['args'][arg]}</td>"
        table_html += f"<td>{float(results['dice']):.2f}</td>"
        table_html += f"<td>{float(results['baseline_dice']):.2f}</td>"
        table_html += f"<td>{float(results['binary_mode_dice']):.2f}</td>"
        table_html += f"<td>{results['convergence_dice']}</td>"

        return table_html

    def write_all_tables(self, results_dict, changing_args):
        table_html = "<table>"

        table_html += "<thread><thead><tr>"
        table_html += "<th>path</th>"
        for arg in changing_args + ['dice', 'baseline dice', 'binary mode dice', "convergence dice"]:
            table_html += f"<th>{arg}</th>"
        table_html += "</tr></thead></thread>"

        table_html += "<tbody>"
        for i, results in enumerate(results_dict):
            table_html += "<tr>"
            table_html += self.write_table(results, changing_args)
            table_html += "</tr>"
        table_html += "</tbody>"

        table_html += "</table>"
        return table_html

    def write_html_from_dict_deep_morpho(self, results_dict: List[Dict], save_path: str, title: str = "",):
        html = html_template()

        tb_paths = [res["tb_path"] for res in results_dict]

        global_args, changing_args = detect_identical_values([results['args'] for results in results_dict])

        table_html = self.write_all_tables(results_dict, changing_args)
        results_html = self.write_all_results(results_dict, changing_args)


        html = html.format(
            title=title,
            tb_paths=tb_paths,
            global_args=global_args,
            changing_args=changing_args,
            table=table_html,
            results=results_html,
        )

        pathlib.Path(save_path).parent.mkdir(exist_ok=True, parents=True)
        with open(save_path, "w") as f:
            f.write(html)
        return html

    @staticmethod
    def update_results_PlotWeightsBiSE(path):
        res = {}

        if os.path.exists(path):
            weights = {}
            normalized_weights = {}
            for file_ in os.listdir(path):
                if re.search(r"^weights", file_):
                    layer_idx, chin, chout = [int(s) for s in re.findall(r'weights_layer_(\d+)_chin_(\d+)_chout_(\d+).png', file_)[0]]
                    weights[layer_idx, chin, chout] = load_png_as_fig(join(path, file_))
                elif re.search(r"^normalized_weights", file_):
                    layer_idx, chin, chout = [int(s) for s in re.findall(r'normalized_weights_layer_(\d+)_chin_(\d+)_chout_(\d+).png', file_)[0]]
                    normalized_weights[layer_idx, chin, chout] = load_png_as_fig(join(path, file_))
            res['weights'] = weights
            res['normalized_weights'] = normalized_weights
        return res

    @staticmethod
    def update_results_PlotParametersBiSE(path):
        res = {}

        file_parameters = join(path, "parameters.json")
        if os.path.exists(file_parameters):
            parameters = load_json(file_parameters)
            for key in ['bias', 'activation_P']:
                res[key] = [parameters[layer_idx].get(key, None) for layer_idx in sorted(parameters.keys(), key=int)]
        return res



    @staticmethod
    def update_results_ConvergenceBinary(path):
        res = {}

        file_convergence_binary = join(path, "convergence_step.json")
        if os.path.exists(file_convergence_binary):
            convergence_steps = load_json(file_convergence_binary)
            # res['convergence_layer'] = [convergence_steps[layer_idx] for layer_idx in sorted(convergence_steps.keys(), key=int)]
            res['convergence_layer'] = {eval(k): v for k, v in convergence_steps['bisel'].items()}

        return res

    @staticmethod
    def update_results_InputAsPredMetric(path):
        res = {}

        file_baseline = join(path, "baseline_metrics.json")
        if os.path.exists(file_baseline):
            res['baseline_dice'] = load_json(file_baseline)["dice"]

        return res

    @staticmethod
    def update_results_BinaryModeMetric(path):
        res = {}

        file_binary_mode = join(path, "metrics.json")
        if os.path.exists(file_binary_mode):
            res['binary_mode_dice'] = load_json(file_binary_mode)["dice"]

        return res

    @staticmethod
    def update_results_CalculateAndLogMetrics(path):
        res = {}

        file_metrics = join(path, "metrics.json")
        if os.path.exists(file_metrics):
            res['dice'] = load_json(file_metrics)["dice"]

        return res

    @staticmethod
    def update_results_ConvergenceMetrics(path):
        res = {}

        file_convergence_metrics = join(path, "convergence_step.json")
        if os.path.exists(file_convergence_metrics):
            res['convergence_dice'] = load_json(file_convergence_metrics)['train']['dice']

        return res

    @staticmethod
    def update_results_ShowSelemBinary(path):
        res = {}

        if os.path.exists(path):
            learned_selem = {}
            for file_ in os.listdir(path):
                layer_idx, chin, chout = [int(s) for s in re.findall(r'layer_(\d+)_chin_(\d+)_chout_(\d+).png', file_)[0]]
                learned_selem[layer_idx, chin, chout] = load_png_as_fig(join(path, file_))
            res['learned_selem'] = learned_selem

        return res

    @staticmethod
    def update_results_target_SE(path):
        res = {}

        if os.path.exists(path):
            all_files_target = os.listdir(path)
            target_selem = [0 for _ in range(len(all_files_target))]
            for file_ in all_files_target:
                layer_idx = int(re.findall(r'target_SE_(\d+)', file_)[0])
                # target_selem[layer_idx] = load_png_as_fig(join(folder_plot_weights, file_))
                target_selem[layer_idx] = (load_png_as_fig(join(path, file_)))
            res['target_selem'] = target_selem

        return res

    @staticmethod
    def update_results_PlotModel(tb_path):
        res = {}

        path_fig = join(tb_path, "observables", "PlotModel")
        if os.path.exists(path_fig):
            res['model_weights_viz'] = load_png_as_fig(join(path_fig, "model_weights.png"))
            res['model_learned_viz'] = load_png_as_fig(join(path_fig, "model_learned.png"))
            res['model_closest_viz'] = load_png_as_fig(join(path_fig, "model_closest.png"))

        else:
            from deep_morpho.models import LightningBiMoNN
            from deep_morpho.viz import BimonnVizualiser
            file_ckpt = os.listdir(join(tb_path, "checkpoints"))[0]
            model = LightningBiMoNN.load_from_checkpoint(join(tb_path, "checkpoints", file_ckpt)).model
            res['model_weights_viz'] = BimonnVizualiser(model, mode="weights").get_fig()
            res['model_learned_viz'] = BimonnVizualiser(model, mode="learned").get_fig()
            res['model_closest_viz'] = BimonnVizualiser(model, mode="closest").get_fig()

        return res

    @staticmethod
    def update_results_target_operation(path):
        res = {}
        if os.path.exists(path):
            res['target_operation'] = load_png_as_fig(join(path, "morp_operations.png"))

        return res


    def get_results_from_tensorboard(self, tb_path: str):
        res = {
            "args": [None],
            "tb_path": None,
            "weights": None,
            "normalized_weights": {},
            "bias": None,
            "dice": None,
            "baseline_dice": None,
            "convergence_dice": None,
            "activation_P": [None],
            "learned_selem": dict(),
            "convergence_layer": None,
            "target_selem": [None],
            "target_operation": None,
            "learned_weights_viz": None,
            "learned_selems_viz": None,
        }
        obs_path = join(tb_path, "observables")

        res['tb_path'] = tb_path
        if os.path.exists(join(tb_path, 'args.yaml')):
            res['args'] = load_args(join(tb_path, 'args.yaml'))

        for obs_name in [
            "PlotWeightsBiSE",
            "BinaryModeMetric",
            "PlotParametersBiSE",
            "ConvergenceBinary",
            "InputAsPredMetric",
            "CalculateAndLogMetrics",
            "ConvergenceMetrics",
            "ShowSelemBinary",
        ]:
            res.update(getattr(self, f"update_results_{obs_name}")(join(obs_path, obs_name)))

        # res.update(self.update_results_target_SE(join(tb_path, "target_SE")))
        res.update(self.update_results_target_operation(join(tb_path, "morp_operations")))
        res.update(self.update_results_PlotModel(tb_path))

        return res



    def save(self, tb_paths: List[str], save_path: str, title: str = ""):
        results_dict = [self.get_results_from_tensorboard(tb_path) for tb_path in tb_paths]
        return self.write_html_from_dict_deep_morpho(results_dict, save_path, title)
