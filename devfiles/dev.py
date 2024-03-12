import re
from pathlib import Path
import deep_morpho.save_results_template.display_results as dr


def get_tb_paths_recompute():
    path_global = Path(
        "deep_morpho/results/results_tensorboards/Bimonn_exp_75/multi/recompute_projected"
    )
    all_paths = []
    for batch_exp in path_global.iterdir():
        for dataset in batch_exp.iterdir():
            if not dataset.is_dir():
                continue
            if dataset.name == "code":
                continue
            for operation in (dataset / "bimonn").iterdir():
                for selem in operation.iterdir():
                    all_paths += sorted([str(p) for p in selem.iterdir()], key=lambda x: int(
                        re.findall(r'version_(\d+)$', x)[0]
                    ))
    return all_paths


all_paths = get_tb_paths_recompute()
big_df_recompute, _, _ = dr.DisplayResults().get_df_from_tb_paths(all_paths, show_details=False)
big_df_recompute["dice"] = big_df_recompute["train_dice"]
big_df_recompute["binary_mode_dice"] = big_df_recompute["binary_mode_train_dice"]
big_df_recompute["dataset_type"] = big_df_recompute["dataset"].apply(lambda x: x.replace("morpho", "").replace("inverted", "inverted_").replace("dataset", ""))
big_df_recompute["atomic_element"] = big_df_recompute["atomic_element.net"]

big_df_recompute["before_recomputed_projected"] = False
big_df_recompute["after_recomputed_projected"] = True


