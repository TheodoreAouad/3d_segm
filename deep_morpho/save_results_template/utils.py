from typing import List, Dict

import base64
import io


def plot_to_base64(figure):
    my_stringIObytes = io.BytesIO()
    figure.savefig(my_stringIObytes, format='jpg')
    my_stringIObytes.seek(0)
    return base64.b64encode(my_stringIObytes.read()).decode()


def plot_to_html(figure):
    base64_fig = plot_to_base64(figure)
    return f"<img src='data:image/png;base64, {base64_fig}'>"


def detect_identical_values(all_args: List[Dict]) -> (List[str], List[str]):
    """
    Given a list of dicts, gives the keys where the values are the same.
    """

    same_values = []

    for key in all_args[0].keys():
        bad_key = False
        for args in all_args[1:]:
            if args[key] != all_args[0][key]:
                bad_key = True
                break

        if bad_key:
            break
        same_values.append(key)

    return same_values, list(set(all_args[0].keys()).difference(same_values))
