import io

import matplotlib.pyplot as plt
import numpy as np


def get_hist_as_array(x, dpi=128, figure_kwargs={}, **kwargs):
    """Returns an image array containing the plotted histogram.

    Args:
        x (_type_): _description_
        dpi (int, optional): _description_. Defaults to 128.
        figure_kwargs (dict, optional): _description_. Defaults to {}.

    Returns:
        _type_: _description_
    """
    figure_kwargs['dpi'] = dpi
    fig, ax = plt.subplots(**figure_kwargs)
    ax.hist(x, **kwargs)

    return get_figure_as_array(fig)


def get_figure_as_array(fig):
    io_buf = io.BytesIO()
    fig.savefig(io_buf, format='raw', dpi=fig.dpi, pad_inches=0)
    io_buf.seek(0)
    img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                        newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
    io_buf.close()
    plt.close(fig)
    return img_arr
