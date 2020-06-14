"""Save figures"""

import matplotlib.pyplot as plt
import farms_pylog as pylog


def save_figure(figure, name=None, **kwargs):
    """ Save figure """
    for extension in kwargs.pop("extensions", ["pdf"]):
        fig = figure.replace(" ", "_").replace(".", "dot")
        if name is None:
            name = "{}.{}".format(fig, extension)
        else:
            name = "{}.{}".format(name, extension)
        fig = plt.figure(figure)
        size = plt.rcParams.get('figure.figsize')
        fig.set_size_inches(0.7*size[0], 0.7*size[1], forward=True)
        plt.savefig(name, bbox_inches='tight')
        pylog.debug("Saving figure {}...".format(name))
        fig.set_size_inches(size[0], size[1], forward=True)


def save_figures(**kwargs):
    """Save_figures"""
    figures = [str(figure) for figure in plt.get_figlabels()]
    pylog.debug("Other files:\n    - " + "\n    - ".join(figures))
    for name in figures:
        save_figure(name, extensions=kwargs.pop("extensions", ["pdf"]))

