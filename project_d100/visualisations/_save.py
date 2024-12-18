import re

import matplotlib.pyplot as plt


def save_graph(
    path,
    fig: plt.Figure,
    name: str = "graph",
    dpi: int = 300,
    file_format: str = "png",
) -> None:
    """Exports a graph to a file in the specified format

    Arguments:
        path: the path to the directory where the file will be saved
        fig (plt.Figure): the figure object to be saved
        name (str): the name of the file to be saved
        dpi (int): the resolution of the image
        file_format (str): the format of the file to be saved

    Returns:
        None
    """
    clean_name = re.sub(
        r"[^\w\-_\. ]", "_", name
    )  # Replace illegal characters with underscores

    filename = f"{clean_name}.{file_format}"

    # Save the plot using Matplotlib's savefig method
    fig.savefig(path / filename, format=file_format, dpi=dpi, bbox_inches="tight")
