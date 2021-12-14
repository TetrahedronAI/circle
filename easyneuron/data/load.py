from typing import Optional
import requests
from io import StringIO
from numpy import array, loadtxt, ndarray


def get_cloud_data(url, *, encoding: str = "utf-8") -> str:
    return requests.get(url).content.decode(encoding)


def write_cloud_data(url, filename) -> None:
    with open(filename, "w") as file:
        file.write(get_cloud_data(url))


def cloud_csv_to_np_array(url):
    return array(
        loadtxt(
            StringIO(get_cloud_data(url)),
            dtype=object,
            delimiter=","
        )
    )


def load_random_humans(filename: Optional[str] = None) -> ndarray:
    """Get the random_humans dataset (random numbers chosen by people).

    Parameters
    ----------
    filename : Optional[str], optional
            The filename (if you want write it to a file, otherwise, leave blank), by default None

    Returns
    -------
    ndarray
            The dataset as a numpy array.
    """
    if filename:
        write_cloud_data(
            "https://raw.githubusercontent.com/neuron-ai/easyneuron-datasets/main/humans_random_numbers/random_humans.csv", filename)
    return cloud_csv_to_np_array(
        "https://raw.githubusercontent.com/neuron-ai/easyneuron-datasets/main/humans_random_numbers/random_humans.csv"
    )
