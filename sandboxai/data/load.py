# Copyright 2021 Neuron-AI GitHub Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from io import StringIO
from typing import Optional

import requests
from sandboxai.types.types import BufferedWriter, WritableFile
from numpy import array, loadtxt, ndarray


def get_cloud_data(url: str, *, encoding: str = "utf-8") -> str:
    """Returns a string of the request from the specified url.

    Parameters
    ----------
    url : str
        The url to request from.
    encoding : str, optional
        The encoding to decode the response with, by default "utf-8"

    Returns
    -------
    str
        The response decoded.
    """
    return requests.get(url).content.decode(encoding, "ignore")


def write_cloud_data(url: str, file: WritableFile) -> None:
    """Write cloud data to a file.

    Parameters
    ----------
    url : str
        The URL to get it from
    file : WritableFile
        The file to write it to
    """
    if isinstance(file, BufferedWriter):
        file.write(get_cloud_data(url))
    else:
        with open(file, "w") as file:
            file.write(get_cloud_data(url))


def cloud_csv_to_np_array(url: str, **kwargs) -> ndarray:
    """Returns the CSV data from the URL as a numpy ndarray.

    Pass delimiter to the function to use a custom delimeter.

    Parameters
    ----------
    url : str
        The URL to get data from

    Returns
    -------
    ndarray
        The array of the CSV data
    """
    return array(
        loadtxt(
            StringIO(get_cloud_data(url)),
            dtype=object,
            delimiter=kwargs.get("delimiter") or ",",
        )
    )


def load_random_humans(filename: Optional[str] = ...) -> ndarray:
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
    if filename != ...:
        write_cloud_data(
            "https://raw.githubusercontent.com/neuron-ai/datasets/main/humans-random-numbers/random_humans.csv",
            filename,
        )
    return cloud_csv_to_np_array(
        "https://raw.githubusercontent.com/neuron-ai/datasets/main/humans-random-numbers/random_humans.csv"
    )
