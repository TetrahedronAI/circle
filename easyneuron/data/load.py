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
            "https://raw.githubusercontent.com/neuron-ai/datasets/main/humans_random_numbers/random_humans.csv", filename)
    return cloud_csv_to_np_array(
        "https://raw.githubusercontent.com/neuron-ai/datasets/main/humans_random_numbers/random_humans.csv"
    )
