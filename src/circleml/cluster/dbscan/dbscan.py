# Copyright 2023 CircleML GitHub Authors. All Rights Reserved.
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

"""Internal DBScan module. Use circleml.cluster.dbscan instead."""

import typing as t
from collections import namedtuple

import numpy as np

from ... import log
from ...core import euclidean_distance
from ...core.base import UnsupervisedModelABC
from ...core.graph import Graph, Node

dbscan_sample = namedtuple("dbscan_sample", ["val", "cor", "lab", "dead"])


class DBScanCla(UnsupervisedModelABC):
    """The DBScan clustering algorithm, which finds samples with lots of neighbours to find clusters."""

    def __init__(
        self,
        min_dist: float = 0.5,
        min_neighbours: int = 4,
        distance_func: t.Callable[[t.Sized], float] = euclidean_distance,
    ) -> None:
        self.min_dist = min_dist
        self.min_neighbours = min_neighbours
        self.distance_func = distance_func
        self.__preds = []

    def __find_neighbours(self, X: np.ndarray, sample: np.ndarray) -> np.ndarray:
        return [
            i
            for i in range(len(X))
            if self.distance_func(X[i], sample) <= self.min_dist
        ]

    def fit(self, X: np.ndarray, verbose: bool = False) -> "DBScanCla":
        """Fit the DBScan algorithm to the data.

        Args:
            X (np.ndarray): The data to fit to.

        Returns:
            np.ndarray: The labels of the data.
        """
        logger = log.create_logger(log.info, verbose=verbose)
        logger("Constructing search graph")
        gr = Graph(
            [
                Node(dbscan_sample(cor=False, val=sample, dead=False, lab=-1))
                for sample in X
            ]
        )

        logger("Finding neighbours")
        for i, _ in enumerate(gr.nodes):
            ns = self.__find_neighbours(X, gr.nodes[i].val.val)
            gr.nodes[i].set_edges(ns)

            # if the node has enough neighbours, it is a core node
            if len(ns) >= self.min_neighbours:
                gr.nodes[i].val = gr.nodes[i].val._replace(cor=True)

        logger("Selecting core samples")
        # find all the core nodes
        core_nodes = [i for i, node in enumerate(gr.nodes) if node.val.cor]

        logger("Labelling samples")
        current_class = 0
        for node in core_nodes:
            n = gr.nodes[node]
            if n.val.dead:  # already visited
                continue

            # find all the neighbours of the core node
            self.__label_node(n, current_class, gr)
            current_class += 1

        self.__preds = [node.val.lab for node in gr.nodes]
        return self

    def __label_node(self, node: Node, label: int, gr: Graph) -> None:
        """Label a node and all its neighbours."""
        if not node.val.dead:
            node.val = node.val._replace(lab=label, dead=True)

            if node.val.cor:
                for n in node.edges:
                    self.__label_node(gr.nodes[n], label, gr)

    def predict(self, verbose: bool = False) -> np.ndarray:
        """Predict the labels of the data.

        Returns:
            np.ndarray: The labels of the data.
        """
        return self.__preds
