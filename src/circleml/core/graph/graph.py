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

"""Internal graph implementation."""

import typing as t
from pprint import pformat

T = t.TypeVar("T")
EdgeType = t.Union["Node", int]


class Node(t.Generic[T]):
    """A node in a graph."""

    __slots__ = "val", "edges"
    val: T
    edges: t.List[EdgeType]

    def __init__(self, val) -> None:
        self.val = val
        self.edges = []

    def __iadd__(self, other) -> "Node":
        self.edges.append(other)
        return self

    def __repr__(self) -> str:
        return f"Node({pformat(self.val)})"

    def __str__(self) -> str:
        return f"Node({pformat(self.val)})"

    def __iter__(self) -> t.Iterator[EdgeType]:
        return iter(self.edges)

    def __getitem__(self, idx):
        return self.edges[idx]

    def __len__(self):
        return len(self.edges)

    def set_edges(self, edges):
        """Set the edges of the node."""
        self.edges = edges

    def set_val(self, val):
        """Set the value of the node."""
        self.val = val


class Graph:
    """A graph data structure."""

    nodes: t.List[Node]

    def __init__(self, nodes: t.List[Node] = None):
        self.nodes = nodes or []

    def __add__(self, other):
        self.nodes.append(other)
        return self

    def __repr__(self):
        return f"Graph({pformat(self.nodes)})"

    def __str__(self):
        return f"Graph({pformat(self.nodes)})"

    def __iter__(self):
        return iter(self.nodes)

    def __getitem__(self, idx):
        return self.nodes[idx]

    def __len__(self):
        """Returns the number of nodes in the graph."""
        return len(self.nodes)

    def set_nodes(self, nodes):
        """Set the nodes of the graph."""
        self.nodes = nodes

    def add_node(self, node):
        """Add a node to the graph."""
        self.nodes.append(node)

    def add_bidirectional_edge(self, node1, node2):
        """Add a two-way edge between two nodes."""
        node1 += node2
        node2 += node1

    def add_unidirectional_edge(self, node1, node2):
        """Add a one-way edge between two nodes."""
        node1 += node2
