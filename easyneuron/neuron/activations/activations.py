# Copyright 2022 Neuron-AI GitHub Authors. All Rights Reserved.
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

from numpy import exp, square, tanh
from typing import Any

from easyneuron.neuron.activations._classes import Activation


class Sigmoid(Activation):
    """The sigmoid activation function."""

    def forward(self, X: Any) -> Any:
        """Returns the sigmoid, element wise, of the iterable inputted.

        Parameters
        ----------
        X : Any
                        The iterable of floats to input.

        Returns
        -------
        Any
                        The iterable of results after calculating the sigmoid function on all of the inputs.
        """

        return [self.__forward(i) for i in X]

    def backward(self, X: Any) -> Any:
        """Returns the derivative of the sigmoid element wise, of the iterable inputted.

        Parameters
        ----------
        X : Any
                        The iterable of floats to input.

        Returns
        -------
        Any
                        The iterable of results after calculating the sigmoid function on all of the inputs.
        """
        return [self.__backward(i) for i in X]

    def __forward(self, value: float) -> float:
        return 1 / (1 + exp(-value))

    def __backward(self, value: float) -> float:
        return self.__forward(value) * (1 - self.__forward(value))


class Tanh(Activation):
    """The tanh activation function."""

    def forward(self, X: Any) -> Any:
        """Returns the hyperbolic tangent element wise, of the iterable inputted.

        Parameters
        ----------
        X : Any
                        The iterable of floats to input.

        Returns
        -------
        Any
                        The iterable of results after calculating the tanh function on all of the inputs.
        """

        return [self.__forward(i) for i in X]

    def backward(self, X: Any) -> Any:
        """Returns the derivative of the hyperbolic tangent element wise, of the iterable inputted.

        Parameters
        ----------
        X : Any
                        The iterable of floats to input.

        Returns
        -------
        Any
                        The iterable of results after calculating the tanh function on all of the inputs.
        """
        return [self.__backward(i) for i in X]

    def __forward(self, value: float) -> float:
        return tanh(value)

    def __backward(self, value: float) -> float:
        return 1 - square(tanh(value))


class ReLU(Activation):
    """The ReLU activation function."""

    def forward(self, X: Any) -> Any:
        """Returns the ReLU activation element wise, of the iterable inputted.

        Parameters
        ----------
        X : Any
                        The iterable of floats to input.

        Returns
        -------
        Any
                        The iterable of results after calculating the ReLU function on all of the inputs.
        """

        return [self.__forward(i) for i in X]

    def backward(self, X: Any) -> Any:
        """Returns the derivative of the ReLU element wise, of the iterable inputted.

        Parameters
        ----------
        X : Any
                        The iterable of floats to input.

        Returns
        -------
        Any
                        The iterable of results after calculating the ReLU function on all of the inputs.
        """
        return [self.__backward(i) for i in X]

    def __forward(self, value: float) -> float:
        return max(value, 0)

    def __backward(self, value: float) -> float:
        return 0 if value < 0 else 1
