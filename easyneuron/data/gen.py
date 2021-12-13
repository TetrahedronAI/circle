from random import normalvariate, randint
from typing import Tuple

from easyneuron.types import Int
from numpy import array, ndarray


def gen_stairs(classes: int, features: int, samples: Int = 10000, sd: float = 0.3, factor: float = 3) -> Tuple[ndarray, ndarray]:
    X = []
    y = []

    for _ in range(samples):
        label = randint(1, classes)*factor
        to_append = [normalvariate(label, sd) for _ in range(features)]

        X.append(to_append)
        y.append(label)

    return array(X), array(y)
