from array import array


class QTable(object):
    __slots__ = "actions", "states", "table"

    def __init__(self) -> None:
        self.actions = []
        self.states = []
        self.table = []
