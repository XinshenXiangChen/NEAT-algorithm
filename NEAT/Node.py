from enum import Enum, auto
from utils import sigmoid


class Node:
    _node_id_counter = 0

    def __init__(self):
        self.node_id = Node._node_id_counter
        Node._node_id_counter += 1


class InputNode(Node):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self):
        return self.value

class HiddenNode(Node):
    def __init__(self):
        super().__init__()


    def forward(self, value):
        return sigmoid(value)


class OutputNode(Node):
    def __init__(self):
        super().__init__()


