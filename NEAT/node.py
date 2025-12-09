from enum import Enum, auto
from utils import sigmoid


class Node:


    def __init__(self, node_id):
        self.node_id = node_id


class InputNode(Node):
    def __init__(self, node_id):
        super().__init__(node_id)


    def forward(self, value):
        return value

class HiddenNode(Node):
    def __init__(self, node_id, layer):
        super().__init__(node_id)

        self.layer = layer


    def forward(self, value):
        return sigmoid(value)


class OutputNode(Node):
    def __init__(self, node_id):
        super().__init__(node_id)


