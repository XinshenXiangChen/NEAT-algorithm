from enum import Enum, auto
from utils import sigmoid


class Node:
    pass


class InputNode(Node):
    def forward(self, value):
        return value

class HiddenNode():
    def __init__(self, layer):
        self.layer = layer

    def forward(self, value):
        return sigmoid(value)


class OutputNode(Node):
    def forward(self, value):
        return value  # Output nodes typically use identity/linear activation


