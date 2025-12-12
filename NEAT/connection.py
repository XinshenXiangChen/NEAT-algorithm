import random

from .config import config


class Connection:
    _innov_num_counter = 0
    connection_list = []

    def __init__(self, from_node, to_node, innov_num):
        self.from_to = (from_node, to_node)

        self.disabled = False
        self.weight = random.uniform(-1, 1)

        self.innov_num = None


        if innov_num == None:
            self.innov_num = Connection._innov_num_counter
            Connection._innov_num_counter += 1
            Connection.connection_list.append(self)
        else:
            self.innov_num = innov_num

    def forward(self, value):
        return value*self.weight


    def perturbate_weight(self):
        self.weight = self.weight + config.get("perturbation_step")

    def replace_weight(self):
        self.weight = random.uniform(-1, 1)




def same_connection(from_node, to_node, connection2):
    if (from_node, to_node) == connection2.from_to:
        return True
    return False


