import random
from node import HiddenNode, Node, InputNode, OutputNode
from connection import same_connection
from connection import Connection

from config import config


class Genotype:
    def __init__(self, input_nodes, output_nodes):
        self.node_counter = 0
        self.input_nodes = input_nodes[:] # layer -1
        self.hidden_layers = []
        self.output_nodes = output_nodes[:] # layer len(hidden_layers)

        self.fn_fitness = None # set after evaluate

        self.connections = []

    def add_layer(self):
        self.hidden_layers.append([])
    
    def add_node_to_layer(self, layer_index):
        """Add a node to the specified layer. Creates layer if it doesn't exist."""

        """ This should only run once, since new layers are always + 1 of previous deepest hidden layer"""
        while len(self.hidden_layers) <= layer_index:
            self.hidden_layers.append([])
        node = HiddenNode(self.node_counter, layer_index)
        self.node_counter += 1
        self.hidden_layers[layer_index].append(node)
        return node

    def forward(self):
        pass

    def mutate(self):
        if random.random() < config.get("new_connection_rate"):
            from_node_layer = random.randint(-1, len(self.hidden_layers) - 1)
            to_node_layer = random.randint(from_node_layer + 1, len(self.hidden_layers))

            if from_node_layer == -1:
                from_node = random.choice(self.input_nodes)
            else :
                from_node = random.choice(self.hidden_layers[from_node_layer])

            if to_node_layer == len(self.hidden_layers):
                to_node = random.choice(self.output_nodes)
            else :
                to_node = random.choice(self.hidden_layers[to_node_layer])

            connection_exists = any(same_connection(from_node, to_node, conn) for conn in self.connections)
            if not connection_exists:
                innov_num = None
                for conn in Connection.connection_list:
                    if same_connection(from_node, to_node, conn):
                        innov_num = conn.innov_num
                        break
                new_connection = Connection(from_node, to_node, innov_num)
                self.connections.append(new_connection)




        if random.random() < config.get("new_node_rate"):
            new_node_layer = random.randint(0, len(self.hidden_layers) - 1) if self.hidden_layers else 0

            if random.random() < config.get("new_node_layer_rate"):
                new_node_layer += 1

            new_node = self.add_node_to_layer(new_node_layer)

            # Find connections from layer-1 to layer+1 to split
            from_layer = new_node_layer - 1
            to_layer = new_node_layer + 1

            """
            this part gets all the connections from the genotype and finds the ones
            that have the same from_layer and to_layer
            """

            valid_connections = []
            for conn in self.connections:
                if conn.disabled:
                    continue

                from_node, to_node = conn.from_to

                if isinstance(from_node, InputNode):
                    from_node_layer = -1
                elif isinstance(from_node, OutputNode):
                    from_node_layer = len(self.hidden_layers)
                else:  # HiddenNode
                    from_node_layer = from_node.layer

                if isinstance(to_node, InputNode):
                    to_node_layer = -1
                elif isinstance(to_node, OutputNode):
                    to_node_layer = len(self.hidden_layers)
                else:  # HiddenNode
                    to_node_layer = to_node.layer

                if from_node_layer == from_layer and to_node_layer == to_layer:
                    valid_connections.append(conn)

            # gets the random valid layer and splits it
            if valid_connections:
                conn = random.choice(valid_connections)
                conn.disabled = True
                Connection(conn.from_to[0], new_node, None).weight = 1.0
                Connection(new_node, conn.from_to[1], None).weight = conn.weight



        for conn in self.connections:
            if random.random() < config.get("weight_perturbation_rate"):
                conn.perturbate_weight()

            if random.random() < config.get("weight_replace_rate"):
                conn.replace_weight()


    def crossover(self):
        pass

    def fitness(self):
        pass