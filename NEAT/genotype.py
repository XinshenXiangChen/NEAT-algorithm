import random
from node import HiddenNode, Node, InputNode, OutputNode
from connection import same_connection
from connection import Connection

from config import config


class Genotype:
    def __init__(self, input_nodes, output_nodes):
        self.input_nodes = input_nodes[:] # layer -1
        self.hidden_layers = []
        self.output_nodes = output_nodes[:] # layer len(hidden_layers)

        self.fn_fitness = None # set after evaluate
        self.fitness_score = 0.0
        self.adjusted_fitness = 0.0  # Fitness after speciation sharing

        self.connections = []
        self.dict_cache_connections = None

    def add_layer(self):
        self.hidden_layers.append([])
    
    def add_node_to_layer(self, layer_index):
        """Add a node to the specified layer. Creates layer if it doesn't exist."""

        """ This should only run once, since new layers are always + 1 of previous deepest hidden layer"""
        while len(self.hidden_layers) <= layer_index:
            self.hidden_layers.append([])
        node = HiddenNode(layer_index)
        self.hidden_layers[layer_index].append(node)
        return node

    def forward(self, input_values):
        if self.dict_cache_connections is None:
            self.dict_cache_connections = self.create_connection_cache()

        incoming_connections = self.dict_cache_connections
        node_values = {}

        # stores input numbers to be processed later
        for i, node in enumerate(self.input_nodes):
            node_values[node] = input_values[i] if i < len(input_values) else 0.0

        # process hidden layers and gets their weighted sum / activation function (sigmoid)
        for layer in self.hidden_layers:
            for node in layer:
                weighted_sum = 0.0
                for from_node, conn in incoming_connections[node]:
                    if from_node in node_values:
                        weighted_sum += conn.forward(node_values[from_node])

                node_values[node] = node.forward(weighted_sum)

        # Process output layer
        output_values = []
        for node in self.output_nodes:
            # Sum weighted inputs from incoming connections
            weighted_sum = 0.0
            for from_node, conn in incoming_connections[node]:
                if from_node in node_values:
                    weighted_sum += conn.forward(node_values[from_node])
            # Apply activation function and store
            output_values.append(node.forward(weighted_sum))

        return output_values




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
                new_conn1 = Connection(conn.from_to[0], new_node, None)
                new_conn1.weight = 1.0
                self.connections.append(new_conn1)
                new_conn2 = Connection(new_node, conn.from_to[1], None)
                new_conn2.weight = conn.weight
                self.connections.append(new_conn2)



        for conn in self.connections:
            if random.random() < config.get("weight_perturbation_rate"):
                conn.perturbate_weight()

            if random.random() < config.get("weight_replace_rate"):
                conn.replace_weight()

    """
    Calculate the compatibility between 2 genotypes, thou in the paper compatibility is defined as:
    compatibility = c1*Ne/N + c2*Nd/N + c3W
    
    The calculation of compatibility in this function is:
    c1*Ned/N + c3W
    
    Meaning it treats excess nodes and disjoints nodes as equal
    """
    def compatibility_distance(self, other):


        c1 = config.get("compatibility_disjoint_coefficient")
        c3 = config.get("compatibility_weight_coefficient")

        # dict comprehension again
        self_innov = {conn.innov_num: conn.weight for conn in self.connections if not conn.disabled}
        other_innov = {conn.innov_num: conn.weight for conn in other.connections if not conn.disabled}
        
        if not self_innov and not other_innov:
            return 0.0
        
        # Find matching and disjoint/excess genes (treated the same)
        self_innov_nums = set(self_innov.keys())
        other_innov_nums = set(other_innov.keys())
        
        matching = self_innov_nums & other_innov_nums
        # All non-matching genes (disjoint + excess treated the same)
        disjoint_excess = (self_innov_nums - other_innov_nums) | (other_innov_nums - self_innov_nums)
        
        # Normalization factor
        N = max(len(self_innov_nums), len(other_innov_nums), 1)
        
        # Average weight difference of matching genes
        if matching:
            weight_diff = sum(abs(self_innov[n] - other_innov[n]) for n in matching)
            avg_weight_diff = weight_diff / len(matching)
        else:
            avg_weight_diff = 0.0
        
        # Calculate compatibility distance
        distance = (c1 * len(disjoint_excess) / N) + (c3 * avg_weight_diff)
        
        return distance

    """
    Cache creator for forward pass, maps a Node to all of its incomming nodes and connections
    """
    def create_connection_cache(self):
        incoming_connections = {}

        for node in self.input_nodes:
            incoming_connections[node] = []
        for layer in self.hidden_layers:
            for node in layer:
                incoming_connections[node] = []
        for node in self.output_nodes:
            incoming_connections[node] = []

        # Build the connection map - only iterate connections once
        for conn in self.connections:
            if conn.disabled:
                continue
            from_node, to_node = conn.from_to
            if to_node in incoming_connections:
                incoming_connections[to_node].append((from_node, conn))

        return incoming_connections

def crossover(strong_genotype: Genotype, weak_genotype: Genotype):

    strong_connections = strong_genotype.connections
    weak_connections = weak_genotype.connections
    
    #dictionary comprehension xD
    strong_conn_dict = {conn.innov_num: conn for conn in strong_connections if not conn.disabled}
    weak_conn_dict = {conn.innov_num: conn for conn in weak_connections if not conn.disabled}

    all_innov_nums = set(strong_conn_dict.keys()) | set(weak_conn_dict.keys())
    
    offspring_connections = []
    
    for innov_num in sorted(all_innov_nums):  # Sort for consistency
        if innov_num in strong_conn_dict and innov_num in weak_conn_dict:

            chosen_conn = random.choice([strong_conn_dict[innov_num], weak_conn_dict[innov_num]])
            offspring_connections.append(chosen_conn)
        elif innov_num in strong_conn_dict:

            offspring_connections.append(strong_conn_dict[innov_num])

    
    return offspring_connections

