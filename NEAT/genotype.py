import random
from .node import HiddenNode, InputNode, OutputNode
from .connection import same_connection
from .connection import Connection

from .config import config


class Genotype:
    def __init__(self, input_nodes, output_nodes):
        self.connections = []
        self.hidden_layers = []
        self.dict_cache_connections = None

        self.next_node_id = 0

        self.input_nodes = self._create_nodes(input_nodes, InputNode)  # layer -1
        self.output_nodes = self._create_nodes(output_nodes, OutputNode)  # layer len(hidden_layers)
        self._fully_connect_inputs_to_outputs()

        # fitness attributes
        self.hidden_layers = []
        self.fn_fitness = None # set after evaluate
        self.fitness_score = 0.0
        self.adjusted_fitness = 0.0  # Fitness after speciation sharing

    def _next_node_id(self):
        node_id = self.next_node_id
        self.next_node_id += 1
        return node_id

    def _create_nodes(self, count, node_cls):
        """Create nodes from a count or clone existing nodes, ensuring unique ids."""
        nodes = []
        for _ in range(count):
            nodes.append(node_cls(self._next_node_id()))

        return nodes

    def _fully_connect_inputs_to_outputs(self):
        """Connect every input node to every output node, reusing innovation numbers."""
        for in_node in self.input_nodes:
            for out_node in self.output_nodes:
                innov_num = None
                for conn in Connection.connection_list:
                    if same_connection(in_node, out_node, conn):
                        innov_num = conn.innov_num
                        break
                self.connections.append(Connection(in_node, out_node, innov_num))

    def add_layer(self):
        self.hidden_layers.append([])
    
    def add_node_to_layer(self, layer_index):
        """Add a node to the specified layer. Creates layer if it doesn't exist."""

        """ This should only run once, since new layers are always + 1 of previous deepest hidden layer"""
        while len(self.hidden_layers) <= layer_index:
            self.hidden_layers.append([])
        node = HiddenNode(self._next_node_id(), layer_index)
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
            else:
                # If chosen hidden layer is empty, skip this mutation attempt
                if not self.hidden_layers or not self.hidden_layers[from_node_layer]:
                    from_node = None
                else:
                    from_node = random.choice(self.hidden_layers[from_node_layer])

            if to_node_layer == len(self.hidden_layers):
                to_node = random.choice(self.output_nodes)
            else:
                if not self.hidden_layers or not self.hidden_layers[to_node_layer]:
                    to_node = None
                else:
                    to_node = random.choice(self.hidden_layers[to_node_layer])

            if from_node is not None and to_node is not None:
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

def crossover(strong_genotype: Genotype, weak_genotype: Genotype) -> Genotype:
    """Return an offspring genotype built from the strong and weak parents."""
    # Build connection maps, ignoring disabled links
    strong_conn = {c.innov_num: c for c in strong_genotype.connections if not c.disabled}
    weak_conn = {c.innov_num: c for c in weak_genotype.connections if not c.disabled}

    all_innov = set(strong_conn) | set(weak_conn)
    chosen_conns = []
    for innov in sorted(all_innov):
        if innov in strong_conn and innov in weak_conn:
            chosen_conns.append(random.choice([strong_conn[innov], weak_conn[innov]]))
        elif innov in strong_conn:
            chosen_conns.append(strong_conn[innov])

    # Collect all nodes referenced by chosen connections from both parents
    def collect_nodes(genotype):
        nodes = {n.node_id: n for n in genotype.input_nodes}
        nodes.update({n.node_id: n for n in genotype.output_nodes})
        for layer in genotype.hidden_layers:
            for n in layer:
                nodes[n.node_id] = n
        return nodes

    node_lookup = collect_nodes(strong_genotype)
    node_lookup.update(collect_nodes(weak_genotype))

    # Create offspring with zeroed counts to control node construction manually
    offspring = Genotype(0, 0)

    # Clone input/output nodes following the strong parent's ordering
    def clone_node(node):
        if isinstance(node, InputNode):
            return InputNode(node.node_id)
        if isinstance(node, OutputNode):
            return OutputNode(node.node_id)
        return HiddenNode(node.node_id, node.layer)

    offspring.input_nodes = [clone_node(n) for n in strong_genotype.input_nodes]
    offspring.output_nodes = [clone_node(n) for n in strong_genotype.output_nodes]

    # Prepare hidden layers sized by max layer seen
    hidden_nodes = [
        clone_node(node_lookup[node_id])
        for conn in chosen_conns
        for node_id in (conn.from_to[0].node_id, conn.from_to[1].node_id)
        if isinstance(node_lookup[node_id], HiddenNode)
    ]
    if hidden_nodes:
        max_layer = max(n.layer for n in hidden_nodes)
        offspring.hidden_layers = [[] for _ in range(max_layer + 1)]
        for n in hidden_nodes:
            offspring.hidden_layers[n.layer].append(n)

    # Clone connections to reference the offspring's nodes
    id_to_node = {
        n.node_id: n for n in offspring.input_nodes + offspring.output_nodes
    }
    for layer in offspring.hidden_layers:
        for n in layer:
            id_to_node[n.node_id] = n

    offspring.connections = []
    for conn in chosen_conns:
        from_node, to_node = conn.from_to
        new_conn = Connection(id_to_node[from_node.node_id], id_to_node[to_node.node_id], conn.innov_num)
        new_conn.weight = conn.weight
        new_conn.disabled = conn.disabled
        offspring.connections.append(new_conn)

    offspring.next_node_id = max(id_to_node.keys(), default=-1) + 1
    offspring.dict_cache_connections = None
    offspring.fn_fitness = None
    offspring.fitness_score = 0.0
    offspring.adjusted_fitness = 0.0

    return offspring

