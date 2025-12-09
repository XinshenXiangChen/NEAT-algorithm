class ConfigDict(dict):
    def __setitem__(self, key, value):
        if key not in self:
            raise KeyError(f"Key not in config file: {key}")
        super().__setitem__(key, value)


config = ConfigDict({
    "input_nodes": 5,
    "output_nodes": 5,
    "population_size": 100,
    "perturbation_step": 0.2,
    "weight_perturbation_rate": 0.8,
    "weight_replace_rate": 0.2,
    "new_connection_rate": 0.05,
    "new_node_rate": 0.03,
    "new_node_layer_rate": 0.2
})



with open("config.txt", "r") as f:
    for line in f.readlines():
        if line != "\n":
            list_line = line.strip("\n").split(" ")
            print(list_line)
            config[list_line[0]] = list_line[2]
