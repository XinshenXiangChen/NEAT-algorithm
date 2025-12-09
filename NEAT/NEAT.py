from config import config


class NEAT:
    def __init__(self, fn_fitness):
        self.num_input = config.get("input_nodes")
        self.num_output = config.get("output_nodes")
        self.generation = 0
        self.population_size = config.get("population_size")

        self.fn_fitness = fn_fitness
        """
        The list of genotypes
        """
        self.population = []

    """
    fm_step should return with a list of all input values, these are then fed forward to each of the genotypes 
    """
    def run(self, fn_step):
        input_values = fn_step()

    def evaluate(self, fn_eval):
        for g in self.population:
            g.fitness_score = fn_eval(g)