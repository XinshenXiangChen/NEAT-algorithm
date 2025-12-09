class NEAT:
    def __init__(self, num_input, num_output):
        self.num_input = num_input
        self.num_output = num_output
        self.generation = 0


        """
        The list of genotypes
        """
        self.population = []

    """
    fm_step should return with a list of all input values, these are then fed forward to each of the genotypes 
    """
    def run(self, fn_step):
        input_values = fn_step()
