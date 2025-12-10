import random
from config import config
from node import *
from genotype import Genotype, crossover
from connection import Connection

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
        self.species = []  # List of species, each is a list of genotypes
        self.species_representatives = []  # Representative genotype for each species

        self.num_generations = config.get("num_generations")

        self.input_nodes = config.get("input_nodes")
        self.output_nodes = config.get("output_nodes")

    def init_population(self):
        self.population = []
        for _ in range(self.population_size):
            genotype = Genotype(self.input_nodes, self.output_nodes)
            self.population.append(genotype)

    """
    fm_step should return with a list of all input values, these are then fed forward to each of the genotypes 
    """
    def run(self, fn_step):
        input_values = fn_step()

    def evaluate(self, fn_eval):
        for g in self.population:
            g.fitness_score = fn_eval(g)

    def compatibility_distance(self, g1, g2):
        """Wrapper for genotype compatibility distance."""
        return g1.compatibility_distance(g2)

    def speciate(self):
        """
        Group genotypes into species based on compatibility distance.
        Each species has a representative (first member).
        """
        compatibility_threshold = config.get("compatibility_threshold", 3.0)
        
        self.species = []
        self.species_representatives = []
        
        for genotype in self.population:
            # Try to find a compatible species
            assigned = False
            for i, representative in enumerate(self.species_representatives):
                distance = genotype.compatibility_distance(representative)
                if distance < compatibility_threshold:
                    # Add to existing species
                    self.species[i].append(genotype)
                    assigned = True
                    break
            
            # If no compatible species found, create new one
            if not assigned:
                self.species.append([genotype])
                self.species_representatives.append(genotype)

    def calculate_adjusted_fitness(self):
        """
        Calculate adjusted fitness (fitness sharing within species).
        Adjusted fitness = raw fitness / species size
        """
        for species in self.species:
            species_size = len(species)
            for genotype in species:
                # Fitness sharing: divide by species size
                genotype.adjusted_fitness = genotype.fitness_score / species_size

    def create_population(self):
        # Initialize if empty
        if not self.population:
            self.init_population()
            return

        # Sort by fitness descending and keep top 20% as breeders
        sorted_pop = sorted(self.population, key=lambda g: g.fitness_score, reverse=True)
        survivors_count = max(1, int(self.population_size * config.get("population_cut")))
        breeders = sorted_pop[:survivors_count]

        new_population = []

        # Elitism: clone the best
        elite = crossover(breeders[0], breeders[0])
        new_population.append(elite)

        # Fill the rest via crossover/mutation from survivors
        while len(new_population) < self.population_size:
            strong = random.choice(breeders)
            weak = random.choice(breeders)
            child = crossover(strong, weak)
            child.mutate()
            new_population.append(child)

        self.population = new_population
        self.generation += 1

    def select_parents(self):

        sorted_pop = sorted(self.population, key=lambda g: g.fitness_score, reverse=True)

        # best fitness
        strong_parent = sorted_pop[0]


        if self.species:

            strong_species = None
            for species in self.species:
                if strong_parent in species:
                    strong_species = species
                    break
            
            if strong_species and len(strong_species) > 1:
                # Select from same species
                weak_parent = random.choice([g for g in strong_species if g != strong_parent])
            else:
                # Fallback to random from population
                weak_parent = random.choice(self.population)
        else:
            # No speciation, random from population
            weak_parent = random.choice(self.population)

        return strong_parent, weak_parent

    def evolve(self, fn_evaluate):
        if len(self.population) == 0:
            self.init_population()

        """
        NEAT steps for each generation
        1. evaluate 
        2. speciation
        3. selection
        4. reproduction
            i. Crossover
            ii. Mutation
        """

        best_seen = None
        gen_best_list = []
        for _ in range(self.num_generations):
            # 1. Evaluate
            self.evaluate(fn_evaluate)

            # Track best of this evaluated population
            gen_best = max(self.population, key=lambda g: g.fitness_score)
            if best_seen is None or gen_best.fitness_score > best_seen.fitness_score:
                best_seen = gen_best
            gen_best_list.append(gen_best)

            # 2. Speciation
            self.speciate()
            
            # 3. Calculate adjusted fitness (fitness sharing)
            self.calculate_adjusted_fitness()

            # 4. Selection & Reproduction
            self.create_population()

        # Final evaluation of the last generated population
        self.evaluate(fn_evaluate)
        gen_best = max(self.population, key=lambda g: g.fitness_score)
        if best_seen is None or gen_best.fitness_score > best_seen.fitness_score:
            best_seen = gen_best

        return best_seen, gen_best_list