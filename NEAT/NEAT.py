import random
from .config import config

from .genotype import Genotype, crossover

class NEAT:
    def __init__(self, fn_fitness):
        self.num_input = config.get("input_nodes")
        self.num_output = config.get("output_nodes")
        self.generation = 0
        self.population_size = config.get("population_size")

        self.fn_fitness = fn_fitness
        self.population = []
        self.species = []
        self.species_representatives = []

        self.num_generations = config.get("num_generations")

        self.input_nodes = config.get("input_nodes")
        self.output_nodes = config.get("output_nodes")

    def print_params(self):
        print(f"Number of species: {len(self.species)}")

        for species in self.species_representatives:
            print("Species representatives:")
            print(f"Number of hidden layers {len(species.hidden_layers)}")
            for hidden_layer in species.hidden_layers:
                print(f"Hidden layer nodes: {len(hidden_layer)}")

            print("---" * 10)

    def init_population(self):
        self.population = []
        for _ in range(self.population_size):
            genotype = Genotype(self.input_nodes, self.output_nodes)
            self.population.append(genotype)

    def run(self, fn_step):
        input_values = fn_step()

    def evaluate(self, fn_eval):
        for g in self.population:
            g.fitness_score = fn_eval(g)

    def compatibility_distance(self, g1, g2):
        return g1.compatibility_distance(g2)

    def speciate(self):
        compatibility_threshold = config.get("compatibility_threshold")
        
        if not self.species_representatives:
            self.species = []
            self.species_representatives = []
            for genotype in self.population:
                assigned = False
                for i, representative in enumerate(self.species_representatives):
                    distance = genotype.compatibility_distance(representative)

                    if distance < compatibility_threshold:
                        self.species[i].append(genotype)
                        assigned = True
                        break
                
                if not assigned:
                    self.species.append([genotype])
                    self.species_representatives.append(genotype)
        else:
            self.species = [[] for _ in self.species_representatives]
            
            for genotype in self.population:
                assigned = False
                for i, representative in enumerate(self.species_representatives):
                    distance = genotype.compatibility_distance(representative)

                    if distance < compatibility_threshold:
                        self.species[i].append(genotype)
                        assigned = True
                        break
                
                if not assigned:
                    self.species.append([genotype])
                    self.species_representatives.append(genotype)
            
            non_empty_species = []
            non_empty_representatives = []
            for i, species in enumerate(self.species):
                if len(species) > 0:
                    non_empty_species.append(species)
                    best_in_species = max(species, key=lambda g: g.fitness_score)
                    non_empty_representatives.append(best_in_species)
            
            self.species = non_empty_species
            self.species_representatives = non_empty_representatives


    def calculate_adjusted_fitness(self):
        for species in self.species:
            species_size = len(species)
            for genotype in species:
                genotype.adjusted_fitness = genotype.fitness_score / species_size

    def create_population(self):
        if not self.population:
            self.init_population()
            return

        new_population = []

        if not self.species or len(self.species) == 0:
            sorted_pop = sorted(self.population, key=lambda g: g.fitness_score, reverse=True)
            survivors_count = max(1, int(self.population_size * config.get("population_cut")))
            breeders = sorted_pop[:survivors_count]
            
            elite = crossover(breeders[0], breeders[0])
            new_population.append(elite)
            
            while len(new_population) < self.population_size:
                strong = random.choice(breeders)
                weak = random.choice(breeders)
                child = crossover(strong, weak)
                child.mutate()
                new_population.append(child)
        else:
            species_adjusted_fitness = []
            for species in self.species:
                species_total = sum(g.adjusted_fitness for g in species)
                species_adjusted_fitness.append(species_total)
            
            total_adjusted_fitness = sum(species_adjusted_fitness)
            
            best_genotype = max(self.population, key=lambda g: g.fitness_score)
            elite = crossover(best_genotype, best_genotype)
            new_population.append(elite)
            
            while len(new_population) < self.population_size:
                if total_adjusted_fitness > 0:
                    rand_val = random.uniform(0, total_adjusted_fitness)
                    cumulative = 0
                    selected_species_idx = 0
                    for i, species_fitness in enumerate(species_adjusted_fitness):
                        cumulative += species_fitness
                        if rand_val <= cumulative:
                            selected_species_idx = i
                            break
                else:
                    selected_species_idx = random.randint(0, len(self.species) - 1)
                
                selected_species = self.species[selected_species_idx]
                
                if len(selected_species) >= 2:
                    sorted_species = sorted(selected_species, key=lambda g: g.fitness_score, reverse=True)
                    strong = sorted_species[0]
                    weak = random.choice(sorted_species[1:]) if len(sorted_species) > 1 else sorted_species[0]
                elif len(selected_species) == 1:
                    strong = selected_species[0]
                    weak = selected_species[0]
                else:
                    strong = random.choice(self.population)
                    weak = random.choice(self.population)
                
                child = crossover(strong, weak)
                child.mutate()
                new_population.append(child)

        self.population = new_population
        self.generation += 1

    def select_parents(self):

        sorted_pop = sorted(self.population, key=lambda g: g.fitness_score, reverse=True)

        strong_parent = sorted_pop[0]


        if self.species:

            strong_species = None
            for species in self.species:
                if strong_parent in species:
                    strong_species = species
                    break
            
            if strong_species and len(strong_species) > 1:
                weak_parent = random.choice([g for g in strong_species if g != strong_parent])
            else:
                weak_parent = random.choice(self.population)
        else:
            weak_parent = random.choice(self.population)

        return strong_parent, weak_parent

    def evolve(self, fn_evaluate):
        if len(self.population) == 0:
            self.init_population()

        best_seen = None
        gen_best_list = []
        for _ in range(self.num_generations):
            self.evaluate(fn_evaluate)

            gen_best = max(self.population, key=lambda g: g.fitness_score)
            if best_seen is None or gen_best.fitness_score > best_seen.fitness_score:
                best_seen = gen_best
            gen_best_list.append(gen_best)

            self.speciate()
            
            self.calculate_adjusted_fitness()

            self.create_population()

        self.evaluate(fn_evaluate)
        gen_best = max(self.population, key=lambda g: g.fitness_score)
        if best_seen is None or gen_best.fitness_score > best_seen.fitness_score:
            best_seen = gen_best

        return best_seen, gen_best_list