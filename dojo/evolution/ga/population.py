from .chromosome import Chromosome


class Population:

    def __init__(self, pop_size, mutation_rate):
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate

        self.pop_arr = []
        self.init_population()

    def init_population(self):
        self.pop_arr = [Chromosome(self.mutation_rate) for _ in range(self.pop_size)]
    
    def calc_fitness(self):
        for chromosome in self.pop_arr:
            