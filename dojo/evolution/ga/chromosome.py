__all__ = [
    "Chromosome",
]


class Chromosome:
    
    @staticmethod
    def genotype_generator(seed):
        pass

    @staticmethod
    def calc_fitness(chromo):
        pass

    @staticmethod
    def crossover(chromoA, chromoB):
        pass

    @staticmethod
    def mutate(chromo):
        pass
    
    def __init__(self, mutation_rate):
        self.mutation_rate = mutation_rate

        self.genotype = Chromosome.genotype_generator(42)
        self.fitness_score = 0
