import neat
import numpy as np
from evojax.algo.base import NEAlgorithm
from jax import numpy as jnp
from neat.activations import tanh_activation
from neat.aggregations import sum_aggregation
from neat.nn import FeedForwardNetwork
from neat.population import CompleteExtinctionException
from typing import Union


ACTIVATION_TO_INT = {
    tanh_activation: 0,
}
INT_TO_ACTIVATION = {
    0: tanh_activation,
}

AGGREGATION_TO_INT = {
    sum_aggregation: 0,
}
INT_TO_AGGREGATION = {
    0: sum_aggregation
}


class NEAT(NEAlgorithm):
    def __init__(self, config_path, max_nodes, max_connections_per_node):
        self.config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
            neat.DefaultStagnation, config_path)
        self.pop_size = self.config.pop_size
        self.max_nodes = max_nodes
        self.max_connections_per_node = max_connections_per_node
        self.pop = neat.Population(self.config)
        self.pop.add_reporter(neat.StatisticsReporter())
        self.pop.add_reporter(neat.StdOutReporter(True))

    def class_to_array(self, genome: neat.genome.DefaultGenome) -> jnp.ndarray:
        genome = FeedForwardNetwork.create(genome, self.config)
        arr = jnp.full((self.max_nodes, 2 + 2 * self.max_connections_per_node), jnp.nan)
        for node, act_func, agg_func, bias, response, links in genome.node_evals:
            links = [item for link in links for item in link]
            arr = arr.at[node, 0].set(bias)
            arr = arr.at[node, 1].set(response)
            arr = arr.at[node, 2:(2 + len(links))].set(links)
        return arr

    def ask(self) -> jnp.ndarray:
        self.genome_keys, params = [], []
        for genome_key, genome in self.pop.population.items():
            self.genome_keys.append(genome_key)
            params.append(self.class_to_array(genome))
        return jnp.stack(params)

    def tell(self, fitness: Union[np.ndarray, jnp.ndarray]) -> None:
        self.pop.reporters.start_generation(self.pop.generation)

        for i, fitness_elem in enumerate(fitness):
            self.pop.population[self.genome_keys[i]].fitness = fitness_elem.item()

        best = None
        for g in self.pop.population.values():
            if g.fitness is None:
                raise RuntimeError("Fitness not assigned to genome {}".format(g.key))

            if best is None or g.fitness > best.fitness:
                best = g
        self.pop.reporters.post_evaluate(self.pop.config, self.pop.population, self.pop.species, best)

        # Track the best genome ever seen.
        if self.pop.best_genome is None or best.fitness > self.pop.best_genome.fitness:
            self.pop.best_genome = best

        # Create the next generation from the current generation.
        self.pop.population = self.pop.reproduction.reproduce(self.pop.config, self.pop.species,
            self.pop.config.pop_size, self.pop.generation)

        # Check for complete extinction.
        if not self.pop.species.species:
            self.pop.reporters.complete_extinction()

            # If requested by the user, create a completely new population,
            # otherwise raise an exception.
            if self.pop.config.reset_on_extinction:
                self.population = self.pop.reproduction.create_new(self.pop.config.genome_type,
                    self.pop.config.genome_config, self.pop.config.pop_size)
            else:
                raise CompleteExtinctionException()

        # Divide the new population into species.
        self.pop.species.speciate(self.pop.config, self.pop.population, self.pop.generation)

        self.pop.reporters.end_generation(self.pop.config, self.pop.population, self.pop.species)

        self.pop.generation += 1

        if self.pop.config.no_fitness_termination:
            self.pop.reporters.found_solution(self.pop.config, self.pop.generation, self.pop.best_genome)