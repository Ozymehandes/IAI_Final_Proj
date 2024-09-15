import os
import sys
import random
import numpy as np
import csv

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import traci
from sumo_rl import SumoEnvironment
from evaluation import *

class GeneticAlgorithm:
    """
    A class that implements a Genetic Algorithm for optimizing traffic light control.

    Attributes:
        env (SumoEnvironment): The SUMO environment.
        population_size (int): The number of individuals in each generation.
        generations (int): The number of generations to run the algorithm.
        elitism_rate (float): The proportion of top individuals to keep in each generation.
        chromosome_length (int): The length of each individual's chromosome (action sequence).
        population (numpy.ndarray): The current population of individuals.
    """
    def __init__(self, env, population_size=50, generations=100, elitism_rate=0.1, num_seconds=100000, population=None):
        self.env = env
        self.population_size = population_size
        self.generations = generations
        self.elitism_rate = elitism_rate
        self.chromosome_length = num_seconds
        self.population = population


    def fitness(self, individual):
        """
        Calculates the fitness of an individual.

        Args:
            individual (numpy.ndarray): The individual to evaluate.

        Returns:
            float: The average mean waiting time (lower is better).
        """
        obs = self.env.reset()
        total_mean_wait_time = 0
        step_count = 0
        done = False
        
        for action in individual:
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            total_mean_wait_time += info['system_mean_waiting_time']
            step_count += 1
            if done:
                break
        
        avg_mean_wait_time = total_mean_wait_time / step_count if step_count > 0 else float('inf')
        return avg_mean_wait_time

    def crossover(self, parent1, parent2):
        """
        Performs crossover between two parents to create two children.

        Args:
            parent1 (numpy.ndarray): The first parent.
            parent2 (numpy.ndarray): The second parent.

        Returns:
            tuple: Two children created from the parents.
        """
        crossover_point = np.random.randint(1, len(parent1) - 1)
        child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
        return child1, child2

    def mutate(self, individual, mutation_rate=0.01):
        """
        Applies mutation to an individual.

        Args:
            individual (numpy.ndarray): The individual to mutate.
            mutation_rate (float): The probability of each gene being mutated.

        Returns:
            numpy.ndarray: The mutated individual.
        """
        mutation_mask = np.random.random(len(individual)) < mutation_rate
        individual[mutation_mask] = np.random.randint(0, self.env.action_space.n, size=np.sum(mutation_mask))
        return individual

    def run(self, path=''):
        """
        Runs the genetic algorithm learning process.

        Args:
            path (str): The path to save output files.

        Returns:
            tuple: The final population, final fitnesses, best individual, and its index.
        """
        with open(path + 'ga_evaluation.csv', 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['Generation', 'Average Waiting Time(Fitness)'])

        if self.population is None:
            self.population = np.random.randint(0, self.env.action_space.n, size=(self.population_size, self.chromosome_length))

        population = self.population
        best_ever_individual = None
        best_ever_fitness = float('inf')
        for generation in range(1, self.generations + 1):
            if generation != 1:
                population = np.load(path + f'population_{generation-1}.npy')

            fitnesses = np.array([self.fitness(ind) for ind in population])

            # Update best ever individual if necessary
            current_best_index = np.argmin(fitnesses)
            if fitnesses[current_best_index] < best_ever_fitness:
                best_ever_individual = population[current_best_index]
                best_ever_fitness = fitnesses[current_best_index]

    
            # Find the best individuals
            num_elite = int(self.elitism_rate * self.population_size) - 1  # Reserve one spot for best_ever_individual
            elite_indices = np.argsort(fitnesses)[:num_elite]
            elite = population[elite_indices]

            with open(path + 'ga_evaluation.csv', 'a', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([generation, best_ever_fitness])
            
            # Select parents using adjusted fitnesses
            parent_indices = np.random.choice(self.population_size, size=self.population_size - num_elite - 1)
            parents = population[parent_indices]
            
            # Create next generation
            next_generation = np.empty((self.population_size, self.chromosome_length), dtype=int)
            next_generation[0] = best_ever_individual  # Always include the best ever individual
            next_generation[1:num_elite+1] = elite
            
            for i in range(num_elite+1, self.population_size - 1, 2):
                parent1, parent2 = parents[i-num_elite-1], parents[i-num_elite]
                child1, child2 = self.crossover(parent1, parent2)
                next_generation[i] = self.mutate(child1)
                if i + 1 < self.population_size:
                    next_generation[i+1] = self.mutate(child2)
            
            population = next_generation
            np.save(f'{path}population_{generation}.npy', population)
            np.save(f'{path}fitnesses_{generation}.npy', fitnesses)
            
        # Final evaluation to find the best individual
        final_fitnesses = np.array([self.fitness(ind) for ind in population])
        best_individual_index = np.argmin(final_fitnesses)
        best_individual = population[best_individual_index]
        return population, final_fitnesses, best_individual, best_individual_index


def train_genetic(net_file, route_file, out_csv_name=None, path='', use_gui=False, num_seconds=N_SECONDS, population=None, generations=100, elitism_rate=0.1, population_size=50):
    """
    Trains the genetic algorithm on a SUMO environment.

    Args:
        net_file (str): Path to the SUMO network file.
        route_file (str): Path to the SUMO route file.
        out_csv_name (str, optional): Name of the output CSV file.
        path (str): Path to save output files.
        use_gui (bool): Whether to use SUMO GUI.
        num_seconds (int): Number of seconds to run each simulation.
        population (numpy.ndarray, optional): Initial population.
        generations (int): Number of generations to run.
        elitism_rate (float): Elitism rate for the genetic algorithm.
        population_size (int): Size of the population.

    Returns:
        tuple: The final generation, final fitnesses, best individual, and its index.
    """
    env = create_env_with_reward(net_file=net_file, route_file=route_file, out_csv_name=out_csv_name, reward_fn=system_mean_waiting_time_reward)
    ga = GeneticAlgorithm(env, num_seconds=num_seconds, population=population, generations=generations, elitism_rate=elitism_rate, population_size=population_size)
    final_generation, final_fitnesses, best_individual, best_individual_index = ga.run(path=path)
    np.savez(f'{path}ga_results.npz', 
             final_generation=final_generation,
             final_fitnesses=final_fitnesses,
             best_individual=best_individual,
             best_individual_index=best_individual_index)
    return final_generation, final_fitnesses, best_individual, best_individual_index

def load_genetic_results(path):
    """
    Loads the results of a genetic algorithm run.

    Args:
        path (str): Path to the results file.

    Returns:
        tuple: The final generation, final fitnesses, best individual, and its index.
    """
    results = np.load(f'{path}')
    final_generation = results['final_generation']
    final_fitnesses = results['final_fitnesses']
    best_individual = results['best_individual']
    best_individual_index = results['best_individual_index']
    return final_generation, final_fitnesses, best_individual, best_individual_index

if __name__ == "__main__":
    np.random.seed(SEED)
    # Running this will overwrite current results, change paths to avoid this
    train_genetic(NET_FILE, ROUTE_FILE, out_csv_name='outputs/2way-single-intersection/genetic/genetic', path='outputs/2way-single-intersection/genetic/',
                   use_gui=False, num_seconds=N_SECONDS, population=None, generations=24, elitism_rate=0.1, population_size=50)