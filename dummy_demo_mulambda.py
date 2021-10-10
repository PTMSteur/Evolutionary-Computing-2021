import sys, os
sys.path.insert(0, 'evoman') 
from environment import Environment
from demo_controller import player_controller
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt
from collections.abc import Sequence
from itertools import repeat

class HiddenPrints:
    # https://stackoverflow.com/questions/8391411/how-to-block-calls-to-print
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

import random
import argparse
import multiprocessing
import numpy as np
import pandas as pd
import time

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--enemy', nargs='+', type=int, default=[1, 2, 3])          # Training Enemy IDs
parser.add_argument('--runs', type=int, default=10)                             # Number of runs per EA
parser.add_argument('--headless', type=bool, default=True)                      # If True, does not depict games on screen
parser.add_argument('--plot', type=bool, default=False)                         # If True, skips algorithm and just plots with latest CSVs

# Add these to the command to run with or without multiprocessing; By default multi is used
parser.add_argument('--multi', dest='multi', default=True, action='store_true')
parser.add_argument('--single', dest='single', default=False, action='store_false')

# Example use: 
# python dummy_demo_mulambda.py --enemy 1 2 3 --runs=10 --multi
FLAGS, unparsed = parser.parse_known_args()

# Initializes EvoMan environment
experiment_name = 'Task_II_Enemy_' + str(FLAGS.enemy)
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name) 
if FLAGS.headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    
controller_neurons = 10 
env = Environment(experiment_name=experiment_name,
                  multiplemode="yes" if len(FLAGS.enemy) > 1 else "no",
                  enemies=FLAGS.enemy,
                  playermode="ai",
                  player_controller=player_controller(controller_neurons),
                  randomini="yes",
                  enemymode="static",
                  level=2,
                  speed="fastest",
                  timeexpire = 500)

# Structure of the genotype of each individual, +1 for parameter control of the mutation step size
geno_length = (env.get_num_sensors()+1) * controller_neurons + (controller_neurons+1)*5 + 1

# Evaluation funtions for each individual; uses default returned by EvoMan for now
def fitness(ind):
    f,p,e,time = env.play(pcont=ind[:-1])
    return f

printlog = True # Print logbook stream if True; otherwise env.play() output
def evalIndividual(ind):
    if printlog:
        with HiddenPrints():
            f_avg = fitness(ind)
            return (f_avg,)
    else:
        f_avg = fitness(ind)
        return (f_avg,)

## Tweakable parameters
npop = 25                       # Population size
gens = 30                       # Nr of generations per run
elite_group = 5                 # Elite group size
mutation = 0.2                  # Mutation probability
cross = 0.8                     # Crossover probability
cpg = 75                        # Children per gen
delta = 12                      # Clearing allowed distance
n_top = 2                       # Clearing allowed group size
order = 1                       # Norm order for distance metric

# Setup DEAP Framework
tb = base.Toolbox()
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", np.ndarray, fitness = creator.FitnessMax)
    
tb.register("weights", np.random.uniform, -1, 1)
tb.register("individual", tools.initRepeat, creator.Individual, tb.weights, n=geno_length)
tb.register("population", tools.initRepeat, list, tb.individual, n=npop)
    
# Evaluation, Crossover, Mutation and Selection functions of choice
tb.register("evaluate", evalIndividual)
tb.register("mate", tools.cxBlend, alpha=0.5)
tb.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.5)
tb.register("select", tools.selTournament, tournsize=5)

# Logbook for keeping track of the statistics during te experiment
stats = tools.Statistics(key=lambda p: p.fitness.values)
stats.register("mean", np.mean)
stats.register("max", np.max)
logbook = tools.Logbook()


# Function for plotting the line plots for the average mean and maximum of the algorithm runs
def visualize(logbook):
    df = pd.read_csv(logbook)
    
    name1, name2 = df.ea.unique()
    df_c = df[df['ea']==name1].groupby('gen').mean()
    df_p = df[df['ea']==name2].groupby('gen').mean()
    
    plt.plot(df_c.index, df_c['max'], label='Avg. Max Fit '+name1)
    plt.plot(df_c.index, df_c['mean'], label='Avg. Mean Fit '+name1)
    plt.plot(df_p.index, df_p['max'], label='Avg. Max Fit '+name2)
    plt.plot(df_p.index, df_p['mean'], label='Avg. Mean Fit '+name2)
    plt.title("Avg. Fitness Stats over " + str(FLAGS.runs) + " Runs - Enemy " + str(FLAGS.enemy))
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend()
    plt.show()


def mutGaussian(individual, mu, sigma, indpb, control_sigma=True):
    """
	Modified Gaussian mutation from DEAP accounting for a sigma attribute in the genotype
    """
    if control_sigma is True:
        sigma = np.abs(individual[-1]) # Overwrite the global sigma with the one in the genotype
    
    # Need to repeat mu en sigma for every entry in the genotype
    size = len(individual)
    if not isinstance(mu, Sequence):
        mu = repeat(mu, size)
    elif len(mu) < size:
        raise IndexError("mu must be at least the size of individual: %d < %d" % (len(mu), size))
    if not isinstance(sigma, Sequence):
        sigma = repeat(sigma, size)
    elif len(sigma) < size:
        raise IndexError("sigma must be at least the size of individual: %d < %d" % (len(sigma), size))

    if control_sigma is True:
        for i, m, s in zip(range(size-1), mu, sigma):
            if random.random() < indpb:
                individual[i] += random.gauss(m, s)
        if random.random() < indpb:
            individual[-1] = np.abs(individual[-1])*np.exp(random.gauss(0, 1/np.sqrt(size-1)))
    else:
        for i, m, s in zip(range(size), mu, sigma):
            if random.random() < indpb:
                individual[i] += random.gauss(m, s)

    return individual,


def clearing_algorithm(offs):
    offs.sort(key=lambda x: x.fitness, reverse=True)
    
    distances = []
    for i in range(len(offs)):
        fit_i = offs[i].fitness.values[0]
        if fit_i > -9:
            winners = 1
            for j in range(i+1, len(offs)):
                fit_j = offs[j].fitness.values[0]
                dist = np.linalg.norm(np.array(offs[i][:-1])-np.array(offs[j][:-1]), ord=order)
                distances.append(dist)
                if fit_j > -9 and dist < delta:
                    if winners < n_top:
                        winners = winners + 1
                    else:
                        offs[j].fitness.values = (-10.0,)
    return offs

# Function to evaluate all (new) individuals that do not yet have a fitness  
def evalPopulation(pop, env):
    to_evaluate = [ind for ind in pop if not ind.fitness.valid]
    fit_map = tb.map( tb.evaluate, to_evaluate )
    for p, f in zip(pop, fit_map):
        p.fitness.values = f
    
    return pop, len(to_evaluate) * len(FLAGS.enemy)

# muLambda algorithm from DEAP adjusted for being able to have a protected elite group of individuals
def muLambda(runs, eatype='Comma', clearing=True):
    assert cpg >= npop, "Must create enough offspring to replace population"
    run_winners = pd.DataFrame(index=range(1,runs+1), columns=['w' + str(x) for x in range(1,geno_length+1)] )
        
    for r in range(1,runs+1):
        print("--- Run " + str(r) + " starts now ---")
        pop = tb.population(n=npop)
        pop, nevals = evalPopulation(pop, env)
        tevals = nevals
            
        hof = tools.HallOfFame(elite_group if elite_group > 0 else 1, similar=np.array_equal)
        hof.update(pop)
        record = stats.compile(pop)
        logbook.record(ea=eatype, nevals=tevals, run=r, gen=0, **record)
        if printlog:
            print(logbook.stream)
            
        # Generational loop
        for g in range(1,gens+1):
            offs = algorithms.varOr(pop,tb,cpg,cross,mutation)
            offs, nevals = evalPopulation(offs, env)
            tevals += nevals 
                
            if eatype=='Comma':
                offs = clearing_algorithm(offs) if clearing else offs
                pop = tb.select(offs, npop)
            elif eatype=='CommaElite':
                offs = clearing_algorithm(offs + hof.items) if clearing else offs
                pop = tb.select(offs, npop if clearing else npop - len(hof.items))
                pop = pop if clearing else pop + hof.items
                    
            hof.update(pop)
            record = stats.compile(pop)
            logbook.record(ea=eatype, nevals=tevals, run=r, gen=g, **record)
            if printlog:
                print(logbook.stream)
            
        # Assign the highest fitness individual from the run to a dataframe
        run_winners.loc[r,:] = hof.items[0]
    return run_winners


# Execute the algorithms
def experiment():    
    ini = time.time()
    
    if not FLAGS.plot:
        run_winners_EA1 = muLambda(FLAGS.runs, eatype='CommaElite', clearing=True) # Execute Algorithm 1
        run_winners_EA2 = muLambda(FLAGS.runs, eatype='CommaElite', clearing=False) # Execute Algorithm 2
        
    fim = time.time()
    print( '\nExecution time: '+str(round((fim-ini)/60))+' minutes \n')
    
    if not FLAGS.plot:
        # Write the best individual from each run to a csv    
        run_winners_EA1.to_csv('run_winners_EA1_enemy_' + str(FLAGS.enemy) + '_task_II.csv')
        run_winners_EA2.to_csv('run_winners_EA2_enemy_' + str(FLAGS.enemy) + '_task_II.csv')
        
        # Write the logbook to a csv
        log_df = pd.DataFrame(logbook)
        log_df.to_csv('logbook_enemy_' + str(FLAGS.enemy) + '_task_II.csv')
    
    # Visualization of logbook
    visualize('logbook_enemy_' + str(FLAGS.enemy) + '_task_II.csv')


# Make variables passable from command line for quick changes
if __name__ == '__main__':
    
    # Allow multiprocessing on the mapping of fitness evaluations to individuals, if desired
    if FLAGS.multi:
        pool = multiprocessing.Pool()
        tb.register("map", pool.map)
    
    experiment()