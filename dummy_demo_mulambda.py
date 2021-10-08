# Import framework
import sys, os
sys.path.insert(0, 'evoman') 
from environment import Environment
from demo_controller import player_controller
from deap import base, creator, tools, algorithms
import scipy.stats as stats
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import argparse
import numpy as np
import pandas as pd
import time

class HiddenPrints:
    # https://stackoverflow.com/questions/8391411/how-to-block-calls-to-print
    
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        
# Plot the line plots for the average mean and maximum of the algorithm runs
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
    
def computeAvgGains(df, repeats, env):
    avg_gains = []
    for i in range(0,FLAGS.runs):
        gain_sum = 0
        for r in range(repeats):
            with HiddenPrints():
                f,p,e,t = env.play(pcont=np.array(df.iloc[i][1:]))
            gain_sum = gain_sum + p-e
        avg_gains.append(gain_sum/repeats)
            
    return avg_gains
    
# Let the individuals in the files play a repeats amount of times for a robust indication of their performance
def testSols(repeats, env, file1, file2, file3=None, file4=None, file5=None, file6=None):
    df_c1 = pd.read_csv(file1)
    df_p1 = pd.read_csv(file2)
    if file3:
        df_c2 = pd.read_csv(file3)
        df_p2 = pd.read_csv(file4)
        df_c3 = pd.read_csv(file5)
        df_p3 = pd.read_csv(file6)
    
    gains = []
    gains.append(computeAvgGains(df_c1, repeats, env))
    gains.append(computeAvgGains(df_p1, repeats, env))
    if file3:
        gains.append(computeAvgGains(df_c2, repeats, env))
        gains.append(computeAvgGains(df_p2, repeats, env))
        gains.append(computeAvgGains(df_c3, repeats, env))
        gains.append(computeAvgGains(df_p3, repeats, env))
    
    # Statistical test
    stat, pvalue = stats.ttest_ind(gains[0], gains[1])
    print('Student T-test p-value: ' + str(round(pvalue, 4)))
    
    # Boxplot function here
    
    
    
    b = sns.boxplot(data=[gains[0], gains[1]],
                    palette=[sns.xkcd_rgb["pale red"], sns.xkcd_rgb["medium green"]])
    
    b.set_title('Averaged Gain of Best Players per Run on Enemy ' + str(FLAGS.enemy))
    b.set_ylabel('Individual Gain')
    b.set_xticklabels(['CommaElite', 'Comma'])
    plt.show()
    
    
def experiment():
    experiment_name = 'Task_I_Enemy_' + str(FLAGS.enemy)
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)  
    if FLAGS.headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    
    ## Tweakable parameters
    ntest = 1                       # Plays per individual over which fitness is averaged; more than 1 will take a lot of time
    npop = 20                       # Population size
    gens = 30                       # Nr of generations per run
    elite_group = 2                 # Elite group size
    mutation = 0.2                  # Mutation probability
    cross = 0.8                     # Crossover probability
    cpg = 60                        # Children per gen
    controller_neurons = 10         # Hidden neurons in the controller
    printlog = True

    # Initializes environment
    env = Environment(experiment_name=experiment_name,
                         multiplemode="yes",
                         enemies=FLAGS.enemy,
                         playermode="ai",
                         player_controller=player_controller(controller_neurons),
                         randomini="yes",
                         enemymode="static",
                         level=2,
                         speed="fastest",
                         timeexpire = FLAGS.budget)
    
    # Structure of the genotype of each individual and their fitness evaluation function
    geno_length = (env.get_num_sensors()+1) * controller_neurons + (controller_neurons+1)*5
    
    # Change this if you wish
    # def fitness(p,e,t):
    #     return 0.9*(100 - e) + 0.1*p - np.log(t)
    
    def nTest(n, ind, env):
        score = 0
        fit = 0
        for t in range(n):
            f,p,e,time = env.play(pcont=ind)
            score = score + p-e
            fit += f
        return score/n, fit/n
        
    def evalIndividual(ind, env):
        if printlog:
            with HiddenPrints():
                g_avg, f_avg = nTest(ntest, ind, env)
                return (f_avg,)
        else:
            g_avg, f_avg = nTest(ntest, ind, env)
            return (f_avg,)
    
    ## DEAP Framework Setup
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", np.ndarray, fitness = creator.FitnessMax)
    
    tb = base.Toolbox()
    tb.register("weights", np.random.uniform, -1, 1)
    tb.register("individual", tools.initRepeat, creator.Individual, tb.weights, n=geno_length)
    tb.register("population", tools.initRepeat, list, tb.individual, n=npop)
    
    # Evaluation, Crossover, Mutation and Selection functions of choice
    tb.register("evaluate", evalIndividual)
    tb.register("mate", tools.cxBlend, alpha=0.5)
    tb.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.5)
    tb.register("select", tools.selTournament, tournsize=4)
    
    # Logbook for keeping track of the statistics
    stats = tools.Statistics(key=lambda p: p.fitness.values)
    stats.register("mean", np.mean)
    stats.register("max", np.max)
    logbook = tools.Logbook()
    
    # Function to evaluate all (new) individuals that do not yet have a fitness  
    def evalPopulation(g, pop, env):
        to_evaluate = [ind for ind in pop if not ind.fitness.valid]
        fit_map = tb.map( tb.evaluate, to_evaluate, [env for i in range(len(pop))] )
        for p, f in zip(pop, fit_map):
            p.fitness.values = f
            p.origin = g
            
        return pop, len(to_evaluate) * ntest * len(FLAGS.enemy)
     
    # muLambda algorithm adjusted for having a protected elite group
    def muCommaLambda(runs, eatype='Comma'):
        assert cpg >= npop, "Must create enough offspring to replace population"
            
        run_winners = pd.DataFrame(index=range(1,runs+1), columns=['w' + str(x) for x in range(1,geno_length+1)] )
        for r in range(1,runs+1):
            print("--- Run " + str(r) + " starts now ---")
            pop = tb.population(n=npop)
            pop, nevals = evalPopulation(0, pop, env)
            tevals = nevals
            
            hof = tools.HallOfFame(elite_group if elite_group > 0 else 1, similar=np.array_equal)
            hof.update(pop)
            
            record = stats.compile(pop)
            logbook.record(ea=eatype, nevals=tevals, run=r, gen=0, **record)
            if printlog:
                print(logbook.stream)
            
            for g in range(1,gens+1):
                offs = algorithms.varOr(pop,tb,cpg,cross,mutation)
                offs, nevals = evalPopulation(g, offs, env)
                tevals += nevals 
                
                # hof.update(offs)
                
                # The only difference between the options is this if statement
                if eatype == 'CommaElite':
                    pop = tb.select(offs, npop-len(hof.items) if elite_group > 0 else npop)
                    pop = pop + hof.items
                elif eatype == 'Comma':
                    pop = tb.select(offs, npop)
                elif eatype == 'Plus':
                    pop = tb.select(pop + offs, npop)
                    
                hof.update(pop)
                    
                record = stats.compile(pop)
                logbook.record(ea=eatype, nevals=tevals, run=r, gen=g, **record)
                
                if printlog:
                    print(logbook.stream)
            
            # Since the final population is less likely to be in a local minimum than earlier populations,
            # We favour younger individuals for the position of best player,
            # If they have a winning fitness and aren't much worse than the best fitness ever seen
            pop.sort(key=lambda x: x.fitness.values, reverse=True)
            if pop[0].fitness.values[0] > 90 and pop[0].fitness.values[0] > hof.items[0].fitness.values[0] - 1:
                run_winners.loc[r,:] = pop[0]
            else:
                run_winners.loc[r,:] = hof.items[0]
                
        return run_winners
    
    # Execute Code
    ini = time.time()
    if not FLAGS.test:
        run_winners_c = muCommaLambda(FLAGS.runs) # Execute Algorithm 1
        run_winners_p = muCommaLambda(FLAGS.runs) # Execute Algorithm 2
    fim = time.time()
    print( '\nExecution time: '+str(round((fim-ini)/60))+' minutes \n')
    
    if not FLAGS.test:
        # Write the best individual from each run to a csv    
        run_winners_c.to_csv('run_winners_Comma_enemy_' + str(FLAGS.enemy) + '_task_II.csv')
        run_winners_p.to_csv('run_winners_Comma_enemy_' + str(FLAGS.enemy) + '_task_II.csv')
        # Write the logbook to a csv
        log_df = pd.DataFrame(logbook)
        log_df.to_csv('logbook_enemy_' + str(FLAGS.enemy) + '_task_II.csv')
    
    # Visualization and Repeated Testing
    visualize('logbook_enemy_' + str(FLAGS.enemy) + '_task_II.csv')
        
    testSols(repeats=5, env=env, 
             file1='run_winners_Comma_enemy_' + str(FLAGS.enemy) + '_task_II.csv',
             file2='run_winners_Comma_enemy_' + str(FLAGS.enemy) + '_task_II.csv'
             )


# Make variables passable from command line for quick changes
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--enemy', type=list, default=[4,6,7])
    parser.add_argument('--budget', type=int, default=500)
    parser.add_argument('--headless', type=bool, default=True)  # If True, does not depict games on screen
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--test', type=bool, default=False) # If True, skips algorithm and just plots with latest CSVs
    
    FLAGS, unparsed = parser.parse_known_args()
    
    experiment()