"""
Created on Sun Dec 13 19:13:35 2020

@author: migue
"""

import array
import random
import numpy
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
import pandas as pd

lista=[]
lista2=[]
datos=pd.read_csv('ruta.csv',header = 0)

for j in range(4):
    fila=datos.iloc[j]
    for i in range(4):
        lista2.append(fila[i])
    lista.append(lista2)
    lista2=[]
print(lista)
# print(lista[1])
distance_map = lista

IND_SIZE = 4

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", array.array, typecode='i', fitness=creator.FitnessMin)

toolbox = base.Toolbox()

toolbox.register("indices", random.sample, range(IND_SIZE), IND_SIZE)

toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evalTSP(individual):
    distance = distance_map[individual[-1]][individual[0]]
    for gene1, gene2 in zip(individual[0:-1], individual[1:]):
        distance += distance_map[gene1][gene2]
    return distance,

toolbox.register("mate", tools.cxPartialyMatched)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=30)
toolbox.register("evaluate", evalTSP)

def main():
    random.seed(169)

    pop = toolbox.population(n=50)

    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)
    
    algorithms.eaSimple(pop, toolbox, 0.7, 0.2, 50, stats=stats, 
                        halloffame=hof)
    
    return pop, stats, hof

if __name__ == "__main__":
    pop,stats,hof=main()
    print(hof)
    print(evalTSP(hof[0]))






