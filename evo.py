import torch
import random
from net_01 import *
#from net_02 import *


def popu_sort(population, popu_fitness): # sort population + fitness
    
    popu_sorted = dict()
    fit_sorted = dict()

    key_num = 0
    for key, value in sorted(popu_fitness.items(), key = lambda item:item[1], reverse = True):
        popu_sorted[key_num] = population[key]
        fit_sorted[key_num] = popu_fitness[key]
        key_num = key_num + 1


    return popu_sorted, fit_sorted


def popu_init(population_size, size_U,size_B, num_U, num_B, n_gpu, n_seed, n_saveP, n_epoch, n_batch):  # size_U: size of unitary op, size_B: size of binary op
    print("----->Initializing Population")
    population = dict()
    popu_fitness = dict()
    
    num_popu = 0

    
    while num_popu < population_size:
        
        Un_op = torch.randint(0, num_U,[size_U])
        Bi_op = torch.randint(0, num_B,[size_B])
        UB_op = torch.cat((Un_op, Bi_op),-1)

        
        flag = True
        for hh in range(len(population)):
            if torch.equal(population[hh], UB_op):
                flag = False
                print('--- already existed----')
        
        if flag:
            print(UB_op)
            op_score = fitness_score(gpu=n_gpu, seed = n_seed, UBs = UB_op, save_p = n_saveP, epochs = n_epoch, batch_size = n_batch)
            if op_score >11:
                population[num_popu] = UB_op
                popu_fitness[num_popu] = op_score
                print(population)
                print(popu_fitness)
                num_popu = num_popu +1
    

    popu_sorted, fit_sorted = popu_sort(population, popu_fitness)

    
    return popu_sorted, fit_sorted




def selection(k, population, popu_fitness):
    num_population = len(population)
    if k == 0:                                              # elitism selection
        #print("----->Elitism selection")
        return population[0], population[1]
    elif k == 1:                                            # tournament selection
        #print("----->Tournament selection")
        i = random.randint(0, num_population - 1)
        j = i
        while j < num_population - 1:
            j += 1
            if random.randint(1, 100) <= 50:
                return population[i], population[j]
        return population[i], population[0]
    else:                                                   # proportionate selection
        #print("----->Proportionate selection")
        cum_sum = 0
        for i in range(num_population):
            cum_sum += popu_fitness[i]
        perc_range = []
        for i in range(num_population):
            count = int(100 * popu_fitness[i] / cum_sum)
            for j in range(count):
                perc_range.append(i)

        i = random.randint(0, len(perc_range)-1)
        j = random.randint(0, len(perc_range)-1)
        while i == j:
            j = random.randint(0, len(perc_range)-1)
        
        return population[perc_range[i]], population[perc_range[j]]


def crossover(parent1, parent2):
    print("----->Crossover")
    size_ind = len(parent1)


    if random.randint(0, 1):
        first = parent1
        second = parent2
    else:
        first = parent2
        second = parent1

    c_num = random.randint(1, size_ind-2)
    
    child = torch.cat((first[:c_num], second[c_num:]),-1)

    return child

def mutation(indi,t,size_U, num_U, num_B):
    print('---->> mutation------')
    individual = indi.clone()
    size_ind = len(individual)
    if t == 1:
        num_m = random.randint(0, size_ind-1)

        temp = individual[num_m]
        if num_m < size_U:
            mutated = random.randint(0, num_U-1)
            while temp == mutated:
                mutated = random.randint(0, num_U-1)
        else:
            mutated = random.randint(0, num_B-1)
            while temp == mutated:
                mutated = random.randint(0, num_B-1)

        individual[num_m] = mutated

        
    elif t == 2:
        i = random.randint(0, size_ind-1)
        j = random.randint(0, size_ind-1)
        while i == j:
            j = random.randint(0, size_ind-1)
        temp = individual[i]
        if i < size_U:
            mutated = random.randint(0, num_U-1)
            while temp == mutated:
                mutated = random.randint(0, num_U-1)
        else:
            mutated = random.randint(0, num_B-1)
            while temp == mutated:
                mutated = random.randint(0, num_B-1)
        individual[i] = mutated

        temp = individual[j]
        if j < size_U:
            mutated = random.randint(0, num_U-1)
            while temp == mutated:
                mutated = random.randint(0, num_U-1)
        else:
            mutated = random.randint(0, num_B-1)
            while temp == mutated:
                mutated = random.randint(0, num_B-1)
        individual[j] = mutated

        
    else:
        i = random.randint(0, size_ind-1)
        j = random.randint(0, size_ind-1)
        k = random.randint(0, size_ind-1)
       
        while i == j:
            j = random.randint(0, size_ind-1)
        while ((k ==i)|(k ==j)):
            k = random.randint(0, size_ind-1)

        temp = individual[i]
        if i < size_U:
            mutated = random.randint(0, num_U-1)
            while temp == mutated:
                mutated = random.randint(0, num_U-1)
        else:
            mutated = random.randint(0, num_B-1)
            while temp == mutated:
                mutated = random.randint(0, num_B-1)
        individual[i] = mutated

        temp = individual[j]
        if j < size_U:
            mutated = random.randint(0, num_U-1)
            while temp == mutated:
                mutated = random.randint(0, num_U-1)
        else:
            mutated = random.randint(0, num_B-1)
            while temp == mutated:
                mutated = random.randint(0, num_B-1)
        individual[j] = mutated

        temp = individual[k]
        if k < size_U:
            mutated = random.randint(0, num_U-1)
            while temp == mutated:
                mutated = random.randint(0, num_U-1)
        else:
            mutated = random.randint(0, num_B-1)
            while temp == mutated:
                mutated = random.randint(0, num_B-1)
        individual[k] = mutated

        
    return individual


    

