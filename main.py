import torch
from evo import *
import random
from net_01 import *

import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.autograd import Variable
from data import get_dataset
from preprocess import get_transform
from utils import *
from datetime import datetime

import torchvision.transforms as transforms


def main():

    # param:
    p_gpu = 4
    p_seed = 2
    p_save = './result/Encode_1/'
    p_epoch = 15
    p_batch = 128

    p_gen = 200
    p_child = 10

    p_popu = 30
    p_sizeU = 2
    p_sizeB = 1
    p_numU = 22
    p_numB = 11
    
    All_num = 1
    


    
    save_path = os.path.join(p_save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    setup_logging(os.path.join(save_path, 'log.txt'))
    logging.info("saving to %s", save_path)
    
    
    population = dict()
    fitness = dict()
    population, fitness = popu_init(p_popu, p_sizeU, p_sizeB, p_numU, p_numB, p_gpu, p_seed, p_save, p_epoch, p_batch)
     
        
    
    logging.info('>>>>>>>>>> random init population  finished >>>>>>>>>>')
    logging.info(population)
    logging.info(fitness)
    
    for gen in range(1, p_gen + 1):

        '''
            k is the selection parameter:
                k = 0 -> elitism selection
                k = 1 -> tournament selection
                k = 2 -> proportionate selection
        '''
        k = random.randint(0, 2)

        for c in range(p_child):

            print("\nCreating Child", c)

            parent1, parent2 = selection(k, population, fitness)                 # selection
            
            if random.randint(0,1):
                child_1 = parent1
            else:
                child_1 = parent2

            
            child_2 = mutation(child_1, 1, p_sizeU, p_numU, p_numB)
            print('----mutated---')
            print(child_2)
            
            flag = True
            for hh in range(len(population)):
                if torch.equal(population[hh], child_2):
                    flag = False
                    print('--- already existed----')
        
            if flag:
                op_score = fitness_score(gpu=p_gpu, seed = p_seed, UBs = child_2, save_p = p_save, epochs = p_epoch, batch_size = p_batch)
                if op_score > fitness[p_popu-1]:
                    population[p_popu-1] = child_2
                    fitness[p_popu-1] = op_score
                    print(population)
                    print(fitness)
                    popu_sorted, fit_sorted = popu_sort(population, fitness)

                    population = popu_sorted
                    fitness = fit_sorted

                    logging.info('Evolution: %d', All_num)
                    All_num = All_num + 1
                    logging.info(population)
                    logging.info(fitness)


if __name__ == '__main__':
    main()
