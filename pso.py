import numpy as np 
import math
import matplotlib.pyplot as plt 


def pso_init(domain_low, domain_high, particle_num = 100) :
    particle_group = np.random.uniform(domain_low, domain_high, (1, particle_num))
    velocity = np.random.uniform(-1, 1, (1, particle_num))
    weight = 0.8                      
    c1_learn = 0.5                    
    c2_learn = 0.5
    return particle_group, velocity, weight, c1_learn, c2_learn


def compute_fitness_test(particle_group) : 
    assert(particle_group.shape[0] == 1)
    fitness = np.multiply(np.multiply(particle_group, np.sin(particle_group)), 
                          np.cos(2*particle_group)) \
              - 2 * np.multiply(particle_group, np.sin(3 * particle_group))
    return fitness


def paticle_swarm_optimization(domain_low, domain_high, particle_num = 50, iteration_num = 100) :
    # 生成粒子群、初始化粒子群的相关参数
    particle_group, velocity, weight, c1_learn, c2_learn = \
    pso_init(domain_low, domain_high, particle_num)

    # 初始化粒子群的当前个体最优解、当前群体最优解
    particle_best = particle_group
    fitness = compute_fitness_test(particle_group)
    pbest_fitness = fitness
    gbest_fitness = np.max(fitness)
    group_best = particle_group[0][np.argmax(fitness)]

    # 进行迭代
    for cnt in range(0, iteration_num) : 
        velocity = weight * velocity + c1_learn * np.random.rand() * (particle_best - particle_group) \
                   + c2_learn * np.random.rand() * (group_best - particle_group)
        particle_group = particle_group + velocity
        fitness = compute_fitness_test(particle_group)
        
        for i in range(0, particle_num) : 
            if pbest_fitness[0][i] < fitness[0][i] : 
                pbest_fitness[0][i] = fitness[0][i]
                particle_best[0][i] = particle_group[0][i]
        
        if gbest_fitness < np.max(fitness) : 
            gbest_fitness = np.max(fitness)
            group_best = particle_group[0][np.argmax(fitness)]
    
    # 输出最后的群体最优解
    return particle_group, particle_best, group_best