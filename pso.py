import numpy as np 
import math
import matplotlib.pyplot as plt 


def pso_init(domain_low, domain_high, dimension = 2, particle_num = 100) :
    particle_group = np.random.uniform(domain_low, domain_high, (dimension, particle_num))
    velocity = np.random.uniform(-1, 1, (dimension, particle_num))    
    weight = 0.8                                                
    c1_learn = 0.5                        
    c2_learn = 0.5
    return particle_group, velocity, weight, c1_learn, c2_learn


def compute_fitness_f1(particle_group) : 
    particle_num = particle_group.shape[1]
    fitness = np.zeros((1, particle_num))
    for i in range(0, particle_num) : 
        particle_x = particle_group[0][i]
        particle_y = particle_group[1][i]
        fitness[0][i] = math.sin(particle_x) * math.sin(particle_y) / (particle_x * particle_y)
    return fitness


def compute_fitness_f2(particle_group) :
    particle_num = particle_group.shape[1]
    fitness = np.zeros((1, particle_num))
    for i in range(0, particle_num) :
        particle_x = particle_group[0][i]
        particle_y = particle_group[1][i]
        fitness[0][i] = 1 / (100 * ((particle_y - particle_x ** 2) ** 2) 
                        + (1 - particle_x) ** 2 + 1)
    return fitness


def compute_result_f2(particle_group) :
    particle_num = particle_group.shape[1]
    result = np.zeros((1, particle_num))
    for i in range(0, particle_num) :
        particle_x = particle_group[0][i]
        particle_y = particle_group[1][i]
        result[0][i] = 100 * ((particle_y - particle_x ** 2) ** 2) + (1 - particle_x) ** 2 
    return result


def velocity_check(velocity) : 
    particle_num = velocity.shape[1]
    var_num = velocity.shape[0]
    for i in range(0, particle_num) :
        for j in range(0, var_num) :
            if velocity[j][i] > 1 : 
                velocity[j][i] = 1
            elif velocity[j][i] < -1 : 
                velocity[j][i] = -1        
    return velocity


def particle_check_f1(particle_group, domain_low, domain_high) : 
    particle_num = particle_group.shape[1]
    var_num = particle_group.shape[0]
    for i in range(0, particle_num) : 
        for j in range(0, var_num) : 
            if particle_group[j][i] > domain_high : 
                particle_group[j][i] = domain_high
            elif particle_group[j][i] < domain_low : 
                particle_group[j][i] = domain_low
            elif particle_group[j][i] == 0 : 
                particle_group[j][i] = 0.0001
    return particle_group


def particle_check_f2(particle_group, domain_low, domain_high) : 
    particle_num = particle_group.shape[1]
    var_num = particle_group.shape[0]
    for i in range(0, particle_num) : 
        for j in range(0, var_num) : 
            if particle_group[j][i] > domain_high : 
                particle_group[j][i] = domain_high
            elif particle_group[j][i] < domain_low : 
                particle_group[j][i] = domain_low
    return particle_group


def gbest_particle_f1(gbest_fitness, particle_group) : 
    particle_num = particle_group.shape[1]
    var_num = particle_group.shape[0]
    for i in range(0, particle_num) : 
        particle_x = particle_group[0][i]
        particle_y = particle_group[1][i]
        fitness = math.sin(particle_x) * math.sin(particle_y) / (particle_x * particle_y)
        if fitness == gbest_fitness : 
            break
    gbest = np.zeros((var_num, 1))
    gbest[0][0] = particle_x
    gbest[1][0] = particle_y
    return gbest


def gbest_particle_f2(gbest_fitness, particle_group) : 
    particle_num = particle_group.shape[1]
    var_num = particle_group.shape[0]
    for i in range(0, particle_num) : 
        particle_x = particle_group[0][i]
        particle_y = particle_group[1][i]
        fitness = 1 / (100 * ((particle_y - particle_x ** 2) ** 2) + (1 - particle_x) ** 2 + 1)
        if fitness == gbest_fitness :
            break
    gbest = np.zeros((var_num, 1))
    gbest[0][0] = particle_x
    gbest[1][0] = particle_y
    return gbest


def particle_update(domain_low, domain_high, weight, c1_learn, c2_learn, 
                    particle_group, velocity, pbest, gbest) :   
    velocity = weight * velocity + c1_learn * np.random.rand() * (pbest - particle_group) \
               + c2_learn * np.random.rand() * (gbest - particle_group)
    velocity = velocity_check(velocity)
    particle_group = particle_group + velocity
    particle_group = particle_check_f2(particle_group, domain_low, domain_high)
    return particle_group, velocity


def best_update(particle_group, fitness, gbest, gbest_fitness, pbest, pbest_fitness) : 
    particle_num = pbest.shape[1]
    var_num = pbest.shape[0]
    for i in range(0, particle_num) : 
        if pbest_fitness[0][i] < fitness[0][i] : 
            pbest_fitness[0][i] = fitness[0][i]
            for j in range(0, var_num) : 
                pbest[j][i] = particle_group[j][i]

    if gbest_fitness < np.max(fitness) : 
        gbest_fitness = np.max(fitness)
        gbest = gbest_particle_f2(gbest_fitness, particle_group)      
    return gbest, pbest


def paticle_swarm_optimization(low, high, dimension, particle_num = 50, iteration_num = 100) :
    particle_group, velocity, weight, c1_learn, c2_learn = pso_init(low, high, dimension, 
                                                           particle_num)
    particle_group = particle_check_f2(particle_group, low, high)
    velocity = velocity_check(velocity)

    fitness = compute_fitness_f2(particle_group)
    pbest_fitness = fitness
    gbest_fitness = np.max(fitness)
    pbest = particle_group
    gbest = gbest_particle_f2(gbest_fitness, particle_group)

    for cnt in range(0, iteration_num) : 
        particle_group, velocity = particle_update(low, high, weight, c1_learn, c2_learn, 
                                   particle_group, velocity, pbest, gbest)
        particle_group = particle_check_f2(particle_group, low, high)
        velocity = velocity_check(velocity)

        fitness = compute_fitness_f2(particle_group)
        gbest, pbest = best_update(particle_group, fitness, gbest, gbest_fitness, 
                                   pbest, pbest_fitness)
    
    return particle_group, pbest, gbest

"""
test
"""
particle_group, pbest, gbest = paticle_swarm_optimization(-5, 5, 2, 50, 150)
value = compute_result_f2(particle_group)
print(value)