import numpy as np 
import math
import matplotlib.pyplot as plt 


def population_generate(domain_min, domain_max, size = 100) :
    assert(domain_min < domain_max)
    population = []
    domain_min = int(domain_min * 10000)
    domain_max = int(domain_max * 10000)
    initial_population_x = np.random.randint(domain_min, domain_max, size)
    initial_population_x = initial_population_x.reshape(1, size)
    initial_population_y = np.random.randint(domain_min, domain_max, size)
    initial_population_y = initial_population_y.reshape(1, size)

    for i in range(size) :
        indiv_str_x = bin(initial_population_x[0, i]).replace("0b", "")
        indiv_str_y = bin(initial_population_y[0, i]).replace("0b", "")
        indiv_str = [indiv_str_x, indiv_str_y]
        population.append(indiv_str)
    return population


def population_check(population) : 
    num_indiv = len(population)
    for i in range(0, num_indiv) :
        x = (population[i])[0] 
        y = (population[i])[1]
        if x == "0" : 
            (population[i])[0] = "1"
        if y == "0" : 
            (population[i])[1] = "1"
    return population


def compute_fitness_f1(population) :
    fitness = []
    fitness_pro = []
    population = population_check(population)
    for indiv_str in population : 
        indiv_x_val = float(int(indiv_str[0], 2)) / 10000
        indiv_y_val = float(int(indiv_str[1], 2)) / 10000
        indiv_fitness = math.sin(indiv_x_val) * math.sin(indiv_y_val) \
                        / (indiv_x_val * indiv_y_val)
        fitness.append(indiv_fitness)
    
    min_fitness = min(fitness)
    fitness_pro = [i - min_fitness + 1 for i in fitness]
    return fitness_pro, fitness


def compute_fitness_f2(population) : 
    fitness = []
    fitness_pro = []
    for indiv_str in population : 
        indiv_x_val = float(int(indiv_str[0], 2)) / 10000
        indiv_y_val = float(int(indiv_str[1], 2)) / 10000
        indiv_fitness = 100 * ((indiv_y_val - indiv_x_val ** 2) ** 2) \
                        + (1 - indiv_x_val) ** 2
        fitness.append(indiv_fitness)
    
    fitness_pro = [1. / (i + 1) for i in fitness]
    return fitness_pro, fitness


def wheel_select(population) :
    pointer = np.random.rand()
    probability = 0.
    fitness_pro, _ = compute_fitness_f2(population)
    sum_fitness = sum(fitness_pro)
    num_indiv = len(fitness_pro)

    for i in range(num_indiv) :
        probability +=  fitness_pro[i] / sum_fitness
        if pointer <= probability :
            indiv_select =  population[i]
            break
    return indiv_select


def ga_select(population) : 
    num_indiv = len(population)
    new_population = []
    for i in range(num_indiv) :
        indiv_select = wheel_select(population)
        new_population.append(indiv_select)
    return new_population


def str_to_vector(bin_str, bit_len) :
    vector = np.zeros((1, bit_len))
    if bin_str[0] == "-" :
        vector[0][0] = 1
        for i in range(1, len(bin_str)) : 
            vector[0][bit_len - i] = int(bin_str[len(bin_str) - i])
    else :
        for i in range(1, len(bin_str) + 1) :
            vector[0][bit_len - i] = int(bin_str[len(bin_str) - i])
    return vector


def vector_to_str(vector) :
    num_element = vector.shape[1]
    flag = 0
    if vector[0][0] == 1 :
        bin_str = "-"
        for i in range(1, num_element) :
            if flag != 0 :
                bin_str += str(int(vector[0][i]))
                continue
            if vector[0][i] != 0 :  
                flag = 1
                bin_str += str(int(vector[0][i]))
    else : 
        bin_str = ""
        for i in range(1, num_element) :
            if flag != 0 :
                bin_str += str(int(vector[0][i]))
                continue
            if vector[0][i] != 0 :  
                flag = 1
                bin_str += str(int(vector[0][i]))
    
    if bin_str == "-" or bin_str == "" :
        bin_str = "0"
    return bin_str


def indiv_cross(parent_1, parent_2, bit_len = 17) :
    parent_1_x = parent_1[0]
    parent_2_x = parent_2[0]
    parent_1_y = parent_1[1]
    parent_2_y = parent_2[1]

    parent_1_x_vec = str_to_vector(parent_1_x, bit_len)
    parent_2_x_vec = str_to_vector(parent_2_x, bit_len)
    parent_1_y_vec = str_to_vector(parent_1_y, bit_len)
    parent_2_y_vec = str_to_vector(parent_2_y, bit_len)

    cross_position_x = (np.random.randint(0, bit_len - 1, 1))[0]
    cross_position_y = (np.random.randint(0, bit_len - 1, 1))[0]

    indiv_1_x_vec = np.hstack((parent_1_x_vec[:, 0 : cross_position_x],
                               parent_2_x_vec[:, cross_position_x : bit_len]))
    indiv_2_x_vec = np.hstack((parent_2_x_vec[:, 0 : cross_position_x], 
                               parent_1_x_vec[:, cross_position_x : bit_len]))
    indiv_1_y_vec = np.hstack((parent_1_y_vec[:, 0 : cross_position_y],
                               parent_2_y_vec[:, cross_position_y : bit_len]))                           
    indiv_2_y_vec = np.hstack((parent_2_y_vec[:, 0 : cross_position_y],
                               parent_1_y_vec[:, cross_position_y : bit_len]))

    indiv_1_x = vector_to_str(indiv_1_x_vec)
    indiv_2_x = vector_to_str(indiv_2_x_vec)
    indiv_1_y = vector_to_str(indiv_1_y_vec)
    indiv_2_y = vector_to_str(indiv_2_y_vec)

    indiv_1 = [indiv_1_x, indiv_1_y]
    indiv_2 = [indiv_2_x, indiv_2_y]
    return indiv_1, indiv_2


def ga_cross(population, P_cross = 0.75) : 
    num_indiv = len(population)
    new_population = []
    i = 0
    for cnt in range(0, int(num_indiv / 2)) : 
        pointer = np.random.rand()
        if pointer <= P_cross : 
            indiv_1, indiv_2 = indiv_cross(population[i], population[i + 1])
        else :
            indiv_1 = population[i]
            indiv_2 = population[i + 1]
        i += 2
        new_population.append(indiv_1)
        new_population.append(indiv_2)
    return new_population


def indiv_mutate(indiv, bit_len = 17) : 
    indiv_x = indiv[0]
    indiv_y = indiv[1]
    pointer = np.random.rand()
    mutation_rule = {0 : 1, 1 : 0}
    indiv_x_vec = str_to_vector(indiv_x, bit_len)
    indiv_y_vec = str_to_vector(indiv_y, bit_len)
    mutate_pos = (np.random.randint(0, bit_len - 1, 1))[0]

    if pointer <= 0.5 :
        indiv_x_vec[0][mutate_pos] = mutation_rule[indiv_x_vec[0][mutate_pos]]
    else :
        indiv_y_vec[0][mutate_pos] = mutation_rule[indiv_y_vec[0][mutate_pos]]
    
    indiv_x = vector_to_str(indiv_x_vec)
    indiv_y = vector_to_str(indiv_y_vec)
    indiv_new = [indiv_x, indiv_y]
    return indiv_new


def ga_mutate(population, P_mutate = 0.001) : 
    num_indiv = len(population)
    for i in range(0, num_indiv) : 
        pointer = np.random.rand()
        if pointer <= P_mutate : 
            indiv = population[i]
            population[i] = indiv_mutate(indiv)
    new_population = population
    return new_population


def genetic_algorithm(domain_min, domain_max, num_iteration = 150) : 
    population = population_generate(domain_min, domain_max)
    fitness_average = []
    for i in range(num_iteration) : 
        population = ga_select(population)
        population = ga_cross(population)
        # population = ga_mutate(population)
        _, fitness = compute_fitness_f2(population)
        fitness_average.append(sum(fitness) / len(fitness))
    return population, fitness_average


"""
test
"""
population, fitness_aver = genetic_algorithm(-5, 5, 120)

plt.rcParams["figure.figsize"] = (5.0, 4.0)              # 以英寸为单位的图片宽高设定
plt.rcParams["image.interpolation"] = "nearest"
plt.rcParams["image.cmap"] = "gray"
plt.ylabel("fitness_aver")
plt.xlabel("iterations")
plt.plot(np.squeeze(fitness_aver))
plt.show()