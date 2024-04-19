import torch
import math
import random
import numpy as np
def roulette_selection(population, fitness):
    """
    轮盘赌算法选择函数
    :param population: 种群
    :param fitness: 种群中每个个体的适应度函数值，列表形式
    :return: 选中的个体
    """

    # 计算适应度总和
    total_fitness = sum(fitness)

    # 计算每个个体被选中的概率
    probabilities = [f / total_fitness for f in fitness]
    # 计算每个部分的累积概率
    cumulative_probabilities = []
    probability_sum = 0

    for p in probabilities:
        probability_sum += p
        cumulative_probabilities.append(probability_sum)
    # 选择一个随机数

    random_number = random.uniform(0, 1)
    # 判断落在哪个部分，对应的个体被选中
    for i in range(len(cumulative_probabilities)):
        if random_number <= cumulative_probabilities[i]:
            return population[i]


def roulette_selection1(population, fitness, memory_arr,fc):
    """
    轮盘赌算法选择函数
    :param population: 种群
    :param fitness: 种群中每个个体的适应度函数值，列表形式
    :return: 选中的个体
    """
    # d = 0.2
    total_fitness = 0
    index = 0
    copy_fitness=[]
    # print(memory_arr)
    # print(fitness)
    for memory,fit in zip(memory_arr,fitness):
        total_fitness = total_fitness + fit
        copy_fitness.append(fit)
        index = index + 1

    probabilities = [f / total_fitness for f in fitness]
    # 计算每个部分的累积概率
    cumulative_probabilities = []
    probability_sum = 0

    for p in probabilities:
        probability_sum += p
        cumulative_probabilities.append(probability_sum)
    # 选择一个随机数
    random_number = random.uniform(0, 1)
    # print(cumulative_probabilities)
    # 判断落在哪个部分，对应的个体被选中
    for i in range(len(cumulative_probabilities)):
        if random_number <= cumulative_probabilities[i]:
            # print(population[i])
            return population[i]