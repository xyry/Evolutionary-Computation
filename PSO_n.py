#!/usr/bin/env python
# _*_coding:utf-8 _*_
# @Time    : 2019/11/18 17:16
# @Author  : YPL
# @FileName: PSO_1.py
# @Software: PyCharm
# -*- coding: utf-8 -*-
# import sys

# reload(sys)
# sys.setdefaultencoding('utf-8')

import numpy as np
import matplotlib.pyplot as plt
import time

# 目标函数定义
def func(x, n_value):
    n = n_value
    value=1

    for i in range(n):
        sum=0
        for j in range(1,6):
            sum+=j*np.cos((j+1)*x[i]+j)
        value*=sum
    return value

def solve(n_v):

    # 参数初始化
    w = 1.0
    c1 = 1.6
    c2 = 1.7

    maxgen = 200  # 进化次数
    sizepop = 40  # 种群规模

    # 粒子速度和位置的范围
    Vmax = 1
    Vmin = -1

    popmax = 10
    popmin = -10

    n_value=n_v
    # 产生初始粒子和速度
    pop = 5 * np.random.uniform(-1, 1, (n_value, sizepop))

    v = np.random.uniform(-1, 1, (n_value, sizepop))


    fitness = func(pop, n_value)  # 计算适应度
    i = np.argmin(fitness)  # 找最好的个体
    gbest = pop  # 记录个体最优位置
    zbest = pop[:, i]  # 记录群体最优位置
    fitnessgbest = fitness  # 个体最佳适应度值
    fitnesszbest = fitness[i]  # 全局最佳适应度值

    # 迭代寻优
    t = 0
    record = np.zeros(maxgen)
    while t < maxgen:

        # 速度更新
        v = w * v + c1 * np.random.random() * (gbest - pop) + c2 * np.random.random() * (zbest.reshape(n_value, 1) - pop)
        v[v > Vmax] = Vmax  # 限制速度
        v[v < Vmin] = Vmin

        # 位置更新
        pop = pop + 0.5 * v
        pop[pop > popmax] = popmax  # 限制位置
        pop[pop < popmin] = popmin



        # 计算适应度值
        fitness = func(pop, n_value)

        # 个体最优位置更新
        index = fitness < fitnessgbest
        fitnessgbest[index] = fitness[index]
        gbest[:, index] = pop[:, index]

        # 群体最优更新
        j = np.argmin(fitness)
        if fitness[j] < fitnesszbest:
            zbest = pop[:, j]
            fitnesszbest = fitness[j]

        record[t] = fitnesszbest  # 记录群体最优位置的变化

        t = t + 1
    return zbest,record
# 结果分析
if __name__=='__main__':
    start = time.time()
    n_value=[1,2,3,4]

    for n in range(1,5):
        zbest,record=solve(n)
        print('当n=' + str(n) +'时：最优值' + str(func(zbest, n)) + ',' + 'x=' + str(zbest))
        plt.plot(record, 'b-')
        plt.xlabel('generation')
        plt.ylabel('fitness')
        plt.title('fitness curve')
        plt.show()


    end=time.time()
    print('耗时'+str(end-start))