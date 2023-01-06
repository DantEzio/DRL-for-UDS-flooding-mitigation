# -*- coding: utf-8 -*-
"""
Created on Thu May  5 07:57:43 2022

@author: chong
"""

import numpy as np
import matplotlib.pyplot as plt


class PSO(object):
    def __init__(self, params,fitness_function):
        '''
        初始化最优化算法
        
        w：  # 惯性权重
        c1,c2： pso算法参数
        population_size: # 粒子群数量
        dim: # 搜索空间的维度
        max_steps: # 迭代次数
        x_bound: # 解空间范围
        '''
        self.params = params
        self.x = np.random.uniform(self.params['x_bound'][0], self.params['x_bound'][1],
                                  (self.params['population_size'], self.params['dim']))  # 初始化粒子群位置
        self.v = np.random.rand(self.params['population_size'], self.params['dim'])  # 初始化粒子群速度
        #优化的适应性函数
        self.calculate_fitness=fitness_function
        
        #初始化计算步骤
        fitness = self.calculate_fitness(self.x)
        self.p = self.x  # 个体的最佳位置
        self.pg = self.x[np.argmax(fitness,axis=0)]  # 全局最佳位置
        self.individual_best_fitness = fitness  # 个体的最优适应度
        self.global_best_fitness = np.max(fitness)  # 全局最佳适应度
        
    def set_x(self,x):
        self.x=x
        
    def evolve(self):
        self.best_fit_curv=[]
        for step in range(self.params['max_steps']):
            r1 = np.random.rand(self.params['population_size'], self.params['dim'])
            r2 = np.random.rand(self.params['population_size'], self.params['dim'])
            # 更新速度和权重
            self.v =  self.params['w'] * self.v + self.params['c1'] * r1 * (self.p - self.x) + self.params['c2'] * r2 * (self.pg - self.x)
            self.x = np.around(self.params['a']*self.v + self.x)
            #计算适应度
            fitness = self.calculate_fitness(self.x)
            # 需要更新的个体
            update_id = np.greater(self.individual_best_fitness, fitness)
            self.p[update_id] = self.x[update_id]
            self.individual_best_fitness[update_id] = fitness[update_id]
            # 新一代出现了更大的fitness，所以更新全局最优fitness和位置
            if np.max(fitness) > self.global_best_fitness:
                self.pg = self.x[np.argmax(fitness,axis=0)]    
                self.global_best_fitness = np.max(fitness)
            self.best_fit_curv.append(self.global_best_fitness)


if __name__=='__main__':
    
    def fit_function(x):
        fitvalue=[]
        for item in x:
            fitvalue.append(-np.sum(np.power(item,2)))
        return np.array(fitvalue)
    
    params={
        'w':0.9,
        'c1':0.9,
        'c2':0.9,
        'a':0.01,
        'population_size':10,
        'dim':12,
        'max_steps':100,
        'x_bound':[0,1],
        }
    
    pso = PSO(params,fit_function)
    pso.evolve()
    print(pso.pg)
    plt.plot(pso.best_fit_curv)
