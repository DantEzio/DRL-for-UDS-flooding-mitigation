# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 20:39:01 2022

@author: chong
"""
import numpy as np
import SWMM_ENV
import pso
import pandas as pd
#import Rainfall_data as RD

import matplotlib.pyplot as plt


env_params={
        'orf':'chaohu',
        'advance_seconds':300
    }
env=SWMM_ENV.SWMM_ENV(env_params)
action_table=pd.read_csv('./DQN_action_table.csv').values[:,1:]

params={
    'w':0.9,
    'c1':0.9,
    'c2':0.9,
    'a':0.01,
    'population_size':10,
    'dim':95,
    'max_steps':100,
    'x_bound':[0,127],
    }


def get_y2(data):
    #从数据中泵的策略给出分类的y
    y=np.zeros((data.shape[0],1))
    for line in range(y.shape[0]):
        y[line]=np.where((action_table==data[line,:]).all(axis=1))
    return y

env=SWMM_ENV.SWMM_ENV(env_params)


raindata = np.load('test_raindata.npy').tolist()
for i in range(len(raindata)):
    
    def fitness_function_single(action):
        # simulation on given rainfall
        env.reset(raindata[i])
        done, t, Return= False, 0, 0
        while not done:
            a = int(action[t])
            act = action_table[a,:].tolist()
            _,reward,_,_,done = env.step(act)
            t +=1
            Return += reward
            
        return Return

    def fitness(actions):
        #一组多个action计算多个return
        r=[]
        for action in actions:
            r.append(fitness_function_single(action))
        return np.array(r)    
    
    
    #以PPO的action作为初值
    actions=np.random.randint(128, size=(1,95))
    initial_a=np.array(np.load('./Initial/'+str(i)+'.npy',allow_pickle=True).tolist()['action'])
    initial_action=get_y2(initial_a).T
    for j in range(params['population_size']-1):
        actions=np.concatenate((actions,initial_action),axis=0)
    
    opt = pso.PSO(params,fitness)
    opt.set_x(actions)
    opt.evolve()
    plt.figure()
    plt.plot(opt.best_fit_curv)
    
    #使用优化后的pg控制，得到优化后的效果
    test_history = {'time':[] ,'state': [], 'action': [], 'reward': [], 'F':[], 'C':[]}
    s = env.reset(raindata[i])
    done, t, Return= False, 0, 0
    test_history['time'].append(t)
    test_history['state'].append(s)
    while not done:
        a = int(opt.pg[t])
        act = action_table[a,:].tolist()
        snext,reward,F,C,done = env.step(act)
        s = snext
        t +=1
        
        test_history['time'].append(t)
        test_history['state'].append(s)
        test_history['action'].append(act)
        test_history['reward'].append(reward)
        test_history['F'].append(F)
        test_history['C'].append(C)
    
    np.save('./Results/'+str(i)+'.npy',test_history)

    
    
