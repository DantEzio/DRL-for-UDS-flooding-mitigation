# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 20:39:01 2022

@author: chong
"""
import numpy as np
import SWMM_ENV_HC
#import Rainfall_data as RD
import datetime
#import matplotlib.pyplot as plt

#prepare rainfall
raindata = np.load('test_raindata.npy').tolist()

env_params={
        'orf':'chaohu',
        'advance_seconds':300
    }
env=SWMM_ENV_HC.SWMM_ENV(env_params)

for i in range(len(raindata)):
    # simulation on given rainfall
    test_history = {'time':[] ,'state': [], 'action': [], 'reward': [], 'F':[], 'C':[]}
    s = env.reset(raindata[i])
    done, t= False, 0
    test_history['time'].append(t)
    test_history['state'].append(s)
    while not done:
        snext,reward,F,C,done = env.step()
        s = snext
        t +=1
            
        test_history['time'].append(t)
        test_history['state'].append(s)
        test_history['reward'].append(reward)
        test_history['F'].append(F)
        test_history['C'].append(C)
    
    np.save('./Results/HC'+str(i)+'.npy',test_history)
