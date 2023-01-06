# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 20:39:01 2022

@author: chong
"""
import numpy as np
import SWMM_ENV
import DQN
#import Rainfall_data as RD

import datetime
#import matplotlib.pyplot as plt

#prepare rainfall
raindata = np.load('training_raindata.npy').tolist()


env_params={
        'orf':'chaohu',
        'advance_seconds':300
    }
env=SWMM_ENV.SWMM_ENV(env_params)

agent_params={
    'state_dim':len(env.config['states']),
    'action_dim':2**len(env.config['action_assets']),
    'evalnet_layer_A':3,
    'evalnet_A':[{'num':30},{'num':30},{'num':30}],
    'evalnet_layer_V':3,
    'evalnet_V':[{'num':30},{'num':30},{'num':30}],
    'targetnet_layer_A':3,
    'targetnet_A':[{'num':30},{'num':30},{'num':30}],
    'targetnet_layer_V':3,
    'targetnet_V':[{'num':30},{'num':30},{'num':30}],
    'num_rain':300,
    
    'training_step':20,
    'gamma':0.3,
    'epsilon':0.1,
    'ep_min':0.01,
    'ep_decay':0.1
}
agent = DQN.DQN(agent_params,env)
history = agent.train(raindata[0:])
np.save('./Results/Train1.npy',history)

raindata = np.load('test_raindata.npy').tolist()
agent.load_model()
for i in range(len(raindata)):
    test_his = agent.test(raindata[i])
    np.save('./Results/'+str(i)+'.npy',test_his)
