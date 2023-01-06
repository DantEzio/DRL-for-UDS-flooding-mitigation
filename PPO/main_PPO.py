# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 20:39:01 2022

@author: chong
"""
import numpy as np
import SWMM_ENV
import PPO as PPO
import Rainfall_data as RD

import datetime
import matplotlib.pyplot as plt

#prepare rainfall
raindata = np.load('training_raindata.npy').tolist()

env_params={
        'orf':'chaohu',
        'advance_seconds':300
    }
env=SWMM_ENV.SWMM_ENV(env_params)

agent_params={
    'state_dim':len(env.config['states']),
    'action_dim':len(env.config['action_assets']),
    'actornet_layer_A':3,
    'actornet_A':[{'num':30},{'num':30},{'num':30}],
    
    'actornet_A_mu':len(env.config['action_assets']),
    'actornet_A_sigma':len(env.config['action_assets']),
    
    'bound_low':0,
    'bound_high':1,
    
    'evalnet_layer_V':3,
    'evalnet_V':[{'num':30},{'num':30},{'num':30}],
    
    'clip_ratio':0.01,
    'target_kl':0.03,
    'lam':0.01,
    
    'policy_learning_rate':0.001,
    'value_learning_rate':0.001,
    
    'num_rain':300,
    
    'training_step':20,
    'gamma':0.3,
    'epsilon':0.5,
    'ep_min':0.01,
    'ep_decay':0.1
}
agent = PPO.PPO(agent_params,env)
#agent.load_model()
history = agent.train(raindata)
np.save('./Results/Train'+str(4)+'.npy',history)

raindata = np.load('test_raindata.npy').tolist()
agent.load_model()

for i in range(len(raindata)):
    test_his = agent.test(raindata[i])
    np.save('./Results/'+str(i)+'.npy',test_his)
