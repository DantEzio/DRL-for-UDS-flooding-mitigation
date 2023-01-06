# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 20:10:48 2022

@author: chong
"""
import tensorflow as tf
from tensorflow.keras import layers,models
from tensorflow.keras.optimizers import Adam

from collections import deque
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random


class DQN:
    
    def __init__(self,params,env):
        tf.compat.v1.disable_eager_execution()
        self.params=params
        self.memory_buffer = deque(maxlen=1000)
        self.env=env
        self.action_table=pd.read_csv('./DQN_action_table.csv').values[:,1:]
        print('table shape: ',self.action_table.shape)
        
        self.model=self._build_net()
        self.target_model=self._build_net()
        
        self.max_state=np.array([2.01026872e+00, 4.22421662e+00, 3.25000000e+00, 3.35000000e+00,
               2.00000000e+00, 2.21000000e+00, 1.96000000e+00, 2.05000000e+00,
               1.28939237e+03, 1.38030260e+03, 8.02888818e+02, 2.15337375e+03,
               6.83000000e+02, 6.83000000e+02, 1.00000000e+03, 1.00000000e+03,
               1.57454046e+03, 2.79013935e+02])
        self.min_state=np.array([ 4.98142986e-01,  9.87964095e-01,  1.37786022e-01,  0.00000000e+00,
                4.00174015e-02,  0.00000000e+00,  2.13572798e-04,  4.85072906e-03,
                1.08492137e+01,  0.00000000e+00, -5.03835243e+02,  9.27032212e-02,
                0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                1.11917992e+00,  0.00000000e+00])
        
    def _build_net(self):
        #DQN
        #eval net
        self.s = layers.Input(shape=self.params['state_dim'],name='s_input')
        V_prev = self.s
        for i in np.arange(self.params['evalnet_layer_V']):
            V_prev=layers.Dense(self.params['evalnet_V'][i]['num'], activation='relu', name='evalnet_V'+str(i))(V_prev)
        self.eval_out=layers.Dense(self.params['action_dim'], activation='linear', name='evalnet_out_V')(V_prev)
        
        model=models.Model(inputs=[self.s],outputs=self.eval_out)
        return model
    
    def choose_action(self,state,train_log):
        #input state, output action
        # state归一化
        state=(np.array([state])-self.min_state)/(self.max_state-self.min_state)
        
        if train_log:
            #epsilon greedy
            pa = np.random.uniform()
            if pa < self.params['epsilon']:
                action_value = self.model.predict(state)
                action = np.argmax(action_value)
            else:
                action = np.random.randint(self.params['action_dim'])
        else:
            action_value = self.model.predict(state)
            action = np.argmax(action_value)
        return action

    
    def remember(self, state, action, reward, next_state, done):
        item = (state, action, reward, next_state, done)
        self.memory_buffer.append(item)

    def process_batch(self, batch):
         # 从经验池中随机采样一个batch
        #data = random.sample(self.memory_buffer, batch)
        data=[]
        for i in range(batch):
            data.append(self.memory_buffer[-i])
        # 生成Q_target
        states = np.array([d[0] for d in data])
        next_states = np.array([d[3] for d in data])
        y = self.model.predict(states)
        q = self.target_model.predict(next_states)

        for i, (_, action, reward, _, done) in enumerate(data):
            target = reward
            if not done:
                target += self.params['gamma'] * np.amax(q[i])
            y[i][action] = target
        return states, y
    
    
    def train(self,RainData):
        #sampling and upgrading
        history = {'episode': [], 'Batch_reward': [], 'Episode_reward': [], 'Loss': []}
        self.model.compile(loss='mse', optimizer=Adam(1e-3))
        for j in range(self.params['training_step']):
            count = 0
            for i in range(self.params['num_rain']):
                reward_sum = 0
                print('training step:',j,' sampling num:',i)
                #Sampling: each rainfall represent one round of sampling
                batch=0
                s = self.env.reset(RainData[i])
                done, batch = False, 0
                while not done:
                    a = self.choose_action(s,True)
                    action = self.action_table[a,:].tolist()
                    snext,reward,_,_,done = self.env.step(action)
                    self.remember(s, a, reward, snext, done)
                    s = snext
                    batch+=1
                
                    reward_sum += reward
                #Upgrading: each rainfall for one round of upgrading
                X, y = self.process_batch(batch)
                loss = self.model.train_on_batch(X, y)
                count += 1
                # 减小egreedy的epsilon参数。
                if self.params['epsilon'] >= self.params['ep_min']:
                    self.params['epsilon'] *= self.params['ep_decay']
                # 更新target_model
                if count != 0:
                    self.target_model.set_weights(self.model.get_weights())
                if i % 5 == 0:
                    history['episode'].append(i)
                    history['Episode_reward'].append(reward_sum)
                    history['Loss'].append(loss)
                    print('Episode: {} | Episode reward: {} | loss: {:.3f} | e:{:.2f}'.format(i, reward_sum, loss, self.params['epsilon']))
                    
                self.memory_buffer = deque(maxlen=1000)
                    
            self.model.save_weights('./model/dqn.h5')
        return history
    
    def load_model(self):
        self.model.load_weights('./model/dqn.h5')
    
    def test(self,rain):
        # simulation on given rainfall
        test_history = {'time':[] ,'state': [], 'action': [], 'reward': [], 'F':[], 'C':[]}
        s = self.env.reset(rain)
        done, t= False, 0
        test_history['time'].append(t)
        test_history['state'].append(s)
        while not done:
            a = self.choose_action(s,False)
            action = self.action_table[a,:].tolist()
            snext,reward,F,C,done = self.env.step(action)
            s = snext
            t +=1
            
            test_history['time'].append(t)
            test_history['state'].append(s)
            test_history['action'].append(action)
            test_history['reward'].append(reward)
            test_history['F'].append(F)
            test_history['C'].append(C)
    
        return test_history
    
    
    
    
        