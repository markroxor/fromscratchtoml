#!/usr/bin/env python
# -*- coding: utf-8 -*-
#



import sys
import random
import time
from copy import deepcopy
from collections import deque

import gym

import numpy as np
import matplotlib.pyplot as plt

from fromscratchtoml.neural_network.layers import Dense, Activation
from fromscratchtoml.neural_network.optimizers import Adam, RMSprop
from fromscratchtoml.neural_network.models import Sequential
from fromscratchtoml.toolbox import HiddenPrints
np.random.seed(0)
random.seed(0)

class Experiences(object):
    def __init__(self, experience_size):
        self.experiences = deque([] * experience_size, experience_size)

    def remember(self, state, action ,next_state, reward):
        self.experiences.append((state, action ,next_state, reward))

    def sample(self, n):
        n = min(len(self.experiences), n)
        return random.sample(list(self.experiences), n)


class Network(object):
    def __init__(self, num_states, num_actions):
        self.model = Sequential()
        self.model.add(Dense(input_dim=num_states, units=32, seed=1))
        self.model.add(Activation("relu"))
        self.model.add(Dense(units=num_actions, seed=2))
        self.model.add(Activation("linear"))

        opt = RMSprop(learning_rate=0.005)
        self.model.compile(optimizer=opt, loss="mean_squared_error")        

    def learn(self, states, actions):
        with HiddenPrints():
            self.model.fit(states, actions, epochs=1, batch_size=64)
        
    def predict(self, states, prob=False):
        return self.model.predict(states, prob=prob)


class Agent(object):
    def __init__(self, env, num_experiences=10000, min_explore_prob=0.01,
                max_explore_prob=1, discount=0.99, explore_prob_decay=0.001):
        self.explore_prob = max_explore_prob
        self.max_explore_prob = max_explore_prob
        self.min_explore_prob = min_explore_prob
        self.discount = discount
        self.explore_prob_decay = explore_prob_decay

        state = env.reset()

        self.num_states, self.num_actions = env.observation_space.shape[0], env.action_space.n

        self.rewards = []

        self.learner_network = Network(self.num_states, self.num_actions)
        self.learner_network.model.forwardpass(np.expand_dims(state, axis=0))

        self.actual_network = deepcopy(self.learner_network)

        self.experiences = Experiences(num_experiences)

    def act(self, env, state, time_step):
        if self.explore_prob > np.random.uniform(0, 1):
            action  = env.action_space.sample()
        else:
            action = self.learner_network.predict(np.expand_dims(state, axis=0))[0]

        self.explore_prob = self.min_explore_prob + (self.max_explore_prob - self.min_explore_prob) * \
                            np.exp(-self.explore_prob_decay * time_step)

        next_state, reward, done, _ = env.step(action)

        if done:
            next_state = None

        return action, next_state, reward

    def learn_from_experiences(self, experience_batch_size):
        experience_batch = self.experiences.sample(experience_batch_size)

        X = np.zeros((experience_batch_size, self.num_states))
        y = np.zeros((experience_batch_size, self.num_actions))

        for i, experience in enumerate(experience_batch):
            _state, _action ,_next_state, _reward = experience

            X[i] = _state
            y[i] = self.learner_network.predict(np.expand_dims(_state, axis=0), prob=True)
            y[i][_action] = _reward

            if _next_state is not None:
                y[i][_action] += self.discount * \
                self.actual_network.predict(np.expand_dims(_next_state, axis=0), prob=True)[0]\
                [np.argmax(self.learner_network.predict(np.expand_dims(_next_state, axis=0), prob=True)[0])]

        self.learner_network.learn(X, y)

    def actual_is_learner(self):
        self.actual_network = deepcopy(self.learner_network)


class AdDQN(object):
    def __init__(self, env_name="CartPole-v0", num_experiences=100000, seed=None):
        self.env = gym.make(env_name)
        if seed is not None:
            self.env.seed(seed)
            
        self.agent = Agent(env=self.env, num_experiences=num_experiences)
        self.rewards = []
        self.time_step = 0
        self.num_experiences = num_experiences

    def learn(self, num_episodes=sys.maxsize, experience_batch_size=64):
        try:
            # Fill the experiences before hand
            experiences = Experiences(experience_batch_size)

            while len(self.agent.experiences.experiences) > self.num_experiences:
                state =  self.env.reset()

                while True:
                    action  = self.env.action_space.sample()
                    next_state, reward, done, _ = self.env.step(action)

                    experiences.remember(state, action, next_state, reward)

                    state = next_state
                    if next_state is None:
                        break

            self.agent.experiences = experiences

            for episode in range(num_episodes):
                state =  self.env.reset()
                total_reward = 0

                while True:
                    self.time_step += 1
                    action, next_state, reward = self.agent.act(self.env, state, self.time_step)
                    total_reward += reward

                    self.agent.experiences.remember(state, action, next_state, reward)

                    if self.time_step % 1000 == 0:
                        self.agent.actual_is_learner()

                    self.agent.learn_from_experiences(experience_batch_size)

                    state = next_state
                    if next_state is None:
                        break
        
                if episode % 100 == 0:
                    print("-" * 50 + "Episode - ", episode, "-" * 50)
                print("Reward is ", total_reward)

                self.rewards.append(total_reward)

        except KeyboardInterrupt:
            print("Learning was stopped by Interrupt!")

        finally:
            plt.plot(self.rewards)


    def play(self, num_episodes=sys.maxsize, lapse=0.1):
        for episode in range(num_episodes):
            total_reward = 0
            state = self.env.reset()

            while True:
                time.sleep(lapse)
                self.env.render()

                action, next_state, reward = self.agent.act(self.env, state, sys.maxsize)
                total_reward += reward

                state = next_state
                
                if next_state is None:
                    break

            print("Reward is ", total_reward)
            total_reward = 0
