import os
import gym
import random
from gym import spaces
from gym.spaces import MultiDiscrete
import pandas as pd


class RLPTraining(gym.Env):

    def __init__(self):
        super(RLPTraining, self).__init__()
        self.current_step = 0
        self.max_errors = 10
        self.errors = 0
        self.X, self.Y = RLPTraining._load_dataset()
        self.action_space = spaces.Discrete(3)
        self.observation_space = MultiDiscrete([3, 3, 3, 3, 3, 3, 3, 3, 3])
        self.reward_range = (-1000000, 1000000)

    @staticmethod
    def _load_dataset():
        x = pd.read_csv(os.path.abspath("./gym-phishing/gym_phishing/data/trainX.csv"))
        y = pd.read_csv(os.path.abspath("./gym-phishing/gym_phishing/data/trainY.csv"))
        return x, y

    def reset(self):
        self.current_step = random.randint(0, len(self.X.loc[:, 'SFH'].values) - 1)
        return self._next_observation()

    def _next_observation(self):
        a = self.X.iloc[[self.current_step]].to_numpy()[0]
        a += 1
        return a

    def step(self, action):
        reward = 0.0
        acertou = False
        if action == 0:
            if self.Y['Result'][self.current_step] == -1:
                reward += 2000
                acertou = True
            else:
                reward -= 500
                self.errors = self.errors + 1
        elif action == 1:
            if self.Y['Result'][self.current_step] == 0:
                reward += 2000
                acertou = True
            else:
                reward -= 500
                self.errors = self.errors + 1
        else:
            if self.Y['Result'][self.current_step] == 1:
                reward += 2000
                acertou = True
            else:
                reward -= 500
                self.errors = self.errors + 1

        obs = self.reset()

        info = {'acertou': acertou, 'total de erros': self.errors}
        done = False
        if self.errors >= self.max_errors:
            done = True
            self.errors = 0

        return obs, reward, done, info

    def render(self, mode="human"):
        pass
