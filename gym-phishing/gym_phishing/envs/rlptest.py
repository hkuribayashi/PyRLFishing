import os
import gym
import random
from gym import spaces
from gym.spaces import MultiDiscrete
import pandas as pd


class RLPTest(gym.Env):

    def __init__(self):
        super(RLPTest, self).__init__()
        self.current_step = 0
        self.max_errors = 10
        self.errors = 0
        self.X, self.Y = RLPTest._load_dataset()
        self.action_space = spaces.Discrete(3)
        self.observation_space = MultiDiscrete([3, 3, 3, 3, 3, 3, 3, 3, 3])
        self.reward_range = (-1000000, 1000000)

    @staticmethod
    def _load_dataset():
        x = pd.read_csv(os.path.abspath("./gym-phishing/gym_phishing/data/testX.csv"))
        y = pd.read_csv(os.path.abspath("./gym-phishing/gym_phishing/data/testY.csv"))
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

        # Quando a ação for '0', significa -1 (Talvez)
        if action == 0:
            if self.Y['Result'][self.current_step] == -1:
                reward += 2000
                resultado = "TP"
            else:
                reward -= 500
                self.errors = self.errors + 1
                resultado = "FP"
        # Quando a ação for '1', significa 0 (Não Phishing)
        elif action == 1:
            if self.Y['Result'][self.current_step] == 0:
                reward += 2000
                resultado = "TN"
            else:
                reward -= 500
                self.errors = self.errors + 1
                resultado = "FN"
        # Quando a ação for '2', significa 1 (Phishing)
        else:
            if self.Y['Result'][self.current_step] == 1:
                reward += 2000
                resultado = "TP"
            else:
                reward -= 500
                self.errors = self.errors + 1
                resultado = "FP"

        obs = self.reset()

        info = {'total de erros': self.errors, 'resultado': resultado}
        done = False
        if self.errors >= self.max_errors:
            done = True
            self.errors = 0

        return obs, reward, done, info

    def render(self, mode="human"):
        pass
