import gym
from abc import ABC

from config.default import RLConfig
from utils.metrics import getPrecision, getRecall, getAccuracy, getFalsePositiveRate, getFalseNegativeRate, getF1Score


class BaseModel(ABC):
    def __init__(self, id_, config=None):
        # Define um ID para a Instância/Simulacao
        self.model = None
        self._id = id_

        if config is None:
            self.config = RLConfig()
        else:
            self.config = config

        # Define qual a arquitetura da rede neural interna da DQN
        self.policy = dict(net_arch=self.config.net_arch['DQN'])

        # Instancia o ambiente Gym Customizado para o Problema
        self.env = gym.make('gym_phishing:RLPTraining-v0')

        # Define o total de timesteps
        self.total_timesteps = self.config.total_timesteps

    def train(self):
        self.model.learn(total_timesteps=self.total_timesteps)

    def test(self):
        self.env = gym.make('gym_phishing:RLPTest-v0')
        obs = self.env.reset()
        self.model.set_env(self.env)
        tp = 0
        tn = 0
        fp = 0
        fn = 0

        for _ in range(3649):
            action, _states = self.model.predict(obs, deterministic=True)
            obs, reward, done, info = self.env.step(action)
            if info['resultado'] == 'TP':
                tp += 1
            elif info['resultado'] == 'TN':
                tn += 1
            elif info['resultado'] == 'FP':
                fp += 1
            elif info['resultado'] == 'FN':
                fn += 1

            self.env.render()
            if done:
                obs = self.env.reset()

        precision = getPrecision(tp, fp)
        recall = getRecall(tp, fn)
        accuracy = getAccuracy(tp, tn, fp, fn)
        fpr = getFalsePositiveRate(fp, tn)
        fnr = getFalseNegativeRate(fn, tp)
        f1score = getF1Score(tp, fp, fn)
        result = {'precision': precision, 'recall': recall, 'accuracy': accuracy,
                  'fpr': fpr, 'fnr': fnr, 'f1score': f1score}

        return result
