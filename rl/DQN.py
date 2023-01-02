import os
import gym
import pathlib
from stable_baselines3 import DQN as DQN_

from config.default import RLConfig


class DQN:
    def __init__(self, id_, config=None, pretrained_model=None):
        # Define um ID para a Instância/Simulacao
        self._id = id_

        if config is None:
            self.config = RLConfig()
        else:
            self.config = config

        # Define qual a arquitetura da rede neural interna da DQN
        self.policy = dict(net_arch=self.config.net_arch)

        # Instancia o ambiente Gym Customizado para o Problema
        self.env = gym.make('gym_phishing:RLPTraining-v0')

        # Define o total de timesteps
        self.total_timesteps = self.config.total_timesteps

        # Instancia o Modelo
        if pretrained_model is None:
            self.model = DQN_("MlpPolicy",
                              self.env,
                              policy_kwargs=self.policy,
                              learning_rate=self.config.learning_rate,
                              verbose=self.config.verbose)
        # Ou carrega um modelo treinado anteriormente
        else:
            self.model = DQN_.load(pretrained_model)

    def train(self):
        self.model.learn(total_timesteps=self.total_timesteps)
        full_path = os.path.join(pathlib.Path().resolve(), "models", "DQNmodel_{}.zip".format(self._id))
        self.model.save(full_path)

    @staticmethod
    def getPrecision(tp, fp):
        return tp/(tp + fp)

    @staticmethod
    def getRecall(tp, fn):
        return tp/(tp + fn)

    @staticmethod
    def getAccuracy(tp, tn, fp, fn):
        return (tp + tn)/(tp + tn + fp + fn)

    @staticmethod
    def getSpecificity(tn, fp):
        return tn/(tn/fp)

    def test(self):
        self.env = gym.make('gym_phishing:RLPTest-v0')
        obs = self.env.reset()
        self.model.set_env(self.env)
        tp = 0
        tn = 0
        fp = 0
        fn = 0

        for _ in range(3000):
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

        precision = DQN.getPrecision(tp, fp)
        recall = DQN.getRecall(tp, fn)
        accuracy = DQN.getAccuracy(tp, tn, fp, fn)
        specificity = DQN.getSpecificity(tn, fp)

        return precision, recall, accuracy, specificity

