import os
import gym
import pathlib
from stable_baselines3 import PPO as PPO_
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike


class PPO:
    def __init__(self, id_, config=None, pretrained_model=None):
        # Define um ID para a Instância/Simulacao
        self._id = id_

        # Define qual a arquitetura da rede neural interna da DQN
        # policy = dict(net_arch=config.net_arch)

        # Instancia o ambiente Gym Customizado para o Problema
        self.env = gym.make('gym_phishing:RLPTraining-v0')

        # Define o total de timesteps
        # self.total_timesteps = config.total_timesteps

        # Instancia o Modelo
        if pretrained_model is None:
            if config is not None:
                self.model = PPO_("MlpPolicy", self.env, learning_rate=config.learning_rate, verbose=config.verbose)
            else:
                self.model = PPO_("MlpPolicy", self.env, verbose=0)
        # Ou carrega um modelo treinado anteriormente
        else:
            self.model = PPO_.load(pretrained_model)

    def train(self):
        self.model.learn(total_timesteps=100000)
        full_path = os.path.join(pathlib.Path().resolve(), "models", "PPOmodel_{}.zip".format(self._id))
        self.model.save(full_path)

    def test(self):
        self.env = gym.make('gym_phishing:RLPTest-v0')
        obs = self.env.reset()
        self.model.set_env(self.env)
        contador = 0
        for _ in range(1000):
            action, _states = self.model.predict(obs, deterministic=True)
            obs, reward, done, info = self.env.step(action)
            self.env.render()
            if info['acertou']:
                contador += 1
            if done:
                obs = self.env.reset()
        print("Acurácia: {}".format(contador/1000))
