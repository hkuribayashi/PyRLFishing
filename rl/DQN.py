import gym
from stable_baselines3 import DQN as DQN_
from rl.Base import BaseModel

from utils.metrics import processaResultados


class DQN(BaseModel):
    def __init__(self, id_, config=None):
        # Chama o construtor da Super Classe
        super().__init__(id_, config)

        # Cria um env temporário
        self.env = gym.make('gym_phishing:RLPTraining-v0', fold=0)

        # Instancia o Modelo
        self.model = DQN_("MlpPolicy",
                          self.env,
                          policy_kwargs=self.policy,
                          learning_rate=self.config.learning_rate,
                          verbose=self.config.verbose)

    def run(self):
        resultados = []
        for i in range(self.folds):
            # Instancia o ambiente Gym Customizado para o Problema
            self.env = gym.make('gym_phishing:RLPTraining-v0', fold=i)

            # Setta o Ambiente
            self.model.set_env(self.env)

            # Realiza o treinamento do Modelo
            self.model.learn(total_timesteps=self.total_timesteps)

            # Salva o Modelo
            self.save_model("dqn_{}".format(self._id))

            # Armazena os resultados obtidos
            resultados.append(self._test(i))

        return processaResultados(resultados)

    def load_model(self, path):
        self.model = DQN_.load(path)
