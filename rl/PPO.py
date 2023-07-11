import gym
from stable_baselines3 import PPO as PPO_
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike

from rl.Base import BaseModel
from utils.metrics import processaResultados


class PPO(BaseModel):

    def __init__(self, id_, config=None):
        # Chama o construtor da Super Classe
        super().__init__(id_, config)

        # Cria um env temporário
        self.env = gym.make('gym_phishing:RLPTraining-v0', fold=0)

        # Instancia um novo Modelo
        self.model = PPO_("MlpPolicy",
                          self.env,
                          learning_rate=self.config.learning_rate,
                          verbose=self.config.verbose,
                          policy_kwargs=dict(optimizer_class=RMSpropTFLike,
                                             optimizer_kwargs=dict(eps=1e-5),
                                             net_arch=self.config.net_arch['PPO']))

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
            self.save_model("ppo_{}".format(self._id))

            # Armazena os resultados obtidos
            resultados.append(self._test(i))

        return processaResultados(resultados)

    def load_model(self, path):
        self.model = PPO_.load(path)
