from stable_baselines3 import DQN as DQN_
from rl.Base import BaseModel


class DQN(BaseModel):
    def __init__(self, id_, config=None):
        # Chama o construtor da Super Classe
        super().__init__(id_, config)

        # Instancia o Modelo
        self.model = DQN_("MlpPolicy",
                          self.env,
                          policy_kwargs=self.policy,
                          learning_rate=self.config.learning_rate,
                          verbose=self.config.verbose)
