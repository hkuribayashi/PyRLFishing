from stable_baselines3 import PPO as PPO_

from rl.Base import BaseModel


class PPO(BaseModel):
    def __init__(self, id_, config=None):

        super().__init__(id_, config)

        self.model = PPO_("MlpPolicy", self.env, learning_rate=config.learning_rate, verbose=config.verbose)
