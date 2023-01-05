from stable_baselines3 import A2C as A2C_
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike

from rl.Base import BaseModel


class A2C(BaseModel):
    def __init__(self, id_, config=None):
        super().__init__(id_, config)

        # Define qual a arquitetura da rede neural interna da DQN
        self.policy = dict(net_arch=self.config.net_arch,
                           optimizer_class=RMSpropTFLike,
                           optimizer_kwargs=dict(eps=1e-5))

        # Instancia o Modelo
        self.model = A2C_("MlpPolicy",
                          self.env,
                          policy_kwargs=self.policy,
                          learning_rate=self.config.learning_rate,
                          verbose=self.config.verbose)
