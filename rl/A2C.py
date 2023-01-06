from stable_baselines3 import A2C as A2C_
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike

from rl.Base import BaseModel


class A2C(BaseModel):
    def __init__(self, id_, config=None):
        # Chama o construtor da Super Classe
        super().__init__(id_, config)

        # Instancia o Modelo
        self.model = A2C_("MlpPolicy",
                          self.env,
                          learning_rate=config.learning_rate,
                          verbose=config.verbose,
                          policy_kwargs=dict(optimizer_class=RMSpropTFLike,
                                             optimizer_kwargs=dict(eps=1e-5),
                                             net_arch=self.config.net_arch['A2C']))
