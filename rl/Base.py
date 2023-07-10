from abc import ABC, abstractmethod

from config.default import RLConfig


class BaseModel(ABC):
    def __init__(self, id_, config=None):
        # Define um ID para a Instância/Simulacao
        self.model = None
        self._id = id_

        if config is None:
            self.config = RLConfig()
        else:
            self.config = config

        # Total de folds do Cross Validation
        self.test_size = config.test_size

        # Total de Folds do Cross Validation
        self.folds = self.config.folds

        # Realiza a criação do atributo que guarda o Ambiente Gym
        self.env = None

        # Define qual a arquitetura da rede neural interna da DQN
        self.policy = dict(net_arch=self.config.net_arch['DQN'])

        # Define o total de timesteps
        self.total_timesteps = self.config.total_timesteps

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def load_model(self, path):
        pass

    def save_model(self, path):
        pass
