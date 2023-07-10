import os

from config.default import RLConfig
from rl.DQN import DQN
from utils.datafilter import load_dataset

# Usado para prevenir o BUG "Initializing libomp.dylib, but found libomp.dylib already initialized."
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Define o total de folds do Cross Validation
folds = 5

print("Carregando base de Dados.")
test_size = load_dataset(cv=folds)
print()

# Define Configurações para o DQN
config = RLConfig(test_size=test_size,
                  image_resolution=600,
                  learning_rate=0.0007,
                  total_timesteps=100000,
                  verbose=0,
                  folds=folds, net_arch=128)
print("Definindo Configuração DQN: {}".format(config))
simulacao1 = DQN(1, config)
DQN.load_model("models/dqn.zip")
