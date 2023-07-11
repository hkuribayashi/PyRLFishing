import os

from rl.A2C import A2C
from rl.DQN import DQN
from rl.PPO import PPO
from config.default import RLConfig
from utils.datafilter import load_dataset
from utils.results import save_to_csv

# Usado para prevenir o BUG "Initializing libomp.dylib, but found libomp.dylib already initialized."
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Define o total de folds do Cross Validation
folds = 5

print("Carregando base de Dados.")
test_size = load_dataset(cv=folds)
print()

# Define Configurações Gerais para todas as simulações
config = RLConfig(test_size=test_size, image_resolution=600, learning_rate=0.0007, total_timesteps=100000, verbose=0,
                  folds=folds, net_arch=256)
print("Definindo Configuração Padrão: {}".format(config))
resultados = list()

print("Iniciando Execução DQN")
simulacao1 = DQN(0, config)
resultados_dqn = simulacao1.run()
print(resultados_dqn)
resultados.append(resultados_dqn)
print("Teste DQN Finalizado")
print()

print("Iniciando Execução A2C")
simulacao2 = A2C(0, config)
resultados_a2c = simulacao2.run()
print(resultados_a2c)
resultados.append(resultados_a2c)
print("Teste A2C Finalizado")
print()

print("Iniciando Treinamento PPO")
simulacao3 = PPO(0, config)
resultados_ppo = simulacao3.run()
print(resultados_ppo)
resultados.append(resultados_ppo)
print("Treinamento PPO Finalizado")
print()

save_to_csv(resultados, 'simulacao0')
