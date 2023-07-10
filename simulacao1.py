import os
from rl.A2C import A2C
from rl.DQN import DQN
from rl.PPO import PPO
from config.default import RLConfig
from utils.datafilter import load_dataset

# Usado para prevenir o BUG "Initializing libomp.dylib, but found libomp.dylib already initialized."
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Define o total de folds do Cross Validation
folds = 5

print("Carregando base de Dados.")
test_size = load_dataset(cv=folds)
print()

# Define Configurações Gerais para todas as simulações
config = RLConfig(test_size=test_size, image_resolution=600, learning_rate=0.0007, total_timesteps=100000, verbose=0,
                  folds=folds, net_arch=128)
print("Definindo Configuração Padrão: {}".format(config))


print("Iniciando Execução DQN")
simulacao1 = DQN(1, config)
resultados = simulacao1.run()
print(resultados)
print("Teste DQN Finalizado")
print()

print("Iniciando Execução A2C")
simulacao2 = A2C(2, config)
resultados = simulacao2.run()
print(resultados)
print("Teste A2C Finalizado")
print()

print("Iniciando Treinamento PPO")
simulacao3 = PPO(3, config)
resultados = simulacao3.run()
print(resultados)
print("Treinamento PPO Finalizado")
print()
