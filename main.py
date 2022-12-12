import os
from rl.A2C import A2C
from rl.DQN import DQN
from rl.PPO import PPO
from config.default import RLConfig
from utils.datafilter import load_dataset

# Usado para prevenir o BUG "Initializing libomp.dylib, but found libomp.dylib already initialized."
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Define Configurações Gerais para todas as simulações
config = RLConfig(600, 0.0007, 100000, 0, [128, 128])
print("Definindo Configuração Padrão: {}".format(config))

for _ in range(100):
    # Carrega a base de dados e realiza a separação entre Treinamento (2/3) e Teste (1/3)
    print("Carregando base de Dados.")
    load_dataset(test_size=0.33)
    print("Iniciando Simulações")
    print()

    print("Iniciando Simulação DQN")
    simulacao1 = DQN(1, config=config)
    print("Iniciando Treinamento DQN")
    simulacao1.train()
    print("Iniciando Teste DQN")
    simulacao1.test()
    print()

    print("Iniciando Simulação A2C")
    simulacao2 = A2C(2, config=config)
    print("Iniciando Treinamento A2C")
    simulacao2.train()
    print("Iniciando Teste A2C")
    simulacao2.test()
    print()

    print("Iniciando Simulação PPO")
    simulacao3 = PPO(3, config=config)
    print("Iniciando Treinamento PPO")
    simulacao3.train()
    print("Iniciando Tese PPO")
    simulacao3.test()
