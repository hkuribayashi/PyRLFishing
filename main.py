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

resultados_dqn = dict()
resultados_dqn['precision'] = 0
resultados_dqn['recall'] = 0
resultados_dqn['accuracy'] = 0

resultados_ppo = dict()
resultados_ppo['precision'] = 0
resultados_ppo['recall'] = 0
resultados_ppo['accuracy'] = 0

resultados_a2c = dict()
resultados_a2c['precision'] = 0
resultados_a2c['recall'] = 0
resultados_a2c['accuracy'] = 0

realizacoes = 3
for _ in range(realizacoes):
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
    precision, recall, accuracy = simulacao1.test()
    resultados_dqn['precision'] += precision
    resultados_dqn['recall'] += recall
    resultados_dqn['accuracy'] += accuracy
    print()

    ''' print("Iniciando Simulação A2C")
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
    '''

print(resultados_dqn['precision']/realizacoes)
