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
resultados_dqn['fpr'] = 0
resultados_dqn['fnr'] = 0

resultados_ppo = dict()
resultados_ppo['precision'] = 0
resultados_ppo['recall'] = 0
resultados_ppo['accuracy'] = 0
resultados_ppo['fpr'] = 0
resultados_ppo['fnr'] = 0

resultados_a2c = dict()
resultados_a2c['precision'] = 0
resultados_a2c['recall'] = 0
resultados_a2c['accuracy'] = 0
resultados_a2c['fpr'] = 0
resultados_a2c['fnr'] = 0

realizacoes = 10
for i in range(realizacoes):

    # Iniciando a Simulação
    print("Simulação {}".format(i))

    # Carrega a base de dados e realiza a separação entre Treinamento (2/3) e Teste (1/3)
    print("Carregando base de Dados.")
    load_dataset(test_size=0.33)
    print()

    print("Iniciando Treinamento DQN")
    simulacao1 = DQN(1, config=config)
    simulacao1.train()
    print("Treinamento DQN Finalizado")
    print("Iniciando Teste DQN")
    resultados = simulacao1.test()
    resultados_dqn['precision'] += resultados['precision']
    resultados_dqn['recall'] += resultados['recall']
    resultados_dqn['accuracy'] += resultados['accuracy']
    resultados_dqn['fpr'] += resultados['fpr']
    resultados_dqn['fnr'] += resultados['fnr']
    print("Teste DQN Finalizado")
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

print("Resultados DQN")
print("Precisão: {}".format(resultados_dqn['precision']/realizacoes))
print("Recall: {}".format(resultados_dqn['recall']/realizacoes))
print("Acurácia: {}".format(resultados_dqn['accuracy']/realizacoes))
print("FPR: {}".format(resultados_dqn['fpr']/realizacoes))
print("FNR: {}".format(resultados_dqn['fnr']/realizacoes))

