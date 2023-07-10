import os

import gym

from rl.DQN import DQN
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

dqn = DQN(1, config)
dqn.load_model("models/dqn.zip")

a2c = DQN(2, config)
a2c.load_model("models/dqn.zip")

ppo = DQN(3, config)
ppo.load_model("models/dqn.zip")

tp = 0
tn = 0
fp = 0
fn = 0

for i in range(folds):
    env = gym.make('gym_phishing:RLPTest-v0', fold=i)
    obs = env.reset()
    ppo.model.set_env(env)
    for _ in range(config.test_size):
        action, _ = ppo.model.predict(obs, deterministic=True)

        # action igual a 0 (PPO dizendo tá dizendo que é phishing)
        # Quando a ação for '0' (Phishing)
        if action == 0:
            _, _, done, info = env.step2(action)
            if info['resultado'] == 'TP':
                tp += 1
            elif info['resultado'] == 'TN':
                tn += 1
            elif info['resultado'] == 'FP':
                fp += 1
            elif info['resultado'] == 'FN':
                fn += 1

        # caso contrário (PPO dizendo tá dizendo que não é phishing)
        # Então é preciso testar A2C e DQN
        else:
            a2c.model.set_env(env)
            action_a2c, _ = a2c.model.predict(obs, deterministic=True)

            dqn.model.set_env(env)
            action_dqn, _ = dqn.model.predict(obs, deterministic=True)

            # Se eles concordam
            if action_dqn == action_a2c:
                _, _, done, info = env.step2(action_dqn)
                if info['resultado'] == 'TP':
                    tp += 1
                elif info['resultado'] == 'TN':
                    tn += 1
                elif info['resultado'] == 'FP':
                    fp += 1
                elif info['resultado'] == 'FN':
                    fn += 1

            # Se eles discordam
            else:
                print("Saida")
