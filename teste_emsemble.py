import os

import gym

from rl.A2C import A2C
from rl.DQN import DQN
from rl.PPO import PPO
from config.default import RLConfig
from utils.datafilter import load_dataset
from utils.metrics import getPrecision, getAccuracy, getRecall, getFalsePositiveRate, getFalseNegativeRate, getF1Score, \
    processaResultados

# Usado para prevenir o BUG "Initializing libomp.dylib, but found libomp.dylib already initialized."
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Define o total de folds do Cross Validation
folds = 5

print("Carregando base de Dados.")
test_size = load_dataset(cv=folds)
print()

# Define Configurações Gerais para todas as simulações
config = RLConfig(test_size=test_size, image_resolution=600, learning_rate=0.0001, total_timesteps=100000, verbose=0,
                  folds=folds, net_arch=128)
print("Definindo Configuração Padrão: {}".format(config))

dqn = DQN(1, config)
dqn.load_model("models/dqn_0.zip")

a2c = A2C(2, config)
a2c.load_model("models/a2c_0.zip")

ppo0 = PPO(3, config)
ppo0.load_model("models/ppo_0.zip")

ppo1 = PPO(3, config)
ppo1.load_model("models/ppo_1.zip")

ppo2 = PPO(3, config)
ppo2.load_model("models/ppo_2.zip")

total = list()


for i in range(folds):

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    env = gym.make('gym_phishing:RLPTest-v0', fold=i)
    for _ in range(config.test_size):
        obs = env.reset()

        ppo0.model.set_env(env)
        action0, _ = ppo0.model.predict(obs, deterministic=True)

        ppo1.model.set_env(env)
        action1, _ = ppo1.model.predict(obs, deterministic=True)

        ppo2.model.set_env(env)
        action2, _ = ppo2.model.predict(obs, deterministic=True)

        a2c.model.set_env(env)
        action_a2c, _ = a2c.model.predict(obs, deterministic=True)

        dqn.model.set_env(env)
        action_dqn, _ = dqn.model.predict(obs, deterministic=True)

        if action0 == action1 == action2 == action_a2c == action_dqn:
            _, _, done, info = env.step2(action0)
            if info['resultado'] == 'TP':
                tp += 1
            elif info['resultado'] == 'TN':
                tn += 1
            elif info['resultado'] == 'FP':
                fp += 1
            elif info['resultado'] == 'FN':
                fn += 1
        elif (action0 == action1 == action2 == 1) and (action_dqn == action_a2c == 0):
            _, _, done, info = env.step2(action_a2c)
            if info['resultado'] == 'TP':
                tp += 1
            elif info['resultado'] == 'TN':
                tn += 1
            elif info['resultado'] == 'FP':
                fp += 1
            elif info['resultado'] == 'FN':
                fn += 1
        elif action0 == action1 == action2 == 0:
            _, _, done, info = env.step2(action0)
            if info['resultado'] == 'TP':
                tp += 1
            elif info['resultado'] == 'TN':
                tn += 1
            elif info['resultado'] == 'FP':
                fp += 1
            elif info['resultado'] == 'FN':
                fn += 1
        elif action0 + action1 + action2 < 2:
            _, _, done, info = env.step2(0)
            if info['resultado'] == 'TP':
                tp += 1
            elif info['resultado'] == 'TN':
                tn += 1
            elif info['resultado'] == 'FP':
                fp += 1
            elif info['resultado'] == 'FN':
                fn += 1
        else:
            _, _, done, info = env.step2(action_dqn)
            if info['resultado'] == 'TP':
                tp += 1
            elif info['resultado'] == 'TN':
                tn += 1
            elif info['resultado'] == 'FP':
                fp += 1
            elif info['resultado'] == 'FN':
                fn += 1

    precision = getPrecision(tp, fp)
    recall = getRecall(tp, fn)
    accuracy = getAccuracy(tp, tn, fp, fn)
    fpr = getFalsePositiveRate(fp, tn)
    fnr = getFalseNegativeRate(fn, tp)
    f1score = getF1Score(tp, fp, fn)
    result = {'precision': precision, 'recall': recall, 'accuracy': accuracy, 'fpr': fpr, 'fnr': fnr, 'f1score': f1score}

    print(result)
    total.append(result)

print(processaResultados(total))
