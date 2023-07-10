import pandas as pd


def save_to_csv(resultados, simulacao):
    resultados = pd.DataFrame(resultados, index=['DQN', 'A2C', 'PPO'])
    resultados.to_csv('results/{}.csv'.format(simulacao))
