import pandas as pd

from utils.Probability import WAP


def save_to_csv(resultados, simulacao):
    resultados = pd.DataFrame(resultados, index=['DQN', 'A2C', 'PPO'])
    resultados.to_csv('results/{}.csv'.format(simulacao))


def read_results():
    ens_list = list()
    for i in range(4):
        sim = pd.read_csv('results/simulacao{}.csv'.format(i))
        for j in range(3):
            ens_sim = WAP(sim['id'][j], sim['precision_media'][j], sim['accuracy_media'][j],
                               sim['recall_media'][j], sim['f1score_media'][j], sim['probability_media'][j])
            ens_list.append(ens_sim)
    return ens_list
