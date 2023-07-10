import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, KFold


def load_dataset(cv=10):
    # Importanto do Dataset
    # Obtido em https://www.kaggle.com/datasets/akashkr/phishing-website-dataset

    df = pd.read_csv(os.path.join(os.getcwd(), "data", "dataset.csv"))
    heatmap = df.corr()[['Result']].sort_values(by='Result', ascending=False)
    unselected_features = list()
    unselected_features.append('Result')
    counter = 0
    for feature in heatmap.axes[0]:
        if counter > 9:
            unselected_features.append(feature)
        counter = counter + 1

    # Dividindo os Dados em Treinamento e Teste
    x = df.drop(unselected_features, axis=1)
    y = df['Result']

    # Método via k-Fold cross-validation
    kf = KFold(n_splits=cv, shuffle=True, random_state=None)

    # Estabelece um contador
    counter = 0

    path = os.path.join(os.getcwd(), "gym-phishing", "gym_phishing", "data")
    for train_index, test_index in kf.split(x):
        x_train, x_test = x.iloc[train_index, :], x.iloc[test_index, :]
        y_train, y_test = y[train_index], y[test_index]

        # Salvando CSV de Treino e Teste
        filepath = os.path.join(path, 'trainX{}.csv'.format(counter))
        x_train.to_csv(filepath, index=False)

        filepath = os.path.join(path, 'trainY{}.csv'.format(counter))
        y_train.to_csv(filepath, index=False)

        filepath = os.path.join(path, 'testX{}.csv'.format(counter))
        x_test.to_csv(filepath, index=False)

        filepath = os.path.join(path, 'testY{}.csv'.format(counter))
        y_test.to_csv(filepath, index=False)

        counter += 1

    return len(y_test.index)
