import os
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


def load_dataset(test_size=0.33):
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
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)

    # Salvando CSV de Treino e Teste
    filepath = Path('../gym-phishing/gym_phishing/data/trainX.csv')
    filepath.parent.mkdir(parents=True, exist_ok=True)
    x_train.to_csv(filepath, index=False)

    filepath = Path('../gym-phishing/gym_phishing/data/trainY.csv')
    filepath.parent.mkdir(parents=True, exist_ok=True)
    y_train.to_csv(filepath, index=False)

    filepath = Path('../gym-phishing/gym_phishing/data/testX.csv')
    filepath.parent.mkdir(parents=True, exist_ok=True)
    x_test.to_csv(filepath, index=False)

    filepath = Path('../gym-phishing/gym_phishing/data/testY.csv')
    filepath.parent.mkdir(parents=True, exist_ok=True)
    y_test.to_csv(filepath, index=False)
