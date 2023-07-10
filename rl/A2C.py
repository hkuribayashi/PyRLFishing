import gym
from stable_baselines3 import A2C as A2C_
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike

from rl.Base import BaseModel
from utils.metrics import getPrecision, getRecall, getAccuracy, getFalsePositiveRate, getFalseNegativeRate, \
    getF1Score, processaResultados


class A2C(BaseModel):
    def __init__(self, id_, config=None):
        # Chama o construtor da Super Classe
        super().__init__(id_, config)

        # Instancia o Modelo
        self.model = None

    def run(self):
        resultados = []
        for i in range(self.folds):
            # Instancia o ambiente Gym Customizado para o Problema
            self.env = gym.make('gym_phishing:RLPTraining-v0', fold=i)

            # Instancia um novo Modelo
            self.model = A2C_("MlpPolicy",
                              self.env,
                              learning_rate=self.config.learning_rate,
                              verbose=self.config.verbose,
                              policy_kwargs=dict(optimizer_class=RMSpropTFLike,
                                                 optimizer_kwargs=dict(eps=1e-5),
                                                 net_arch=self.config.net_arch['A2C']))

            # Realiza o treinamento do Modelo
            self.model.learn(total_timesteps=self.total_timesteps)

            # Armazena os resultados obtidos
            resultados.append(self._test(i))

        return processaResultados(resultados)

    def _test(self, fold):
        self.env = gym.make('gym_phishing:RLPTest-v0', fold=fold)
        obs = self.env.reset()
        self.model.set_env(self.env)
        tp = 0
        tn = 0
        fp = 0
        fn = 0

        for _ in range(self.test_size):
            action, _states = self.model.predict(obs, deterministic=True)
            obs, reward, done, info = self.env.step(action)
            if info['resultado'] == 'TP':
                tp += 1
            elif info['resultado'] == 'TN':
                tn += 1
            elif info['resultado'] == 'FP':
                fp += 1
            elif info['resultado'] == 'FN':
                fn += 1

            self.env.render()
            if done:
                obs = self.env.reset()

        precision = getPrecision(tp, fp)
        recall = getRecall(tp, fn)
        accuracy = getAccuracy(tp, tn, fp, fn)
        fpr = getFalsePositiveRate(fp, tn)
        fnr = getFalseNegativeRate(fn, tp)
        f1score = getF1Score(tp, fp, fn)
        result = {'precision': precision, 'recall': recall, 'accuracy': accuracy,
                  'fpr': fpr, 'fnr': fnr, 'f1score': f1score}

        return result
