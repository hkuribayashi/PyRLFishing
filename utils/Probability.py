import math


class WAP:

    def __init__(self, model, precision, accuracy, recall, f1, probability):
        self.model = model
        w = math.tanh(precision) + math.tanh(accuracy) + math.tanh(recall) + math.tanh(f1)
        self.ens = ((w * probability) + (w * (1 - probability))) / w
