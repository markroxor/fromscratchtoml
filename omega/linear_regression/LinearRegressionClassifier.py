import torch as ch

import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class LinearRegressionClassifier(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def fit(self):
        self.cov_x_y = []
        self.var_x = []
        mean_x = ch.mean(self.x)
        mean_y = ch.mean(self.y)
        for i in range(len(self.x)):
            self.cov_x_y.append((self.x[i] - mean_x) * (self.y[i] - mean_y))
            self.var_x.append((self.x[i] - mean_x) ** 2)
        self.b1 = sum(self.cov_x_y) / sum(self.var_x)
        self.b0 = mean_y - self.b1 * mean_x

    def predict(self, test_data):
        for i in range(len(test_data)):
            predicted_value = self.b0 + self.b1 * test_data[i]
            logger.info("Predicted value for test_data {} slope {} and bias {} is {}".format(test_data[i], self.b1,
                        self.b0, predicted_value))

    def save_model(self, file_path):
        ch.save(self.__dict__, file_path)
        return

    def load_model(self, file_path):
        self.__dict__ = ch.load(file_path)
        return
