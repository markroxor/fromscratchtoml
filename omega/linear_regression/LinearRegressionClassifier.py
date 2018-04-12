import torch as ch
import logging
from torch.autograd import Variable


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class LinearRegressionClassifier(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def fit(self, option):
        self.option = option
        ch.manual_seed(2)
        if option == 'Analytical':
            self.cov_x_y = []
            self.var_x = []
            mean_x = ch.mean(self.x)
            mean_y = ch.mean(self.y)
            for i in range(len(self.x)):
                self.cov_x_y.append((self.x[i] - mean_x) * (self.y[i] - mean_y))
                self.var_x.append((self.x[i] - mean_x) ** 2)
            self.b1 = sum(self.cov_x_y) / sum(self.var_x)
            self.b0 = mean_y - self.b1 * mean_x
        if option == 'SGD':
            x_data = Variable(ch.Tensor(self.x), requires_grad=False)
            y_data = Variable(ch.Tensor(self.y), requires_grad=False)
            self.beta = Variable(ch.randn(1, 1), requires_grad=True)
            self.alpha = Variable(ch.randn(1), requires_grad=True)
            optimizer = ch.optim.SGD([self.beta, self.alpha], lr=0.01)
            for i in range(1000):
                y_pred = x_data.mm(self.beta).add(self.alpha)
                loss = (y_pred - y_data).pow(2).sum()
                if i % 50 == 0:
                    logger.info("iteration {} loss {}".format(i, loss.data[0]))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
        if option == 'Adam':
            x_data = Variable(ch.Tensor(self.x), requires_grad=False)
            y_data = Variable(ch.Tensor(self.y), requires_grad=False)
            self.beta = Variable(ch.randn(1, 1), requires_grad=True)
            self.alpha = Variable(ch.randn(1), requires_grad=True)
            optimizer = ch.optim.Adam([self.beta, self.alpha], lr=0.01)
            for i in range(1000):
                y_pred = x_data.mm(self.beta).add(self.alpha)
                loss = (y_pred - y_data).pow(2).sum()
                if i % 50 == 0:
                    logger.info("iteration {} loss {}".format(i, loss.data[0]))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

    def predict(self, test_data):
        if self.option == 'Analytical':
            for i in range(len(test_data)):
                predicted_value = self.b0 + self.b1 * test_data[i]
                logger.info("Predicted value for test_data {} slope {} and bias {} is {}".format(test_data[i], self.b1,
                            self.b0, predicted_value))
        if self.option == 'SGD':
            for i in range(len(test_data)):
                predicted_value = self.alpha + self.beta * test_data[i]
                logger.info("Predicted value for test_data {} slope {} and bias {} is {}".format(test_data[i],
                            self.beta, self.alpha, predicted_value))
        if self.option == 'Adam':
            for i in range(len(test_data)):
                predicted_value = self.alpha + self.beta * test_data[i]
                logger.info("Predicted value for test_data {} slope {} and bias {} is {}".format(test_data[i],
                            self.beta, self.alpha, predicted_value))

    def save_model(self, file_path):
        ch.save(self.__dict__, file_path)
        return

    def load_model(self, file_path):
        self.__dict__ = ch.load(file_path)
        return
