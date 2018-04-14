import torch as ch
import logging
from torch.autograd import Variable


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class LinearRegressionClassifier(object):
    """Objects of this class is a LinearRegressionClassifier.

   Examples
   --------
   >>> import omega as omg
   >>> import torch as ch
   >>> x = ch.Tensor([[24], [50], [15], [55], [14], [12], [18], [62]])
   >>> y = ch.Tensor([[21.54945196], [47.46446305], [17.21865634], [52.789],
                    [16.1234], [12.789], [19.5649], [60.5793278]])
   >>> lr = omg.linear_regression.LinearRegressionClassifier(x,y)
   >>> lr.fit('Adam')
   >>> lr.predict(x)

   """

    def __init__(self, x, y):
        """instanciate an object with two parameters.

        Parameters
        ----------
        x : a float tensor having area of house in square meter.
        y : a float tensor having price on the basis of area of house.

        """
        self.x = x
        self.y = y

    def fit(self, option):
        """ Finds suitable value of parameters which best fit the given output values.

        Parameters
        ----------
        option : We have implemented in two ways. The first approach is the Analytical approach
                 which computes the value of slope and intercept by finding the covariance
                 and variance of the (x, y) pair. The second approach uses two optimizers namely
                 stochastic gradient descent (SGD) and Adam. Therefore, the options are 'Analytical'
                 'SGD' and 'Adam'.

        """

        self.option = option
        if option == 'Analytical':
            self.cov_x_y = []
            self.var_x = []
            mean_x = ch.mean(self.x)
            mean_y = ch.mean(self.y)
            for i in range(len(self.x)):
                self.cov_x_y.append((self.x[i] - mean_x) * (self.y[i] - mean_y))
                self.var_x.append((self.x[i] - mean_x) ** 2)
            self.beta = sum(self.cov_x_y) / sum(self.var_x)
            self.alpha = mean_y - self.beta * mean_x
            return
        ch.manual_seed(2)
        x_data = Variable(ch.FloatTensor(self.x), requires_grad=False)
        y_data = Variable(ch.FloatTensor(self.y), requires_grad=False)
        self.beta = Variable(ch.randn(1, 1), requires_grad=True)
        self.alpha = Variable(ch.randn(1), requires_grad=True)

        if option == 'SGD':
            optimizer = ch.optim.SGD([self.beta, self.alpha], lr=0.01)
        elif option == 'Adam':
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
            """This function evaluates the test dataset by feed forwarding the learned
            weights across the network and calculating the number of correct evaluations
            by the network on test data.

            Parameters
            ----------
            test_data : list of (torch.Tensor, torch.Tensor) or a similar data type.
                The test data on which the results are evaluated generally after each
                epoch.

            """
            self.test_data = Variable(ch.FloatTensor(self.x), requires_grad=False)
            for td in self.test_data:
                    predicted_value = float(self.beta) * float(td) + float(self.alpha)
                    logger.info("Predicted value for test_data {} slope {} and bias "
                            "{} is {}".format(td, self.alpha, self.beta, predicted_value))

    def save_model(self, file_path):
        """This function saves the model in a file for loading it in future.

        Parameters
        ----------
        file_path : str
            The path to file where the model should be saved.

        """
        ch.save(self.__dict__, file_path)
        return

    def load_model(self, file_path):
        """This function loads the saved model from a file.

        Parameters
        ----------
        file_path : str
            The path of file from where the model should be retrieved.

        """
        self.__dict__ = ch.load(file_path)
        return
