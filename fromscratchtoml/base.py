import torch as ch
import inspect


class BaseModel(object):
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

        def __repr__(self):
            class_name = self.__class__.__name__
            _dict = {}
            for arg in inspect.getargspec(self.__init__).args:
                if arg in self.__dict__:
                    _dict[arg] = self.__dict__[arg]

            return '%s(%s)' % (class_name, _dict)
