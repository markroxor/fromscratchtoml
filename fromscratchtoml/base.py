import inspect


class BaseModel(object):

        def __repr__(self):
            class_name = self.__class__.__name__
            _dict = {}
            for arg in inspect.getargspec(self.__init__).args:
                if arg in self.__dict__:
                    _dict[arg] = self.__dict__[arg]

            return '%s(%s)' % (class_name, _dict)
