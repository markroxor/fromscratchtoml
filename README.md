# fromscratchtoml
https://markroxor.github.io/fromscratchtoml/

[![Build Status](https://travis-ci.org/markroxor/fromscratchtoml.svg?branch=master)](https://travis-ci.org/markroxor/fromscratchtoml)
[![Coverage Status](https://coveralls.io/repos/github/markroxor/fromscratchtoml/badge.svg?branch=master)](https://coveralls.io/github/markroxor/fromscratchtoml?branch=master)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Gitter chat](https://badges.gitter.im/gitterHQ/gitter.png)](https://gitter.im/jellAIfish/from-scratch-to-ml)
[![PyPI version](https://badge.fury.io/py/fromscratchtoml.svg)](https://badge.fury.io/py/fromscratchtoml)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/markroxor/fromscratchtoml/master)

*An intuitive machine learning library for beginners, in python.*  
This library is my code base for storing code of popular machine learning algorithms from scratch while I understand them, keeping _code-readability_ and _simplicity_ over efficiency.

**Demo link - https://markroxor.github.io/fromscratchtoml/showroom/**  
_Example [snippet from codebase](https://github.com/markroxor/fromscratchtoml/blob/master/fromscratchtoml/svm/svc.py#L82) for fitting SVC :_
``` python
    def fit(self, X, y, multiplier_threshold=1e-5):
        """Fits the svc model on training data.
        Parameters
        ----------
        X : numpy.array
            The training features.
        y : numpy.array
            The training labels.
        multiplier_threshold : float
            The threshold for selecting lagrange multipliers.
        Returns
        -------
        kernel_matrix : list of svm.SVC
            A list of all the classifiers used for multi class classification
        """
        X = np.array(X)
        self.y = y
        self.n = self.y.shape[0]

        self.uniques, self.ind = np.unique(self.y, return_index=True)
        self.n_classes = len(self.uniques)

        # Do multi class classification
        if sorted(self.uniques) != [-1, 1]:
            y_list = [np.where(self.y == u, 1, -1) for u in self.uniques]

            for y_i in y_list:
                # Copy the current initializer
                clf = SVC()
                clf.kernel = self.kernel
                clf.C = self.C

                self.classifiers.append(clf.fit(X, y_i))
            return

        # create a gram matrix by taking the outer product of y
        gram_matrix_y = np.outer(self.y, self.y)
        K = self.__create_kernel_matrix(X)
        gram_matrix_xy = gram_matrix_y * K

        P = cvxopt.matrix(gram_matrix_xy)
        q = cvxopt.matrix(-np.ones(self.n))

        G1 = cvxopt.spmatrix(-1.0, range(self.n), range(self.n))
        G2 = cvxopt.spmatrix(1, range(self.n), range(self.n))
        G = cvxopt.matrix([[G1, G2]])

        h1 = cvxopt.matrix(np.zeros(self.n))
        h2 = cvxopt.matrix(np.ones(self.n) * self.C)
        h = cvxopt.matrix([[h1, h2]])

        A = cvxopt.matrix(self.y.astype(np.double)).trans()
        b = cvxopt.matrix(0.0)

        lagrange_multipliers = np.array(list(cvxopt.solvers.qp(P, q, G, h, A,
                                                                b)['x']))

        lagrange_multiplier_indices = np.greater_equal(lagrange_multipliers, multiplier_threshold)
        lagrange_multiplier_indices = list(map(list, lagrange_multiplier_indices.nonzero()))[0]

        self.support_vectors = X[lagrange_multiplier_indices]
        self.support_vectors_y = self.y[lagrange_multiplier_indices]
        self.support_lagrange_multipliers = lagrange_multipliers[lagrange_multiplier_indices]
        self.b = 0
        self.n_support_vectors = self.support_vectors.shape[0]

        for i in range(self.n_support_vectors):
            kernel_trick = K[[lagrange_multiplier_indices[i]], lagrange_multiplier_indices]

            self.b += self.support_vectors_y[i] - np.sum(self.support_lagrange_multipliers *
                      self.support_vectors_y * kernel_trick)

        self.b /= self.n_support_vectors

        self.classifiers = [self]
        return self
```

## CUDA Support (Unstable!)
[Cupy](https://cupy.chainer.org/) is used to take advantage of cuda computing of NVIDIA GPUs.
> CuPy is an open-source matrix library accelerated with NVIDIA CUDA. It also uses CUDA-related libraries including cuBLAS, cuDNN, cuRand, cuSolver, cuSPARSE, cuFFT and NCCL to make full use of the GPU architecture.

The backend for mathematical computations can be switched using -   
```python3
   import fromscratchtoml
   fromscratchtoml.use("numpy") # for numpy backend (default)
   fromscratchtoml.use("cupy")  # or cupy backend
```
Since Travis (cloud CI) doesn't support cupy. Cupy must be installed manually using -   
`pip install cupy`

_NOTE_ - To mantain consistency with the backend, it is recommended to use `import fromscratchtoml as np` everywhere in your code.

## Installation
#### Python pypi <a name="pypi"></a>
You can install from [pypi](https://pypi.org/project/fromscratchtoml/).

    pip install fromscratchtoml

This is the most stable build.


#### Compiling manually <a name="manual"></a>
If you are interested in installing the most bleeding edge but not too stable version. You can install
from source -  


    git clone https://github.com/markroxor/fromscratchtoml.git
    pip install -r requirements.txt  
    python setup.py install

## Tutorials and support
* Well documented API usage is available at [pydoc](https://www.pydoc.io/pypi/fromscratchtoml-0.0.1/)
* Feature requests and bugs reports can be tracked on [issue tracker](https://github.com/markroxor/fromscratchtoml/issues).

## Motivation

![](https://imgs.xkcd.com/comics/tasks.png)   
> Good programmers know what to write. Great ones know what to rewrite (and reuse).
While I don't claim to be a great programmer, I try to imitate one. An important trait of the great ones is constructive laziness. They know that you get an A not for effort but for results, and that it's almost always easier to start from a good partial solution than from nothing at all.
Linus Torvalds, for example, didn't actually try to write Linux from scratch. Instead, he started by reusing code and ideas from Minix, a tiny Unix-like operating system for PC clones. Eventually all the Minix code went away or was completely rewritten—but while it was there, it provided scaffolding for the infant that would eventually become Linux.  
~ [The Cathedral Bazaar](http://www.catb.org/esr/writings/cathedral-bazaar/cathedral-bazaar/ar01s02.html)

> The Game of Life (or simply Life) is not a game in the conventional sense. There
are no players, and no winning or losing. Once the "pieces" are placed in the
starting position, the rules determine everything that happens later.
Nevertheless, Life is full of surprises! In most cases, it is impossible to look
at a starting position (or pattern) and see what will happen in the future. The
only way to find out is to follow the rules of the game.  
~ [Paul Callahan](http://www.math.com/students/wonders/life/life.html)

> What I cannot create, I do not understand.  
~ [Richard Feynman](https://en.wikiquote.org/wiki/Richard_Feynman)

## DISCLAIMER:
I created this library while understanding the coding aspects of the machine learning algorithms from various sources and blogs around the internet. A non-exhaustive list of those resources can be found at [the Wiki page](https://github.com/markroxor/fromscratchtoml/wiki).
