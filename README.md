# fromscratchtoml
https://jellAIfish.github.io/fromscratchtoml/

[![Build Status](https://travis-ci.org/jellAIfish/fromscratchtoml.svg?branch=master)](https://travis-ci.org/jellAIfish/fromscratchtoml)
[![Coverage Status](https://coveralls.io/repos/github/jellAIfish/fromscratchtoml/badge.svg?branch=master)](https://coveralls.io/github/jellAIfish/fromscratchtoml?branch=master)
[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Gitter chat](https://badges.gitter.im/gitterHQ/gitter.png)](https://gitter.im/jellAIfish/from-scratch-to-ml)



*An intuitive machine learning library for beginners, in python.*  
This library targets individuals who want to learn machine learning code from scratch keeping _code-readability_ and _simplicity_ over efficiency.

## Highlights
* Every bit of code is well documented and easy to understand.
* The only dependency used is `pytorch` because it is a better alternative of `numpy` as it can run computations by harnessing the power of GPUs as well.
* `pytorch` is only used for mathematical computations - mostly matrix multiplication and operations. It is not used as a backend for machine learning algorithms

## Installation
#### Prerequisites
Before proceeding with installation -
One of the dependencies of our project is `pytorch` which is not availabe on `pip` as of now, you must install it manually -
* Easy (recommended) - Run the script

        wget https://raw.githubusercontent.com/jellAIfish/fromscratchtoml/master/scripts/install_pytorch.sh
        ./install_pytorch.sh

     It will auto-identify the system configuration and install the corresponding `pytorch` version _without GPU support_.
* Manually - In case you face any error, you can head over to [pytorch.org](http://pytorch.org/) for a more customized installation.

#### Python pypi <a name="pypi"></a>
You can install from [pipy](https://pypi.org/project/fromscratchtoml/).

    pip install fromscratchtoml

This is the most stable build.


#### Compiling manually <a name="manual"></a>
If you are interested in installing the most bleeding edge but not too stable version. You can install
from source -  

    pip install -r requirements.txt  
    git clone https://github.com/jellAIfish/fromscratchtoml.git
    python setup.py install

## Tutorials and support
* Well documented API usage is available at [pydoc](https://www.pydoc.io/pypi/fromscratchtoml-0.0.1/)
* For discussion and support visit - [the gitter channel](https://gitter.im/jellAIfish/from-scratch-to-ml)
* Development discussions and bugs reports can be tracked on [issue tracker](https://github.com/jellAIfish/fromscratchtoml/issues).

## Future endeavours
* This library will form a back-bone for teaching and guiding budding machine learning developers via vBlogs.
