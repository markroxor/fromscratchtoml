# Omega

*The most intuitive machine learning library.*  
[![Build Status](https://travis-ci.org/jellAIfish/omega.svg?branch=master)](https://travis-ci.org/jellAIfish/omega)
[![Coverage Status](https://coveralls.io/repos/github/jellAIfish/omega/badge.svg?branch=master)](https://coveralls.io/github/jellAIfish/omega?branch=master)
[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

This library targets individuals who want to learn machine learning code from scratch keeping _code-readability_ and _simplicity_ over efficiency.

## Highlights
* Every bit of code is well documented and easy to understand.
* The only dependency used is `pytorch` because it is a better alternative of `numpy` as it can run computations by harnessing the power of GPUs as well.
* `pytorch` is only used for mathematical computations - mostly matrix multiplication and operations. It is not used as a backend for machine learning algorithms 

## Installation
The sole dependency of our project is `pytorch` and since it is not available via `pip`, you must install it -
* Easy (recommended) - Run the script `omega/blob/master/scripts/install_pytorch.sh`. It will auto-identify the system configuration and install the corresponding `pytorch` version _without GPU support_. 
* Manually - In case you face any error, you can head over to [pytorch.org](http://pytorch.org/) for a more customized installation.

We currently do not provide installation via `pip` but you can still install the bleeding edge build by -
* Clone - `git clone https://github.com/jellAIfish/omega.git`
* Install - `python setup.py install`

## Tutorials and support
* The well documented API usage is available at [https://jellaifish.github.io/omega/](https://jellaifish.github.io/omega/)
* For discussion and support visit - [the slack channel](https://jellaifish.slack.com)
* Development discussions and bugs reports can be tracked on [issue tracker](https://github.com/jellAIfish/omega/issues).

## Future endeavours
* This library will form a back-bone for teaching and guiding budding machine learning developers via vBlogs.
