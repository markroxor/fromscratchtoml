#!/bin/bash

PORT=$1
NOTEBOOK_DIR=/fromscratchtoml/docs/notebooks
DEFAULT_URL=/notebooks/neural_network.ipynb

jupyter notebook --no-browser --ip=* --port=${PORT} --allow-root --notebook-dir=${NOTEBOOK_DIR} --NotebookApp.token=\"\" --NotebookApp.default_url=${DEFAULT_URL}
