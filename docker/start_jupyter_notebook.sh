#!/bin/bash

PORT=$1
NOTEBOOK_DIR=/fromscratchtoml/docs/notebooks

jupyter notebook --no-browser --ip=* --port=${PORT} --allow-root --notebook-dir=${NOTEBOOK_DIR} --NotebookApp.token=\"\"
