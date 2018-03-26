#!/bin/bash

py_version=$(python -c "import sys; print (sys.version_info[:])")

py_major=$(echo $py_version| cut -d ',' -f 1)
py_minor=$(echo $py_version| cut -d ',' -f 2)

if [[ "$py_major" = "(2" ]] && [[ "$py_minor" -eq "7" ]]; then
    echo "Installing pytorch for python 2.7"
    whl=~/".cache/pip/wheels/torch-0.3.1-cp27-cp27mu-linux_x86_64.whl"

    if [ ! -f "$whl" ]; then
        wget http://download.pytorch.org/whl/cpu/torch-0.3.1-cp27-cp27mu-linux_x86_64.whl -P ~/.cache/pip/wheels/
        wget http://download.pytorch.org/whl/cpu/torch-0.3.1-cp27-cp27m-linux_x86_64.whl -P ~/.cache/pip/wheels/
    fi

    pip install ~/.cache/pip/wheels/torch-0.3.1-cp27-cp27mu-linux_x86_64.whl || pip install ~/.cache/pip/wheels/torch-0.3.1-cp27-cp27m-linux_x86_64.whl


elif [[ "$py_major" = "(3" ]] && [[ "$py_minor" -eq "5" ]]; then
    echo "Installing pytorch for python 3.5"
    whl=~/".cache/pip/wheels/torch-0.3.1-cp35-cp35m-linux_x86_64.whl"

    if [ ! -f "$whl" ]; then
        wget http://download.pytorch.org/whl/cpu/torch-0.3.1-cp35-cp35m-linux_x86_64.whl -P ~/.cache/pip/wheels/
    fi

    pip install ~/.cache/pip/wheels/torch-0.3.1-cp35-cp35m-linux_x86_64.whl

elif [[ "$py_major" = "(3" ]] && [[ "$py_minor" -eq "6" ]]; then
    echo "Installing pytorch for python 3.6"
    whl=~/".cache/pip/wheels/torch-0.3.1-cp36-cp36m-linux_x86_64.whl"

    if [ ! -f "$whl" ]; then
        wget http://download.pytorch.org/whl/cpu/torch-0.3.1-cp36-cp36m-linux_x86_64.whl -P ~/.cache/pip/wheels/
    fi

    pip install ~/.cache/pip/wheels/torch-0.3.1-cp36-cp36m-linux_x86_64.whl

else
    echo "Unsupported python version"
    exit 1
fi

echo "Installed pytorch successfully"
