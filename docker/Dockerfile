FROM ubuntu:18.04

MAINTAINER Mohit Rathore mrmohitrathoremr@gmail.com

ENV REPOSITORY https://github.com/jellAIfish/fromscratchtoml.git
ENV BRANCH master

RUN apt-get update
RUN apt-get install -y python3-pip git

# Upgrade pip
RUN pip3 install --upgrade pip

RUN git clone $REPOSITORY
WORKDIR /fromscratchtoml
RUN git checkout $BRANCH
RUN pip3 install -r requirements.txt
RUN pip3 install jupyter
RUN pip3 install scikit-learn==0.20.1
RUN python3 setup.py install

# Fix ipython kernel version
RUN ipython kernel install

# Add running permission to startup script
RUN chmod +x /fromscratchtoml/docker/start_jupyter_notebook.sh

# Define the starting command for this container and expose its running port
CMD sh - c 'flask run -h 0.0.0.0 -p 8080'
EXPOSE 8080
