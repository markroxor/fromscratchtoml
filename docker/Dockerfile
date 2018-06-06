FROM ubuntu:16.04

MAINTAINER Mohit Rathore mrmohitrathoremr@gmail.com

ENV REPOSITORY https://github.com/jellAIfish/fromscratchtoml.git
ENV BRANCH docker

RUN apt-get update
RUN apt-get install -y python-pip python3-pip git

# Upgrade pip
RUN pip2 install --upgrade pip
RUN pip3 install --upgrade pip

RUN git clone $REPOSITORY
RUN cd /fromscratchtoml \
&& git checkout $BRANCH \
&& pip2 install -r requirements.txt \
&& pip2 install jupyter \
&& pip3 install -r requirements.txt \
&& pip3 install jupyter \
&& python2 setup.py install \
&& python3 setup.py install

# Fix ipython kernel version
RUN ipython2 kernel install
RUN ipython3 kernel install

# Add running permission to startup script
RUN chmod +x /fromscratchtoml/docker/start_jupyter_notebook.sh

# Define the starting command for this container and expose its running port
CMD sh -c '/fromscratchtoml/docker/start_jupyter_notebook.sh 7777'
EXPOSE 7777
