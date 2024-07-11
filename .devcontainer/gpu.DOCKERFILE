FROM tensorflow/tensorflow:2.12.0-gpu-jupyter

ARG USERNAME=wod_v2
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    #
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME
ENV SHELL /bin/bash

# RUN add-apt-repository ppa:deadsnakes/ppa
# RUN apt-get update && apt-get install -y \
#   python3.10 python3.10-venv python3.10-dev 

# RUN rm /usr/bin/python
# RUN rm /usr/bin/python3
# RUN ln -s python3.10 /usr/bin/python
# RUN ln -s python3.10 /usr/bin/python3

RUN apt-get update && apt-get install -y \
  git build-essential wget vim findutils curl \
  pkg-config zip g++ zlib1g-dev unzip  python3-pip

RUN apt-get install -y wget golang

# Install miniconda
# ENV PATH="/root/miniconda3/bin:${PATH}"
# ARG PATH="/root/miniconda3/bin:${PATH}"
# RUN apt-get update

# RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*

# Attempt 2
# RUN wget \
#     https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
#     && mkdir /root/.conda \
#     && bash Miniconda3-latest-Linux-x86_64.sh -b \
#     && rm -f Miniconda3-latest-Linux-x86_64.sh 
# RUN conda --version

# RUN conda create -n wod_v2 python=3.10
# RUN conda init
# RUN conda activate wod_v2

# Attempt 3
# FROM continuumio/miniconda3
# ADD environment.yml /tmp/environment.yml
# ADD requirements.txt /tmp/requirements.txt
# RUN conda env create -f /tmp/environment.yml
# RUN echo "source activate $(head -1 /tmp/environment.yml | cut -d' ' -f2)" > ~/.bashrc
# ENV PATH /opt/conda/envs/$(head -1 /tmp/environment.yml | cut -d' ' -f2)/bin:$PATH

# RUN conda init
# RUN conda activate $USERNAME

# RUN pip3 install "waymo-open-dataset-tf-2-12-0==1.6.4" jupyter gcsfs matplotlib &&\
#   pip3 install "notebook>=5.3" "ipywidgets>=7.5" &&\
#   jupyter serverextension enable --py jupyter_http_over_ws

# RUN python3 -m pip install --upgrade "jupyter_http_over_ws>=0.0.7"
# RUN python3 -m pip install --upgrade jupyterlab

WORKDIR /home/wod_v2/src
# COPY . .
# WORKDIR /home/wod_v2/src/data
# WORKDIR /home/wod_v2/src/


EXPOSE 8888
# RUN python3 -m ipykernel.kernelspec

CMD ["bash", "-c", "source /etc/bash.bashrc && /bin/bash"]
