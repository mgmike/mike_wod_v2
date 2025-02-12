FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ARG USERNAME=mike_yolo
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Container for https://www.geeksforgeeks.org/yolov3-from-scratch-using-pytorch/

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    #
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

RUN apt-get update && apt-get install -y \
  git build-essential wget vim findutils curl \
  pkg-config zip g++ zlib1g-dev unzip python3-pip

# RUN apt-get update && apt-get install -y \
#   cmake git python-dev python-opencv \
#   libopencv-dev libav-tools libjpeg-dev libpng-dev \
#   libtiff-dev libjasper-dev libgtk2.0-dev \
#   python-numpy python-pycurl libatlas-base-dev \
#   gfortran webp python-opencv qt5-default \
#   libvtk6-dev golang

ENV SHELL=/bin/bash

# Install Open CV - Warning, this takes absolutely forever
# RUN mkdir -p ~/opencv cd ~/opencv && \
#     wget https://github.com/opencv/opencv/archive/3.0.0.zip && \
#     unzip 3.0.0.zip && \
#     rm 3.0.0.zip && \
#     mv opencv-3.0.0 OpenCV && \
#     cd OpenCV && \
#     mkdir build && \ 
#     cd build && \
#     cmake \
#     -DWITH_QT=ON \ 
#     -DWITH_OPENGL=ON \ 
#     -DFORCE_VTK=ON \
#     -DWITH_TBB=ON \
#     -DWITH_GDAL=ON \
#     -DWITH_XINE=ON \
#     -DBUILD_EXAMPLES=ON .. && \
#     make -j4 && \
#     make install && \ 
#     ldconfig

RUN python3 -m pip install "notebook>=5.3" "ipywidgets>=7.5"
RUN python3 -m pip install --upgrade "jupyter_http_over_ws>=0.0.7"
# RUN jupyter serverextension enable --py jupyter_http_over_ws


WORKDIR /home/mike_yolo/src

EXPOSE 8888

CMD ["bash", "-c", "source /etc/bash.bashrc && /bin/bash"]