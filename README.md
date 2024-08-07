Use waymo2024_310 conda=

```sh
$ docker build . -t wod_v2_gpu -f .devcontainer/gpu.DOCKERFILE
$ docker run -it -u $(id -u):$(id -g) -p 8888:8888 -v /dev/shm:/dev/shm -v ./:/home/wod_v2/src -v /media/mike/Main\ Drive\ Ubunt/Documents/data/:/home/wod_v2/src/data -e DISPLAY=0 -e NVIDIA_VISIBLE_DEVICES=all --runtime=nvidia --env="DISPLAY" --net="host" --gpus all wod_v2_gpu
```

Use cranky_cannon for conda 

    $ sudo wget     https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh     &&    sudo mkdir /root/.conda     && bash Miniconda3-latest-Linux-x86_64.sh -b     && rm -f Miniconda3-latest-Linux-x86_64.sh
    $ . ../miniconda3/bin/activate
    $ conda create -n wod_v2 python=3.10
    $ conda activate wod_v2
    $ conda install nbclassic
    $ conda install tensorflow==2.12.0
    $ python3 -m pip install gcsfs waymo-open-dataset-tf-2-12-0==1.6.4
    $ python3 -m pip install "notebook>=5.3" "ipywidgets>=7.5"
    $ python3 -m pip install --upgrade "jupyter_http_over_ws>=0.0.7"
    $ jupyter serverextension enable --py jupyter_http_over_ws
    $ jupyter notebook

use magical_payne for base python3.8 -> python3.10
PATH="/home/wod_v2/.local/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"