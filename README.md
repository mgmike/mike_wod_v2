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

This one is fragile. Treat like a small helpless child. Used for reading waymo data and saving in a format that yolo can read

use magical_payne for base python3.8 -> python3.10
PATH="/home/wod_v2/.local/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"


Use pytorch dockerfile for my yolov3.
PATH="/home/mike_yolo/.local/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"


I had the idea to just use ultralytics yolo and modify that. Thats what ill try to do next
PATH="/home/mike_yolo_u/.local/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

My implementation of yolov11

All convolutional layers are completely duplicated. Then combined when flattened and fed into the fc layers. 

add 2 more boxes to the 7x7 output prediction vector for bev output
add bev to loss function

Eventually I want to implement some sort of attention and update the model using faster camera data

Okay sike, im actually going back to yolov4
Using this premade docker image
docker pull daisukekobayashi/darknet:darknet_yolo_v4_pre-gpu

docker run --runtime=nvidia --rm -v $PWD:/workspace -w /workspace daisukekobayashi/darknet:gpu darknet
PATH="/home/yolo/.local/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

Todo: 
- Save as jpeg too
- Save txt alongside jpeg
- Save cam and lidar txt alongside jpeg too