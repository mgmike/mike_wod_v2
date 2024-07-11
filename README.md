Use waymo2024_310 conda=

```sh
$ docker build . -t wod_v2 -f .devcontainer/gpu.DOCKERFILE
$ docker run -it -u $(id -u):$(id -g) -p 8888:8888 -v /dev/shm:/dev/shm -v ./:/home/wod_v2/src -v /media/mike/Main\ Drive\ Ubunt/Documents/data/:/home/wod_v2/src/data -e DISPLAY=0 -e NVIDIA_VISIBLE_DEVICES=all --runtime=nvidia --env="DISPLAY" --net="host" --gpus all wod_v2
```