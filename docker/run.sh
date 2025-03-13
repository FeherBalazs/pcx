#!/bin/bash

set -ex

docker image build -t pcax:latest -f ./DockerfileGH200 ..
docker run --gpus all -it \
  -v /home/ubuntu/balazs/pcx:/home/pcax/workspace \
  pcax:latest /bin/bash
