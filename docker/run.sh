#!/bin/bash

set -ex

docker image build -t pcax:latest -f ./DockerfileGH200 ..
docker run --gpus all -it pcax:latest /bin/bash
