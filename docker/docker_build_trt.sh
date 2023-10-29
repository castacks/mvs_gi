#! /bin/bash

# Required to build for aarch64
# docker run --rm --privileged multiarch/qemu-user-static --reset -p yes

docker image build -f docker/Dockerfile.trt -t jehontan/mvs-lightning:trt .