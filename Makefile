.PHONY: build dev

build:
	docker build -t large-sklearn-model .

dev:
	docker run  --rm --gpus all -ti -v $(PWD)/app:/app large-sklearn-model /bin/bash
gpu:
	docker run --rm --gpus all nvidia/cuda:11.0.3-base-ubuntu20.04 nvidia-smi