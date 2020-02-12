FROM pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-devel

RUN apt-get -y update --fix-missing
RUN apt-get -y install nano

ARG PIP_EXTRA_INDEX_URL

RUN pip install sklearn Pillow tqdm matplotlib torchvision==0.4.0

WORKDIR .
COPY . .
