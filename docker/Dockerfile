FROM pytorch/pytorch:1.2-cuda10.0-cudnn7-devel

RUN apt-get -y update --fix-missing
RUN apt-get -y install nano

ARG PIP_EXTRA_INDEX_URL

RUN pip install sklearn Pillow tqdm matplotlib

WORKDIR .
COPY . .
